import pandas as pd
import logging
import os, sys
import argparse
import random
from tqdm import tqdm, trange
import xml.etree.ElementTree as ET
from pprint import pprint
import random
import time
import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from onmt.BertModules import *
from onmt.GraphBert import *
from onmt.Utils import *
import onmt.Opt

sys.path.append("/users5/kxiong/software/apex/")
from apex import amp

parser = argparse.ArgumentParser(
    description='Train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

onmt.Opt.model_opts(parser)
opt = parser.parse_args()

gpu_ls = parse_gpuid(opt.gpuls)


# opt.train_batch_size = 8 * len(n_gpu)
if 'large' in opt.bert_model:
    opt.train_batch_size = 10 * len(gpu_ls)
else:
    opt.train_batch_size = 16 * len(gpu_ls)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

os.makedirs(opt.output_dir, exist_ok=True)

train_examples = None
eval_examples = None
eval_size = None
num_train_steps = None

train_examples = load_examples(os.path.join(opt.train_data_dir))

num_train_steps = int(
    len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs) * 5

# Prepare tokenizer
tokenizer = torch.load(opt.bert_tokenizer)

# Prepare model

if not opt.use_bert:
    model = ini_from_pretrained(opt)
else:
    model = torch.load(opt.bert_model)

model_config = model.config

param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': opt.l2_reg},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
]
t_total = num_train_steps
if opt.local_rank != -1:
    t_total = t_total // torch.distributed.get_world_size()

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=opt.learning_rate,
                     warmup=opt.warmup_proportion,
                     t_total=t_total)
model.cuda(gpu_ls[0])
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

model = nn.DataParallel(model, device_ids=gpu_ls)
model.config = model_config
print("gpu ls:{}".format(gpu_ls))




global_step = 0

if not opt.do_margin_loss:
    loss_nsp_fn = torch.nn.CrossEntropyLoss()
else:
    loss_nsp_fn = torch.nn.MarginRankingLoss(opt.margin)
loss_aa_fn = torch.distributions.kl.kl_divergence
Lambda = opt.Lambda
loss_aa_smooth_term = opt.loss_aa_smooth

best_eval_acc = 0.0
best_test_acc = 0.0
best_step = 0
eval_acc_list = []

if opt.pret:
    train_features = convert_examples_to_features(
        train_examples, tokenizer, opt.max_seq_length, True)
else:
    train_features = train_examples

test_examples_all = load_examples(os.path.join(opt.test_data_dir))
test_features_all = convert_examples_to_features(test_examples_all, tokenizer, opt.max_seq_length, True)


print('Done loading examples!')

all_example_ids = [train_feature.example_id for train_feature in train_features]

test_features = test_features_all
# test_features = random.sample(test_features_all, 5000)
opt.test_batch_size = opt.train_batch_size
# all_example_ids = [train_feature.example_id for train_feature in train_features]
all_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
all_input_masks = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
all_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)
all_sentence_inds = torch.tensor(select_field(test_features, 'sentence_ind'), dtype=torch.long)
all_answers = torch.tensor([f.answer for f in test_features], dtype=torch.long)

all_graphs = select_field(test_features, 'graph')  ##

if all_graphs[0][0] is not None:
    all_graphs = torch.tensor(all_graphs, dtype=torch.float)  ##
    test_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids,
                              all_sentence_inds, all_graphs, all_answers)
else:
    test_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids,
                              all_sentence_inds, all_answers)

# Run prediction for full data
# test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, batch_size=opt.eval_batch_size)

# opt.start_layer = int(start)
# opt.merge_layer = int(merge)
model_name = 'model_' + str(opt.start_layer) + '_' + str(opt.merge_layer) + '.pt'
model_name = os.path.join('./models/start_merge', model_name)


checkpoint = torch.load(model_name)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint["optimizer"])
amp.load_state_dict(checkpoint['amp'])

model = model.eval()
accurancy = do_evaluation(model, test_dataloader, opt, gpu_ls)

print("start-merge: {}-{}, test_accurancy:{}:".format(opt.start_layer, opt.merge_layer, accurancy))



