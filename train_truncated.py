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
from transformers import RobertaTokenizer

#sys.path.append("/users5/kxiong/software/apex/")
sys.path.append("/users4/ldu/git_clones/apex/")
from apex import amp

# torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(
    description='Train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# onmt.opts.py

onmt.Opt.model_opts(parser)
opt = parser.parse_args()
# opt.fp16 = True

#os.environ['CUDA_VISIBLE_DEVICES']="6,7"

gpu_ls = parse_gpuid(opt.gpuls)
#pdb.set_trace()
#n_gpu = torch.cuda.device_count()
#gpu_ls = [n for n in range(n_gpu)]

#opt.train_batch_size = 8 * len(n_gpu)
if 'large' in opt.bert_model:
    opt.train_batch_size = 10 * len(gpu_ls)
else:
    opt.train_batch_size = 32 * len(gpu_ls)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


# if os.path.exists(opt.output_dir) and os.listdir(opt.output_dir):
#     raise ValueError("Output directory ({}) already exists and is not empty.".format(opt.output_dir))
os.makedirs(opt.output_dir, exist_ok=True)


train_examples = None
eval_examples = None
eval_size= None
num_train_steps = None

train_examples = load_examples(os.path.join(opt.train_data_dir))

num_train_steps = int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs) * 5
    
# Prepare tokenizer
tokenizer = torch.load(opt.bert_tokenizer)

# Prepare model

if not opt.use_bert:
    model = ini_from_pretrained(opt)
    # model = GraphBertModel(opt)
else:
    model = torch.load(opt.bert_model)
#model = nn.DataParallel(model.cuda(),  device_ids=gpu_ls)
model_config = model.config

param_optimizer = list(model.named_parameters())

# no_decay = ['bias', 'gamma', 'beta']
# no_decay = ['gamma', 'beta']
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

model = nn.DataParallel(model,  device_ids=gpu_ls)
model.config = model_config
print("gpu ls:{}".format(gpu_ls))


global_step = 0

if opt.pret:
    train_features = convert_examples_to_features(
        train_examples, tokenizer, opt.max_seq_length, True)
    #pdb.set_trace()
else:
    train_features = train_examples

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", opt.train_batch_size)
logger.info("  Num steps = %d", num_train_steps)

all_example_ids = torch.tensor([train_feature.example_id for train_feature in train_features], dtype=torch.long)
all_input_tokens = select_field(train_features, 'tokens')
all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
all_input_masks = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
all_sentence_inds = torch.tensor(select_field(train_features, 'sentence_ind'), dtype=torch.long)

all_graphs = select_field(train_features, 'graph') ##
if all_graphs[0][0] is not None:
    all_graphs = torch.tensor(all_graphs, dtype=torch.float) ##

all_answers = torch.tensor([f.answer for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_example_ids, all_input_ids, all_input_masks, all_segment_ids, all_sentence_inds, all_graphs, all_answers)
if opt.local_rank == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
# train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.train_batch_size)

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
eval_acc_list=[]
# for _ in trange(int(opt.num_train_epochs), desc="Epoch"):
name = parse_opt_to_name(opt)
#name = '_rand_' + name
print(name)
time_start = str(int(time.time()))[-6:]

test_examples_all = load_examples(os.path.join(opt.test_data_dir))
test_features_ls = []

for i in range(len(test_examples_all)):
    test_features_ls.append(convert_examples_to_features(test_examples_all[i], tokenizer, opt.max_seq_length, True))

print('Done loading examples!')
accuracy = 0
for epoch in range(int(opt.num_train_epochs)):

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    
    if epoch > 0:
        accu_ls = []
        for i in range(len(test_features_ls)):
            test_features = test_features_ls[i]
            opt.test_batch_size = opt.train_batch_size
            all_example_ids = [train_feature.example_id for train_feature in train_features]
            all_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
            all_input_masks = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)
            all_sentence_inds = torch.tensor(select_field(test_features, 'sentence_ind'), dtype=torch.long)
            all_answers = torch.tensor([f.answer for f in test_features], dtype=torch.long)
            
            all_graphs = select_field(test_features, 'graph') ##
            
            if all_graphs[0][0] is not None:
                all_graphs = torch.tensor(all_graphs, dtype=torch.float) ##
                test_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_sentence_inds, all_graphs, all_answers)
            else:
                test_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_sentence_inds, all_answers)
            
            # Run prediction for full data
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=opt.eval_batch_size)
    
            model = model.eval()
            accuracy = do_evaluation(model,test_dataloader,opt,gpu_ls)
            accu_ls.append(accuracy)
    
            print('delete_num:', i, "accurancy:", accuracy)
            
            f = open('/users4/ldu/GraphBert/records/' + str(opt.use_bert)[0] + 'truncated' + '_' + time_start + '.csv', 'a+')
            f.write('delete_num:' + ',' + str(i) + ',' + str(accuracy) + '\n')
            f.close()
            
            model = model.train()

        ls = [model.config, model.state_dict()]
        torch.save(ls, "/users4/ldu/GraphBert/models/" + str(np.mean(accu_ls)) + "e_" + str(epoch) + name + time_start + '.pkl')

    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = tuple(t.cuda(gpu_ls[0]) for t in batch)
        '''
        for both multiple choice problem and next sentence prediction, 
        the input is context and one of the choice. 
        '''
        example_ids, input_ids, input_masks, segment_ids, sentence_inds, graphs, answers = batch
        num_choices = input_ids.shape[1]
        
        for n in range(num_choices):
            input_ids_tmp = input_ids[:,n,:]
            input_masks_tmp = input_masks[:,n,:]
            segment_ids_tmp = segment_ids[:,n,:]
            sentence_inds_tmp = sentence_inds[:,n,:]
            graphs_tmp = graphs[:,n,:]
            answers_tmp = answers[:,n]
            
            #print(example_ids)
            
            graphs_tmp_scaled = graphs_tmp
            if not opt.use_bert:
                
                _, cls_scores, attn_scores = model(input_ids = input_ids_tmp, 
                                                      token_type_ids = segment_ids_tmp, 
                                                      sentence_inds = sentence_inds_tmp, 
                                                      graphs = graphs_tmp_scaled) ##
                
                if not opt.do_margin_loss:
                    loss_nsp = loss_nsp_fn(cls_scores, answers_tmp)
                else:
                    #pdb.set_trace()
                    cls_scores = cls_scores.softmax(-1)
                    answers_tmp = answers_tmp.type(torch.FloatTensor).cuda(gpu_ls[0])
                    answers_tmp = -(answers_tmp * 2 - 1)
                    
                    loss_nsp = loss_nsp_fn(cls_scores[:,0], cls_scores[:,1], answers_tmp)
                attn_scores = attn_scores.squeeze(1)
                # print(attn_scores.size())
                # print(graphs_tmp.size())
                loss_aa = loss_graph(attn_scores, graphs_tmp, loss_aa_fn, smooth_term=loss_aa_smooth_term)
                loss = loss_nsp + Lambda * loss_aa
            
                if step % 20 == 0:
                    print("step:", step, "loss_nsp:", loss_nsp.detach().cpu().numpy(), "loss_aa:",loss_aa.detach().cpu().numpy() *  Lambda)
                
                f = open('/users4/ldu/GraphBert/records/' + str(opt.use_bert)[0] + 'truncated' + '_' + time_start + '.csv', 'a+')
                f.write(str(loss.detach().cpu().numpy()) + ',' + str(loss_aa.detach().cpu().numpy()) + '\n')
                f.close()
                
            else:
                #pdb.set_trace()
                model.zero_grad()
                cls_scores = model(input_ids = input_ids_tmp, token_type_ids = segment_ids_tmp) ##
                if not opt.do_margin_loss:
                    loss = loss_nsp_fn(cls_scores, answers_tmp)
                else:
                    
                    cls_scores = cls_scores.softmax(-1)
                    answers_tmp = answers_tmp.type(torch.FloatTensor).cuda(gpu_ls[0])
                    answers_tmp = -(answers_tmp * 2 - 1)
                    loss = loss_nsp_fn(cls_scores[:,0], cls_scores[:,1], answers_tmp)
                
                f = open('/users4/ldu/GraphBert/records/' + str(opt.use_bert)[0] + 'truncated' + '_' + time_start + '.csv', 'a+')
                f.write(str(loss.detach().cpu().numpy()) + '\n')
                f.close()
                
                if step % 20 == 0:
                    print("step:", step, "loss:", loss.detach().cpu().numpy())

            if opt.fp16 and opt.loss_scale != 1.0:
                # rescale loss for fp16 training
                # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                loss = loss * opt.loss_scale
            if opt.gradient_accumulation_steps > 1:
                loss = loss / opt.gradient_accumulation_steps
            #loss.backward()
            #torch.autograd.backward(loss)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # print(scaled_loss)
                scaled_loss.backward()
                # print("down scale loss")

            
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % opt.gradient_accumulation_steps == 0:
                if opt.fp16 or opt.optimize_on_cpu:
                    if opt.fp16 and opt.loss_scale != 1.0:
                        # scale down gradients for fp16 training
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad.data = param.grad.data / opt.loss_scale
                    is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                    if is_nan:
                        logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                        opt.loss_scale = opt.loss_scale / 2
                        model.zero_grad()
                        continue
                    optimizer.step()
                    copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                else:
                    optimizer.step()
                model.zero_grad()
                global_step += 1
        