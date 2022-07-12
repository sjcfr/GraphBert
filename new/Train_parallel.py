
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

sys.path.append("/users4/ldu/git_clones/apex/")
from apex import amp


parser = argparse.ArgumentParser(
    description='Train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# onmt.opts.py

onmt.Opt.model_opts(parser)
opt = parser.parse_args()

gpu_ls = parse_gpuid(opt.gpuls)

if 'large' in opt.bert_model:
    opt.train_batch_size = 16 * len(gpu_ls)
else:
    opt.train_batch_size = 32 * len(gpu_ls)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


os.makedirs(opt.output_dir, exist_ok=True)

train_examples = load_examples(os.path.join(opt.train_data_dir))
tokenizer = torch.load(opt.bert_tokenizer)
if opt.pret:
    train_features = convert_examples_to_features(
        train_examples, tokenizer, opt.max_seq_length, True)
    #pdb.set_trace()
else:
    train_features = train_examples
    
#dev_examples_all = load_examples(os.path.join(opt.dev_data_dir))
#dev_features_all = convert_examples_to_features(dev_examples_all, tokenizer, opt.max_seq_length, True)

test_examples_all = load_examples(os.path.join(opt.test_data_dir))
test_features_all = convert_examples_to_features(test_examples_all, tokenizer, opt.max_seq_length, True)

num_train_steps = int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs) * 5

# Prepare model
if not opt.use_bert:
    model = ini_from_pretrained(opt)
    
else:
    model = torch.load(opt.bert_model)

model_config = model.config

# Prepare optimizer
if opt.fp16:
    param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                        for n, param in model.named_parameters()]
elif opt.optimize_on_cpu:
    param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                        for n, param in model.named_parameters()]
else:
    param_optimizer = list(model.named_parameters())
    
#no_decay = ['bias', 'gamma', 'beta']
#no_decay = ['gamma', 'beta']
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': opt.l2_reg},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

t_total = num_train_steps
    
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=opt.learning_rate,
                     warmup=opt.warmup_proportion,
                     t_total=t_total)
model.cuda(gpu_ls[0])
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = nn.DataParallel(model,  device_ids=gpu_ls)

model.config = model_config

global_step = 0


logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", opt.train_batch_size)
logger.info("  Num steps = %d", num_train_steps)

train_dataloader = load_data(train_features, batch_size=opt.train_batch_size)

loss_nsp_fn = torch.nn.CrossEntropyLoss()
loss_aa_fn = torch.distributions.kl.kl_divergence
Lambda = opt.Lambda
loss_aa_smooth_term = opt.loss_aa_smooth
 
best_eval_acc = 0.0
best_test_acc = 0.0
best_step = 0

eval_acc_list = []
name = parse_opt_to_name(opt)
print(name)
time_start = str(int(time.time()))[-6:]


for epoch in range(int(opt.num_train_epochs)):
    print("Epoch:",epoch)
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    if True:
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.cuda(gpu_ls[0]) for t in batch)
            '''
            for both multiple choice problem and next sentence prediction, 
            the input is context and one of the choice. 
            '''
            example_ids, input_ids, input_masks, segment_ids, sentence_inds, graphs, answers = batch
            num_choices = input_ids.shape[1]
            
            #if opt.do_test:
            if True:
                if (step * opt.train_batch_size) % 4000 == 0:                
                    test_features = random.sample(test_features_all, 5000)
                    test_dataloader = load_data(test_features, batch_size=opt.train_batch_size, dat_type='test')
            
                    model = model.eval()
                    accurancy = do_evaluation(model,test_dataloader,opt,gpu_ls)
            
                    print('step:', step, "accurancy:", accurancy)
                    
                    model = model.train()
            else:
                accurancy = None
            
            for n in range(num_choices):
                input_ids_tmp = input_ids[:,n,:]
                input_masks_tmp = input_masks[:,n,:]
                segment_ids_tmp = segment_ids[:,n,:]
                sentence_inds_tmp = sentence_inds[:,n,:]
                graphs_tmp = graphs[:,n,:]
                answers_tmp = answers[:,n]
                
                graphs_tmp_scaled = graphs_tmp
                if not opt.use_bert:
                    
                    _, cls_scores, attn_scores,  = model(input_ids = input_ids_tmp, 
                                                          token_type_ids = segment_ids_tmp, 
                                                          sentence_inds = sentence_inds_tmp, 
                                                          graphs = graphs_tmp_scaled) ##
                                                          
                    #answers_tmp = -(answers_tmp * 2 - 1)
                    loss_nsp = loss_nsp_fn(cls_scores, answers_tmp)
                        
                    loss_aa = loss_graph(attn_scores, graphs_tmp, loss_aa_fn, smooth_term=loss_aa_smooth_term, distribution=opt.loss_aa_distrib)
                    loss = loss_nsp + Lambda * loss_aa
                
                    if step % 20 == 0:
                        print("step:", step, "loss_nsp:", loss_nsp.detach().cpu().numpy(), "loss_aa:",loss_aa.detach().cpu().numpy() *  Lambda)
                    
                    f = open('/users4/ldu/GraphBert/records/' + name + '_' + time_start + '.csv', 'a+')
                    f.write(str(loss_nsp.detach().cpu().numpy()) + ',' + str(Lambda * loss_aa.detach().cpu().numpy()) + ',' + str(accurancy) + '\n')
                    f.close()
                    
                else:
                    
                    cls_scores = model(input_ids = input_ids_tmp, token_type_ids = segment_ids_tmp) ##

                    #answers_tmp = -(answers_tmp * 2 - 1)
                    loss = loss_nsp_fn(cls_scores, answers_tmp)
                    
                    f = open('/users4/ldu/GraphBert/records/' + name + '_' + time_start + '.csv', 'a+')
                    f.write(str(loss.detach().cpu().numpy()) + ',' + str(accurancy) + '\n')
                    f.close()
                    
                    if step % 20 == 0:
                        print("step:", step, "loss:", loss.detach().cpu().numpy())

                if opt.fp16 and opt.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * opt.loss_scale
                if opt.gradient_accumulation_steps > 1:
                    loss = loss / opt.gradient_accumulation_steps
                loss.backward()
                
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
            if not opt.use_bert:
                ls = [model.config, model.state_dict()]
            else:
                ls = [model.config, model.state_dict()]
            
            #if (step * opt.train_batch_size) % 8000 == 0:
            #    torch.save(ls, "/users4/ldu/GraphBert/models/mcnc_" + str(step) + str(accurancy) + "e_" + str(epoch) + name + time_start + '.pkl')
            
        
