
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


def mask_tokens(inputs, tokenizer):
    inputs_0 = inputs.clone()
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    
    def get_special_tokens_mask(ls):
        is_special_tokens_mask = [1 if l==0 else 0 for l in ls]
        return is_special_tokens_mask
    
    special_tokens_mask = [get_special_tokens_mask(val) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    #inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    inputs[indices_replaced] = 0


    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    
    #pdb.set_trace()
    random_words = torch.randint(len(tokenizer.vocab), labels.shape, dtype=torch.long).cuda(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    #pdb.set_trace()
    return inputs, labels


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            L = len(tokens_a)
            #r = random.randint(0, L - 1)
            #tokens_a.pop()
            tokens_a.pop(0)
        else:
            L = len(tokens_b)
            #r = random.randint(0, L - 1)
            tokens_b.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # roc is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given roc example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    if 'graph' in examples[0].keys():
        has_graph = True
    else:
        has_graph = False
    num_not_append = 0
    for example_index, example in enumerate(examples):
        context_sentences = example['context']

        context_tokens = []
        sentence_ind_context = []
        for ith_sent, sent in enumerate(context_sentences):
            sent_tokens = tokenizer.tokenize(sent)
            context_tokens = context_tokens + sent_tokens
            #context_tokens = context_tokens + ['.']

            sentence_ind_context.extend([ith_sent] * len(sent_tokens))

        choices_features = []
        
        if_append = True
        
        for ending_index, ending in enumerate(example['candidates']):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            
            context_tokens_tmp = copy.deepcopy(context_tokens)
            sentence_ind_context_tmp = copy.deepcopy(sentence_ind_context)
            
            ending_tokens = tokenizer.tokenize(ending)
            #ending_tokens = ending_tokens + ['.']
            
            sentence_ind_ending = [ith_sent + 1] * len(ending_tokens)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            l0 = len(context_tokens_tmp) + len( ending_tokens)
            _truncate_seq_pair(context_tokens_tmp, ending_tokens, max_seq_length - 3)
            _truncate_seq_pair(sentence_ind_context_tmp, sentence_ind_ending, max_seq_length - 3)
            l1 = len(context_tokens_tmp) + len( ending_tokens)
                            
            if 'ask_for' in example.keys():
                if example['ask_for'] == 'cause':
                    tokens = ["[CLS]"] + ending_tokens + ["[SEP]"] + context_tokens_tmp + ["[SEP]"]
                else:
                    tokens = ["[CLS]"] + context_tokens_tmp + ["[SEP]"] + ending_tokens + ["[SEP]"]
            else:
                tokens = ["[CLS]"] + context_tokens_tmp + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_tmp) + 2) + [1] * (len(ending_tokens) + 1)
            
            sentence_ind_context_tmp.insert(0, 0)
            sentence_ind_context_tmp.append(ith_sent)
            sentence_ind_ending.append(ith_sent + 1)
            
            sentence_ind = sentence_ind_context_tmp + sentence_ind_ending
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            sentence_ind += [p-1 for p in padding]
            
            if has_graph:
                graph = example['graph'][ending_index]
            else:
                graph = None
            
            try:
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(sentence_ind) == max_seq_length
            except:
                pdb.set_trace()
            num_0 = len([1 for i in sentence_ind if i == 0])
            
            if (set(sentence_ind) != {0, 1, 2, 3, 4} and set(sentence_ind) != {0, 1, 2, 3, 4, -1} and set(sentence_ind) != {0, 1} and set(sentence_ind) != {0, 1, -1}) or num_0 < 3:
                if_append = False
                num_not_append += 1 
                print(num_not_append)
                #print(context_sentences)
                print("Too long example, id:", example_index)
                #pdb.set_trace()
            
            choices_features.append((tokens, input_ids, input_mask, segment_ids, sentence_ind, graph))

        answer = [0] * len(example['candidates'])
        try:
            answer[example['ans']] = 1
        except:
            try:
                answer[example['answer']] = 1
            except:
                answer[0] = example['ans']
            
        if if_append:
            features.append(
                InputFeatures(
                    example_id = example['ith'],
                    choices_features = choices_features,
                    answer = answer
                )
            )

    return features


parser = argparse.ArgumentParser(
    description='Train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# onmt.opts.py

onmt.Opt.model_opts(parser)
opt = parser.parse_args()


gpu_ls = parse_gpuid(opt.gpuls)

if 'large' in opt.bert_model:
    opt.train_batch_size = 10 * len(gpu_ls)
else:
    opt.train_batch_size = opt.train_batch_size * len(gpu_ls)

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

#pdb.set_trace()

num_train_steps = int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs) * 2
    
# Prepare tokenizer
tokenizer = torch.load(opt.bert_tokenizer)

# Prepare model

if not opt.use_bert:
    model = ini_from_pretrained(opt)
    #model = GraphBertModel(opt)
else:
    model = torch.load(opt.bert_model)
#model = nn.DataParallel(model.cuda(),  device_ids=gpu_ls)
model_config = model.config
model = nn.DataParallel(model,  device_ids=gpu_ls)
model.config = model_config
model.cuda(gpu_ls[0])

'''
if opt.do_margin_loss==1:
    model = BertForMultipleChoiceMarginLoss.from_pretrained(opt.bert_model,cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(opt.local_rank),num_choices = 2, margin=opt.margin)
'''


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
if opt.local_rank != -1:
    t_total = t_total // torch.distributed.get_world_size()
    
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=opt.learning_rate,
                     warmup=opt.warmup_proportion,
                     t_total=t_total)
# optimizer = adabound.AdaBound(optimizer_grouped_parameters, lr=opt.learning_rate, final_lr=0.1)

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
#all_example_ids = None
all_input_tokens = select_field(train_features, 'tokens')
all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
all_input_masks = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
all_sentence_inds = torch.tensor(select_field(train_features, 'sentence_ind'), dtype=torch.long)
all_answers = torch.tensor([f.answer for f in train_features], dtype=torch.long)
all_graphs = select_field(train_features, 'graph') ##

#pdb.set_trace()
if all_graphs[0][0] is not None:
    all_graphs = torch.tensor(all_graphs, dtype=torch.float) ##
    train_data = TensorDataset(all_example_ids, all_input_ids, all_input_masks, all_segment_ids, all_sentence_inds, all_graphs, all_answers)
    with_graph = True
else:
    train_data = TensorDataset(all_example_ids, all_input_ids, all_input_masks, all_segment_ids, all_sentence_inds, all_answers)
    with_graph = False


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
 
best_eval_acc=0.0
best_test_acc=0.0
best_step=0
eval_acc_list=[]

name = parse_opt_to_name(opt)
name = name + str(opt.train_batch_size) + '_' + str(opt.max_seq_length)
print(name)
time_start = str(int(time.time()))[-6:]

test_examples = load_examples(os.path.join(opt.test_data_dir))
test_features = convert_examples_to_features(test_examples, tokenizer, opt.max_seq_length, True)
#test_features = convert_examples_to_features(test_examples, tokenizer, 200, True)
num_save_res = 0

loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)


for epoch in range(opt.num_train_epochs):
    print("Epoch:",epoch)
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    #freeze_params(model, requires_grad=True)
    #if epoch < opt.num_frozen_epochs:
    #freeze_params(model, requires_grad=True)
    '''
    for step, batch in enumerate(train_dataloader):
        example_ids, input_ids, input_masks, segment_ids, sentence_inds, graphs, answers = batch
        if step % 20 == 0:
            print(step)
            print(example_ids)
    '''    
    if True:
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.cuda(gpu_ls[0]) for t in batch)

            if with_graph:
                example_ids, input_ids, input_masks, segment_ids, sentence_inds, graphs, answers = batch
            else:
                example_ids, input_ids, input_masks, segment_ids, sentence_inds, answers = batch
            num_choices = input_ids.shape[1]
            
            #if opt.do_test:
            if not opt.pretrain:
                
                L = len(train_examples)
                if L < 10000:
                    freq_eva = 160
                    #name_pre = 'e'
                else:
                    freq_eva = 4000
                    #name_pre = 'a'
                
                if opt.trained_model:
                    name_pre = 'f'
                else:
                    name_pre = 't'
                
                if (step * opt.train_batch_size) % freq_eva == 0:                
                    
                    opt.test_batch_size = opt.train_batch_size
                    all_example_ids = [test_feature.example_id for test_feature in test_features]
                    #all_example_ids = None
                    all_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
                    all_input_masks = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
                    #pdb.set_trace()
                    all_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)
                    all_sentence_inds = torch.tensor(select_field(test_features, 'sentence_ind'), dtype=torch.long)
                    all_answers = torch.tensor([f.answer for f in test_features], dtype=torch.long)
                    
                    all_graphs = select_field(test_features, 'graph') ##

                    test_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_sentence_inds, all_answers)
                    # Run prediction for full data
                    test_sampler = SequentialSampler(test_data)
                    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=opt.eval_batch_size)
            
                    model = model.eval()
                    accurancy, res = do_evaluation(model, test_dataloader,opt,gpu_ls, True)
                    
                    if accurancy > 0.85:
                        
                        res.to_csv('/users4/ldu/GraphBert/records_roc/predicts/' + name_pre + name + '_' + time_start + '_' + str(num_save_res) + '.csv')
                        num_save_res += 1
            
                    print('step:', step, "accurancy:", accurancy)
                    
                    model = model.train()
            else:
                accurancy = None
            
            for n in range(num_choices):
                input_ids_tmp = input_ids[:,n,:]
                input_masks_tmp = input_masks[:,n,:]
                segment_ids_tmp = segment_ids[:,n,:]
                sentence_inds_tmp = sentence_inds[:,n,:]
                answers_tmp = answers[:,n]
                if with_graph:
                    graphs_tmp = graphs[:,n,:]
                    if not opt.link_predict:
                        graphs_tmp_scaled = graphs_tmp / (graphs_tmp.sum(2).unsqueeze(2))
                    else:
                        graphs_tmp_scaled = graphs_tmp
                else:
                    graphs_tmp_scaled = None
                if not opt.use_bert:
                    if not opt.pretrain:
                        
                        _, cls_scores, attn_scores = model(input_ids = input_ids_tmp, 
                                                              token_type_ids = segment_ids_tmp, 
                                                              sentence_inds = sentence_inds_tmp, 
                                                              graphs = graphs_tmp_scaled) ##
                        
                        if not opt.do_margin_loss:
                            loss_nsp = loss_nsp_fn(cls_scores, answers_tmp)
                        else:
                            cls_scores = cls_scores.softmax(-1)
                            answers_tmp = answers_tmp.type(torch.FloatTensor).cuda(gpu_ls[0])
                            answers_tmp = -(answers_tmp * 2 - 1)
                            loss_nsp = loss_nsp_fn(cls_scores[:,0], cls_scores[:,1], answers_tmp)
                            
                        if not with_graph:
                            if (step * opt.train_batch_size) % 20 == 0:
                                print("step:", step, "loss:", loss_nsp.detach().cpu().numpy())
                            f = open('/users4/ldu/GraphBert/records_roc/' + name_pre + name + '_' + time_start + '.csv', 'a+')
                            if step == 0 and epoch == 0:
                                f.write(str(opt.trained_model) + '\n')
                            f.write(str(loss_nsp.detach().cpu().numpy()) + ',' +  str(accurancy) + '\n')
                            f.close()
                            loss = loss_nsp                                
                        else:
                            #pdb.set_trace()
                            loss_aa = loss_graph(attn_scores, graphs_tmp, loss_aa_fn, smooth_term=loss_aa_smooth_term)
                            loss = loss_nsp + Lambda * loss_aa
                            if (step * opt.train_batch_size) % 20 == 0:
                                print("step:", step, "loss_nsp:", loss_nsp.detach().cpu().numpy(), "loss_aa:",loss_aa.detach().cpu().numpy() *  Lambda)
                            f = open('/users4/ldu/GraphBert/records_roc/' + name_pre + name + '_' + time_start + '.csv', 'a+')
                            f.write(str(loss_nsp.detach().cpu().numpy()) + ',' + str(Lambda * loss_aa.detach().cpu().numpy()) + ',' + str(accurancy) + '\n')
                            f.close()
                    else:
                        input_ids_tmp_msk, labels = mask_tokens(input_ids_tmp, tokenizer)
                        
                        _,  pred_tokens, attn_scores = model(input_ids = input_ids_tmp_msk, 
                                                              token_type_ids = segment_ids_tmp, 
                                                              sentence_inds = sentence_inds_tmp, 
                                                              graphs = graphs_tmp_scaled) ##
                        #pdb.set_trace()
                        masked_lm_loss = sum(loss_fct(pred_tokens[:,i,:], input_ids_tmp_msk[:,i]) for i in range(pred_tokens.shape[1])) 
                        #loss_fct(pred_tokens.view(-1, model.config.vocab_size), input_ids_tmp_msk.view(-1))  
                                                                                    
                        loss_aa = loss_graph(attn_scores, graphs_tmp_scaled, loss_aa_fn, smooth_term=loss_aa_smooth_term)
                        loss = masked_lm_loss + Lambda * loss_aa
 
                        if (step * opt.train_batch_size) % 20 == 0:
                            print("step:", step, "loss_mlm:", masked_lm_loss.detach().cpu().numpy(), "loss_aa:",loss_aa.detach().cpu().numpy() *  Lambda)
                        f = open('/users4/ldu/GraphBert/records_roc/' + 'roc_pretrain_' + name + '_' + time_start + '.csv', 'a+')
                        f.write(str(masked_lm_loss.detach().cpu().numpy()) + ',' + str(Lambda * loss_aa.detach().cpu().numpy()) + '\n')
                        f.close()
                        
                        if str(masked_lm_loss.detach().cpu().numpy()) == 'nan':
                            pdb.set_trace()
                        
                else:
                    #pdb.set_trace()
                    cls_scores = model(input_ids = input_ids_tmp, token_type_ids = segment_ids_tmp) ##
                    if not opt.do_margin_loss:
                        loss = loss_nsp_fn(cls_scores, answers_tmp)
                    else:
                        
                        cls_scores = cls_scores.softmax(-1)
                        answers_tmp = answers_tmp.type(torch.FloatTensor).cuda(gpu_ls[0])
                        answers_tmp = -(answers_tmp * 2 - 1)
                        loss = loss_nsp_fn(cls_scores[:,0], cls_scores[:,1], answers_tmp)
                    
                    f = open('/users4/ldu/GraphBert/records_roc/' + name_pre + name + '_' + time_start + '.csv', 'a+')
                    f.write(str(loss.detach().cpu().numpy()) + ',' + str(accurancy) + '\n')
                    f.close()
                    
                    if step % 20 == 0:
                        print("step:", step, "loss:", loss.detach().cpu().numpy())
                #pdb.set_trace()
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
            #if (step * opt.train_batch_size) % 800 == 0 and not opt.pretrain:
            #    torch.save(ls, "/users4/ldu/GraphBert/models/roc/roc_" + name_pre + str(accurancy) + "e_" + str(epoch) + name + time_start + '.pkl')

    '''
    if not opt.use_bert:
        ls = [model.config, model.state_dict()]
    else:
        ls = [model.config, model.state_dict()]
    #if epoch == 0 or epoch == 1:
    torch.save(ls, "/users4/ldu/GraphBert/models/roc_" + name_pre + str(accurancy) + "e_" + str(epoch) + name + time_start + '.pkl')
    '''
    if opt.pretrain:
        if epoch == 0 or epoch == 1:
            ls = [model.config, model.state_dict()]
            torch.save(ls, "/users4/ldu/GraphBert/models/roc_pretrain/roc_pretrain_" + str(loss.detach().cpu().numpy()) + "e_" + str(epoch) + name + time_start + '.pkl')


