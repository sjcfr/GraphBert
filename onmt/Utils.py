import pickle
import numpy as np
import pandas as pd
import random, json

import torch
from onmt.DataSet import *
from onmt.BertModules import *
from onmt.GraphBert import *
from onmt.GraphTransformer import *
from onmt.ktnet import ktnet, ktnet_baseline, bert_base, double_base_mix, ktnet_xlnet, roberta, xlnet
from onmt.ktnet import ktnet_roberta, ktnet_gpt2
import copy
import pdb
from transformers import RobertaModel, RobertaConfig, GPT2Model, GPT2Config, XLNetModel, XLNetConfig

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def load_examples(input_file):
    f = open(input_file, 'rb')
    try:
        examples = pickle.load(f)
    except:
        f.close()
        f = open(input_file, 'r')
        examples = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    f.close()
    return examples
    

def parse_gpuid(gpuls):
    ls = [int(n) for n in str(gpuls)]
    return ls
    
    
def parse_opt_to_name(opt):
    if 'base' in opt.bert_model:
        model_size = 'b'
    else:
        model_size = 'l'
    if opt.use_bert:
        model_type = "b"
    else:
        model_type = 'gb'
    if opt.pretrain:
        pretrain = 'p'
    else:
        pretrain = 'up'
    if opt.link_predict:
        graph_type = 'k'
    else:
        graph_type = 'e'
    start_layer = opt.start_layer
    merge_layer = opt.merge_layer
    n_layer_extractor = opt.n_layer_extractor
    n_layer_aa = opt.n_layer_aa
    n_layer_gnn = opt.n_layer_gnn
    n_layer_merger = opt.n_layer_merger
    method_extractor = opt.method_extractor[0]
    method_merger = opt.method_merger[0]
    smooth_term = opt.loss_aa_smooth
    smooth_method = opt.loss_aa_smooth_method[0]
    lr = opt.learning_rate
    warmup_proportion = opt.warmup_proportion
    margin = opt.do_margin_loss * opt.margin
    Lambda = opt.Lambda
    sep_sent = str(opt.sep_sent)[0]
    layer_norm = str(opt.layer_norm)[0]
    if 'match' in opt.test_data_dir:
        test_type = 'm'
    else:
        test_type = 'um' 
    name = [str(n) for n in [pretrain, graph_type, model_type, model_size, start_layer, merge_layer, n_layer_extractor, n_layer_aa, n_layer_gnn, n_layer_merger, smooth_term, smooth_method, 
                             Lambda, sep_sent, layer_norm, method_extractor, method_merger, lr, warmup_proportion, margin, test_type]]
    name = "_".join(name)
    return name
    
    
def sentence2ids(sentences, voc):

    def indexesFromSentence(voc, sentence):
        
        EOS_token = 1
        PAD_token = 2
        
        ids = []
        for word in sentence.split(' '):
            try:
                ids.append(voc.word2index[word])
            except:
                ids.append(2)
            
        ids.append(EOS_token)
        return ids

    ids = []
    for sentence in sentences:
        indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
        ids.append(indexes_batch)
        #lengths = [len(indexes) for indexes in indexes_batch]
        #input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        #_, encoder_hidden = graph_embedder(input_batch, lengths, None)
        #encoder_hidden = encoder_hidden.mean(0).unsqueeze(0)
        #embeddings.append(encoder_hidden)
      
    #embeddings = torch.cat(embeddings, axis=1)
    
    return ids


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training, 
                                 baseline=False, voc=None):
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
            context_tokens = context_tokens + ['.']

            sentence_ind_context.extend([ith_sent] * (len(sent_tokens) + 1))

        choices_features = []
        
        if_append = True
        
        for ending_index, ending in enumerate(example['candidates']):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            
            context_tokens_tmp = copy.deepcopy(context_tokens)
            sentence_ind_context_tmp = copy.deepcopy(sentence_ind_context)
            
            ending_tokens = tokenizer.tokenize(ending)
            ending_tokens = ending_tokens + ['.']
            
            sentence_ind_ending = [ith_sent + 1] * len(ending_tokens)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            
            #if example['ith'] == 4927 and ending_index == 3:
            #    pdb.set_trace()
            _truncate_seq_pair(context_tokens_tmp, ending_tokens, max_seq_length - 3)
            _truncate_seq_pair(sentence_ind_context_tmp, sentence_ind_ending, max_seq_length - 3)
            
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
            
            graph_embeddings = []
            
            sentences = example['context'] + [ending]
            
            if baseline:
                sentence_ids = sentence2ids(sentences, voc)
            
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
            
            if (set(sentence_ind) != {0, 1, 2, 3, 4, 5, 6, 7, 8} and set(sentence_ind) != {0, 1, 2, 3, 4, 5, 6, 7, 8, -1} \
               and set(sentence_ind) != {0, 1, 2, 3, 4} and set(sentence_ind) != {0, 1, 2, 3, 4, -1} and 'ask_for' not in example.keys()) or num_0 < 3:
               #and set(sentence_ind) != {0, 1} and set(sentence_ind) != {0, 1, -1}:
                if_append = False
                num_not_append += 1 
                print(num_not_append)
                #print(context_sentences)
                print("Too long example, id:", example_index)
            elif set(sentence_ind) != {0, 1} and set(sentence_ind) != {0, 1, -1} and 'ask_for' in example.keys():
                if_append = False
                print("Too long example, id:", example_index)
            
            if not baseline:
                choices_features.append((tokens, input_ids, input_mask, segment_ids, sentence_ind, graph))
            else:
                choices_features.append((tokens, input_ids, input_mask, segment_ids, sentence_ind, graph, sentence_ids))

        answer = [0] * len(example['candidates'])
        try:
            answer[example['ans']] = 1
        except:
            pdb.set_trace()
            answer[example['answer']] = 1
        
        if if_append:
            features.append(
                InputFeatures(
                    example_id = example['ith'],
                    choices_features = choices_features,
                    answer = answer
                )
            )

    return features
    
    
    
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
            tokens_a.pop()
            #tokens_a.pop(0)
        else:
            L = len(tokens_b)
            #r = random.randint(0, L - 1)
            tokens_b.pop()


def get_config(config):
    if config.trained_model is None:
        if 'roberta' in config.bert_model:
            pretrained_bert = RobertaConfig.from_pretrained(config.bert_model)
        elif 'gpt2' in config.bert_model:
            pretrained_bert = GPT2Config.from_pretrained(config.bert_model)
        else:
            pretrained_bert = XLNetConfig.from_pretrained(config.bert_model)

        graph_bert_config = pretrained_bert

        for k in dir(config):
            if "__" not in k:
                setattr(graph_bert_config, k, getattr(config, k))
    return graph_bert_config


def ini_from_pretrained_bert(config, graph_embedder=None, wordnet_embed=None, nell_embed=None):
    if config.trained_model is None:
        pretrained_bert = torch.load(config.bert_model)
        # state_dict = pretrained_bert.state_dict()

        state_dict = pretrained_bert
        tmp_config = json.load(open('/data/huggingface_transformers/bert-base-uncased/config.json', 'r', encoding='utf-8'))
        bert_config = BertConfig(config)
        for key, value in tmp_config.items():
            bert_config.__dict__[key] = value

        # bert_config = pretrained_bert.config

        graph_bert_config = bert_config
        for k in dir(config):
            if "__" not in k:
                setattr(graph_bert_config, k, getattr(config, k))
        model_config = BertConfig(graph_bert_config)

        if config.link_predict:
            model_config = config.link_predict

    else:
        model_config, state_dict = torch.load(config.trained_model)
        model_config.pretrain = config.pretrain

        # pdb.set_trace()

        if config.link_predict:
            model_config.link_predict = config.link_predict


    graph_bert_model = ktnet(config, model_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)

    old_keys = []
    new_keys = []

    for key in state_dict.keys():
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'layer' in key:
                new_key = new_key.replace('layer', 'bert_layers')

            # if 'pooler' in key:
            #    new_key = new_key.replace('bert.pooler', 'pooler')

            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    for name, parameter in graph_bert_model.state_dict().items():

        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)
    graph_bert_model.keys_bert_parameter = state_dict.keys()

    return graph_bert_model


def ini_from_pretrained_roberta(config, graph_embedder=None, wordnet_embed=None, nell_embed=None):
    if config.trained_model is None:
        pretrained_bert = RobertaModel.from_pretrained(config.bert_model)
        state_dict = pretrained_bert.state_dict()

        model_config = config

    else:
        model_config, state_dict = torch.load(config.trained_model)
        model_config.pretrain = config.pretrain

        if config.link_predict:
            model_config.link_predict = config.link_predict


    graph_bert_model = ktnet_roberta(config, model_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)
    # graph_bert_model = roberta(config, model_config)

    old_keys = []
    new_keys = []

    for key in state_dict.keys():
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'layer' in key:
                new_key = new_key.replace('layer', 'layers')

            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    for name, parameter in graph_bert_model.state_dict().items():

        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)

    return graph_bert_model


def ini_from_pretrained_gpt2(config, graph_embedder=None, wordnet_embed=None, nell_embed=None):

    pretrained_bert = GPT2Model.from_pretrained(config.bert_model)
    state_dict = pretrained_bert.state_dict()

    model_config = config

    graph_bert_model = ktnet_gpt2(config, model_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)

    old_keys = []
    new_keys = []

    for key in state_dict.keys():
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'h.' in key:
                new_key = new_key.replace('h.', 'encoder.layers.')

            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    for name, parameter in graph_bert_model.state_dict().items():

        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)

    return graph_bert_model


def ini_from_pretrained_xlnet(config, wordnet_embed=None, nell_embed=None):

    pretrained_bert = XLNetModel.from_pretrained(config.bert_model)
    state_dict = pretrained_bert.state_dict()

    model_config = config

    graph_bert_model = ktnet_xlnet(config, model_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)
    # graph_bert_model = xlnet(config, model_config)

    old_keys = []
    new_keys = []

    for key in state_dict.keys():
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'layer' in key:
                new_key = new_key.replace('layer.', 'encoder.layers.')

            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    for name, parameter in graph_bert_model.state_dict().items():

        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)

    return graph_bert_model



def mix_init(config, graph_embedder=None, wordnet_embed=None, nell_embed=None):
    if config.trained_model is None:
        pretrained_bert = torch.load(config.bert_model)
        # state_dict = pretrained_bert.state_dict()

        state_dict = pretrained_bert
        tmp_config = json.load(open('/data/huggingface_transformers/bert-base-uncased/config.json', 'r', encoding='utf-8'))
        bert_config = BertConfig(config)
        for key, value in tmp_config.items():
            bert_config.__dict__[key] = value

        # bert_config = pretrained_bert.config

        graph_bert_config = bert_config
        for k in dir(config):
            if "__" not in k:
                setattr(graph_bert_config, k, getattr(config, k))
        model_config = BertConfig(graph_bert_config)

        if config.link_predict:
            model_config = config.link_predict

    else:
        model_config, state_dict = torch.load(config.trained_model)
        model_config.pretrain = config.pretrain

        # pdb.set_trace()

        if config.link_predict:
            model_config.link_predict = config.link_predict


    graph_bert_model = double_base_mix(config, model_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)

    old_keys = []
    new_keys = []

    for key in state_dict.keys():
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'layer' in key:
                new_key = new_key.replace('layer', 'bert_layers')

            # if 'pooler' in key:
            #    new_key = new_key.replace('bert.pooler', 'pooler')

            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    for name, parameter in graph_bert_model.state_dict().items():

        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)
    graph_bert_model.keys_bert_parameter = state_dict.keys()

    return graph_bert_model



def ini_from_pretrained_bert_baseline(config, graph_embedder=None, wordnet_embed=None, nell_embed=None):
    if config.trained_model is None:
        pretrained_bert = torch.load(config.bert_model)
        # state_dict = pretrained_bert.state_dict()

        state_dict = pretrained_bert
        tmp_config = json.load(open('/data/huggingface_transformers/bert-base-uncased/config.json', 'r', encoding='utf-8'))
        bert_config = BertConfig(config)
        for key, value in tmp_config.items():
            bert_config.__dict__[key] = value

        # bert_config = pretrained_bert.config

        graph_bert_config = bert_config
        for k in dir(config):
            if "__" not in k:
                setattr(graph_bert_config, k, getattr(config, k))
        model_config = BertConfig(graph_bert_config)

        if config.link_predict:
            model_config = config.link_predict

    else:
        model_config, state_dict = torch.load(config.trained_model)
        model_config.pretrain = config.pretrain

        # pdb.set_trace()

        if config.link_predict:
            model_config.link_predict = config.link_predict


    graph_bert_model = ktnet_baseline(config, model_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)

    old_keys = []
    new_keys = []

    for key in state_dict.keys():
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'layer' in key:
                new_key = new_key.replace('layer', 'bert_layers')

            # if 'pooler' in key:
            #    new_key = new_key.replace('bert.pooler', 'pooler')

            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    #pdb.set_trace()
    for name, parameter in graph_bert_model.state_dict().items():

        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)
    graph_bert_model.keys_bert_parameter = state_dict.keys()

    return graph_bert_model


def ini_from_pretrained_bert_base(config):
    if config.trained_model is None:
        pretrained_bert = torch.load(config.bert_model)
        # state_dict = pretrained_bert.state_dict()

        state_dict = pretrained_bert
        tmp_config = json.load(open('/data/huggingface_transformers/bert-large-cased/config.json', 'r', encoding='utf-8'))
        bert_config = BertConfig(config)
        for key, value in tmp_config.items():
            bert_config.__dict__[key] = value

        # bert_config = pretrained_bert.config

        graph_bert_config = bert_config
        for k in dir(config):
            if "__" not in k:
                setattr(graph_bert_config, k, getattr(config, k))
        model_config = BertConfig(graph_bert_config)

        if config.link_predict:
            model_config = config.link_predict

    else:
        model_config, state_dict = torch.load(config.trained_model)
        model_config.pretrain = config.pretrain

        # pdb.set_trace()

        if config.link_predict:
            model_config.link_predict = config.link_predict


    graph_bert_model = bert_base(config, model_config)

    old_keys = []
    new_keys = []

    for key in state_dict.keys():
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'layer' in key:
                new_key = new_key.replace('layer', 'bert_layers')
                new_key = new_key.replace('encoder.', '')

            # if 'pooler' in key:
            #    new_key = new_key.replace('bert.pooler', 'pooler')

            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    for name, parameter in graph_bert_model.state_dict().items():

        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)
    graph_bert_model.keys_bert_parameter = state_dict.keys()

    return graph_bert_model


def ini_from_pretrained(config, graph_embedder=None):
    """
    Instantiate a the parameters of GraphBert from a pre-trained BERT model.
    Params:
        pretrained_model_path: 
            - path to a pretrained BERT model. Bert model should be in the list of:
                . nsp: `bert-base-uncased`
                . nsp: `bert-large-uncased`
        config: config for the GraphBert model.
            GraphBert contains four main additional structures: 
                . graph extractor
                . adjacancy approximator
                . GNN
                . merger
            We wish to control the structure of GraphBert only in the type and the number of layers for four main structures, 
            as well as the diverge and merge layer of BERT. 
            The size and number of heads for attention structure is the same as BERT model. 
    """
    if config.trained_model is None:
        pretrained_bert = torch.load(config.bert_model)
        # state_dict = pretrained_bert.state_dict()
        
        state_dict = pretrained_bert
        tmp_config = json.load(open('/data/huggingface_transformers/bert-large-cased/config.json', 'r', encoding='utf-8'))
        bert_config = BertConfig(config)
        for key, value in tmp_config.items():
            bert_config.__dict__[key] = value

        # bert_config = pretrained_bert.config 
        
        graph_bert_config = bert_config    
        for k in dir(config):
            if "__" not in k:
                setattr(graph_bert_config, k, getattr(config, k))
        model_config = BertConfig(graph_bert_config)
        if config.link_predict:
            model_config = config.link_predict
        
    else:
        model_config, state_dict = torch.load(config.trained_model)
        model_config.pretrain = config.pretrain
        
        #pdb.set_trace()

        if config.link_predict:
            model_config.link_predict = config.link_predict
    
    if not config.baseline:    
        graph_bert_model = GraphBertModel(model_config)
    else:
        graph_bert_model = GraphTransformerModel(model_config, graph_embedder)

    old_keys = []
    new_keys = []
    
    for key in state_dict.keys():
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'layer' in key:
                new_key = new_key.replace('layer', 'bert_layers')
            
            #if 'pooler' in key:
            #    new_key = new_key.replace('bert.pooler', 'pooler')
            
            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
        
    for name, parameter in graph_bert_model.state_dict().items():
        
        if name in state_dict.keys():
            try:
                bert_p = state_dict[name]
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            print(name)
    graph_bert_model.keys_bert_parameter = state_dict.keys()
    
    return graph_bert_model


def freeze_params(model, requires_grad=False):
    freeze_ls = ['graph_extractor',
                   'adjacancy_approximator',
                   'gnn',
                   'merger_layers']
    for module in freeze_ls:
        
        parameters = getattr(model.encoder, module) 
        for param in parameters.parameters():
            param.requires_grad = requires_grad


def accuracy(out, labels):
    pdb.set_trace()
    out = np.array(out)
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]
    
    
def loss_graph(appro_matrix, true_graph, loss_fn, smooth_term=0,  method='all'):
    if len(appro_matrix) == 2:
        
        graph_vector, graph_vector_post = appro_matrix
        assert graph_vector.shape == graph_vector_post.shape
        
        L = graph_vector.shape[1]
        
        loss_tot = 0
        
        for i in range(L):
            sigma = torch.ones_like(graph_vector[:,i,:])
            
            p = torch.distributions.Normal(graph_vector[:,i,:], sigma)
            q = torch.distributions.Normal(graph_vector_post[:,i,:], sigma)
            
            loss_tmp = loss_fn(p, q).sum()
            
            loss_tot = loss_tot + loss_tmp
            
    else:
        assert appro_matrix.shape == true_graph.shape
        
        L = appro_matrix.shape[1]
        loss_tot = 0
        if L < 10:
            for i in range(L):
                #pdb.set_trace()
                if method ==  'all':
                    #p = torch.distributions.beta.Beta(appro_matrix[:,i,:] + smooth_term, appro_matrix[:,i,:] + smooth_term)
                    #q = torch.distributions.beta.Beta(true_graph[:,i,:] + smooth_term, appro_matrix[:,i,:] + smooth_term)
                    
                    p = torch.distributions.categorical.Categorical(appro_matrix[:,i,:] + smooth_term)
                    q = torch.distributions.categorical.Categorical(true_graph[:,i,:] + smooth_term)
                    ##p = torch.distributions.categorical.Categorical(appro_matrix[:,i,:])
                    ##q = torch.distributions.categorical.Categorical(true_graph[:,i,:])
                else:
                    x = appro_matrix[:,i,:]
                    y = true_graph[:,i,:]
                    x[:, i] += 1
                    y[:, i] += 1
                    
                    p = torch.distributions.categorical.Categorical(x)
                    q = torch.distributions.categorical.Categorical(y)
                    
                loss_tmp = loss_fn(p, q).sum()
                '''
                if smooth_term  == 0:
                    loss_tmp = loss_fn(q, p).sum()
                else:
                    loss_tmp = loss_fn(p, q).sum()
                '''
                #loss_tmp = -(appro_matrix[:,i,:].log() * true_graph[:,i,:]).sum()
                loss_tot = loss_tot + loss_tmp
        else:
            loss_fn = torch.nn.MSELoss()
            
            loss_tot = loss_fn(appro_matrix, true_graph)
            
    return loss_tot
    
    
def write_result_to_file(args,result):
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(result+"\n")

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def do_evaluation(model,eval_dataloader, opt, gpu_ls=None,  output_res=False):

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all = None
    if gpu_ls:
        gpu_id = gpu_ls[0]
    else:
        gpu_id = opt.gpuid

    with torch.no_grad():
    
        #res = {'ids':[],'pred':[],'ans':[]}
        res = pd.DataFrame()
        for dat in eval_dataloader:
            dat = [t.cuda(gpu_id) for t in dat]
            if len(dat) == 6 and not opt.baseline:
                input_ids, input_masks, segment_ids, sentence_inds, graphs, answers = dat
            elif len(dat) == 6 and opt.baseline:
                input_ids, input_masks, segment_ids, sentence_inds, answers, graph_ids = dat
                graphs = None
            else:
                input_ids, input_masks, segment_ids, sentence_inds, answers = dat
                graphs = None
            
            answers = answers 
            num_choices = input_ids.shape[1]
            cls_score = []
            res_tmp = []
            for n in range(num_choices):
               
                input_ids_tmp = input_ids[:,n,:]
                input_masks_tmp = input_masks[:,n,:]
                segment_ids_tmp = segment_ids[:,n,:]
                sentence_inds_tmp = sentence_inds[:,n,:]
                answers_tmp = answers[:,n]
                
                if opt.baseline:
                    graph_ids_tmp = graph_ids[:,n,:]
                
                if graphs is not None:
                    graphs_tmp = graphs[:,n,:]
                else:
                    graphs_tmp = graphs                    
                graphs_tmp_scaled = graphs_tmp
                
                if not opt.use_bert and not opt.baseline:  
                    _, cls_score_tmp, attn_scores = model(input_ids = input_ids_tmp, 
                                                      token_type_ids = segment_ids_tmp, 
                                                      sentence_inds = sentence_inds_tmp, 
                                                      graphs = graphs_tmp_scaled) ##
                elif not opt.use_bert and opt.baseline: 
                    _, cls_score_tmp = model(input_ids = input_ids_tmp, 
                                                      graph_ids = graph_ids_tmp,
                                                      token_type_ids = segment_ids_tmp, 
                                                      sentence_inds = sentence_inds_tmp, 
                                                      graphs = graphs_tmp_scaled) ##
                else:
                    cls_score_tmp = model(input_ids = input_ids_tmp, token_type_ids = segment_ids_tmp) ##
                
                cls_score_tmp = cls_score_tmp.softmax(-1)
                cls_score.append(cls_score_tmp.detach().cpu().numpy()[:,1].tolist())
                
                res_tmp.append(cls_score_tmp.detach().cpu().numpy()[:,0].tolist())
                res_tmp.append(cls_score_tmp.detach().cpu().numpy()[:,1].tolist())
                
            #pdb.set_trace()
            cls_score = np.array(cls_score).T
            answers = answers.detach().cpu().numpy()
            num_acc_tmp = sum(cls_score.argmax(1) == answers.argmax(1))
            
            res_tmp.append(answers.argmax(1))
            res_tmp = pd.DataFrame(np.array(res_tmp).T)
            res = res.append(res_tmp)
            #print(num_acc_tmp)
            #eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += num_acc_tmp

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

    #eval_loss = eval_loss / nb_eval_steps
    
    eval_accuracy = float(eval_accuracy) / nb_eval_examples

    model.zero_grad()
    if not output_res:
        return eval_accuracy
    else:
        
        return eval_accuracy, res  


def graph_ids_to_tensor(all_graph_ids, opt):
    if 'mcnc' in opt.train_data_dir:
        max_L = 7
    elif 'roc' in opt.train_data_dir:
        max_L = 15
        
    PAD_token = 2
    all_graph_ids_padded = []
    for sample in all_graph_ids:
        all_graph_ids_padded.append([])
        for candidate in sample:
            all_graph_ids_padded[-1].append([])
            for sentence in candidate:
                l_sent = len(sentence[0])
                if l_sent > max_L:
                    sentence[0] = sentence[0][:max_L]
                elif l_sent < max_L:
                    l_diff = max_L - l_sent
                    pad_ls = [PAD_token] * l_diff
                    
                    sentence[0] = sentence[0] + pad_ls
                    #sentence = torch.LongTensor(sentence)
                all_graph_ids_padded[-1][-1].append(sentence)
             
    all_graph_ids_padded = torch.LongTensor(all_graph_ids_padded)
    all_graph_ids_padded = all_graph_ids_padded.squeeze()
    
    return all_graph_ids_padded
    
    
def retro(key, graph):
    key_expand = [key + "_obj", key + '_subj']
    res = []
    for key_tmp in key_expand:
        try:
            fwd_nodes_tmp = graph[key_tmp]
            res.append(fwd_nodes_tmp)
        except:
            pass
            
    return res
  
