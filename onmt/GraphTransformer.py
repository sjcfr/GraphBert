import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from onmt.BertModules import *
from onmt.GraphBert import *


import pdb

class GATResMergerLayer(nn.Module):
    def __init__(self, config):
        super(GATResMergerLayer, self).__init__()
        self.config = copy.deepcopy(config)
        
        self.config.layer_norm = config.layer_norm
        config.method = 'cross'
        #config.layer_norm = False  # !!
        #config.query_size = 128
        
        layer = GATLayer(config)
        self.layer = layer
        self.layer.output_attentions = True
        
        if self.config.layer_norm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, context_vectors, graph_vectors, sent_ind):
        
        context_vectors_updated = self.layer(context_vectors, graph_vectors, sent_ind, drop_first_token=False)
        if self.config.layer_norm:
            context_vectors_updated = self.LayerNorm(context_vectors + context_vectors_updated)
        else:
            context_vectors_updated = context_vectors + context_vectors_updated
            
        return context_vectors_updated


class GraphTransformerEncoder(nn.Module):
    def __init__(self, config, output_attentions,keep_multihead_output):
        super(GraphTransformerEncoder, self).__init__()

        self.config = config
        self.output_attentions = output_attentions
        
        bert_layer = BertLayer(config, output_attentions=output_attentions,
                                  keep_multihead_output=keep_multihead_output)
        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(config.num_hidden_layers)])
        
        config_convert_layer = copy.deepcopy(config)
        config_convert_layer.hidden_size = 128 
        config_convert_layer.num_attention_heads = 8
       
        self.attn_convert_layer = BertSelfAttention(config_convert_layer)
        self.lm_convert_layer = nn.Linear(128, 768)
        self.merger_layer = copy.deepcopy(bert_layer)
        
        
    def forward(self, hidden_states, graph_embeddings, attention_mask, sentence_ind, output_all_encoded_layers=True, head_mask=None):
        all_encoder_layers = []
        all_attentions = []
        
        '''
        denote the start layer as sl, merge layer as ml. sl and ml starts from 0. 
        Therefore: 
        '''
        merge_layer = self.config.merge_layer - 1
        
        num_tot_layers = len(self.bert_layers)
        num_sub_layers = self.config.n_layer_extractor
        num_merger_layers = self.config.n_layer_merger
        
        #assert start_layer + num_sub_layers <= merge_layer
        
        def append(hidden_states):
            if self.output_attentions:
                attentions, hidden_states = hidden_states
                all_attentions.append(attentions)
            #if output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        for i in range(merge_layer):
            hidden_states = self.bert_layers[i](hidden_states, attention_mask, head_mask[i])
            append(hidden_states)
            
        #print(graph_embeddings.shape[0])
        graph_embeddings = self.attn_convert_layer(graph_embeddings)
        graph_embeddings = torch.tanh(self.lm_convert_layer(graph_embeddings))
        
        num_sent = graph_embeddings.shape[1]
        try:
            hidden_states = torch.cat([hidden_states, graph_embeddings],  axis=1)
        except:
            pdb.set_trace()
        
        #pdb.set_trace()
        if merge_layer != 1:
            # method1
            hidden_states = self.merger_layer(hidden_states,  attention_mask, head_mask[0])[:,:-num_sent,:] + hidden_states[:,:-num_sent,:]
            append(hidden_states)
            
            for j in range(merge_layer, num_tot_layers):
                hidden_states = self.bert_layers[j](hidden_states, attention_mask, head_mask[j])
                append(hidden_states)
        else:
            # method2
            #hidden_states = self.bert_layers[merge_layer](hidden_states,  attention_mask, head_mask[0])[:,:-num_sent,:] + hidden_states[:,:-num_sent,:]
            hidden_states = self.bert_layers[merge_layer](hidden_states,  attention_mask, head_mask[0])[:,:-num_sent,:]
            append(hidden_states)
            
            for j in range(merge_layer + 1, num_tot_layers):
                hidden_states = self.bert_layers[j](hidden_states, attention_mask, head_mask[j])
                append(hidden_states)    
            
        '''
        if not output_all_encoded_layers:
            all_encoder_layers = all_encoder_layers[-1]
        '''
        if self.output_attentions:
            return all_attentions, all_encoder_layers
        else:
            return all_encoder_layers
            
            

class GraphTransformerModel(BertPreTrainedModel):

    def __init__(self, config, graph_embedder, output_attentions=False, keep_multihead_output=False):
        super(GraphTransformerModel, self).__init__(config)
        self.config = config
        self.output_attentions = output_attentions
        self.embeddings = BertEmbeddings(config)
        self.encoder = GraphTransformerEncoder(config, output_attentions=output_attentions,
                                           keep_multihead_output=keep_multihead_output)
        self.graph_embedder = graph_embedder
        self.pooler = BertPooler(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        """ Gather all multi-head outputs.
            Return: list (layers) of multihead module outputs with gradients
        """
        return [layer.attention.self.multihead_output for layer in self.encoder.layer]
    ##
        
    def forward(self, input_ids, graph_ids, sentence_inds=None, graphs=None, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        batch_size = graph_ids.shape[0]
        if 'mcnc' in self.config.train_data_dir:
            lengths = torch.LongTensor([7] * batch_size)
        elif 'roc' in self.config.train_data_dir:
            lengths = torch.LongTensor([15] * batch_size)
        
        graph_embeddings = []
        
        for i in range(graph_ids.shape[1]):  
            try:          
                _, graph_embedding = self.graph_embedder(graph_ids[:,i].transpose(0, 1), lengths)
            except:
                pdb.set_trace()
            graph_embedding = graph_embedding.mean(0).unsqueeze(0)
            graph_embeddings.append(graph_embedding)
            
        graph_embeddings = torch.cat(graph_embeddings, axis=0).transpose(0, 1)
        
        encoded_layers = self.encoder(hidden_states = embedding_output,
                                      graph_embeddings = graph_embeddings,
                                      attention_mask = extended_attention_mask,
                                      sentence_ind = sentence_inds, 
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers
        else:
            encoded_layers = encoded_layers
        sequence_output = encoded_layers[-1]
        
        pooled_output = self.pooler(sequence_output)
        cls_scores = self.cls(pooled_output)
        
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.output_attentions:
            return all_attentions, encoded_layers, cls_scores
        else:
            return encoded_layers, cls_scores

