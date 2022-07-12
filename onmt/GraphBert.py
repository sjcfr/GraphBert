import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from onmt.BertModules import *

import pdb        

class GATSelfOutput(nn.Module):
    def __init__(self, config):
        super(GATSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = config.layer_norm
        self.act_fn = ACT2FN[config.act_fn_branch]
        if self.layer_norm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        try:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        except:
            self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        if self.layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

        
        

class GATLayer(nn.Module):
     def __init__(self, config):
         super(GATLayer, self).__init__()
         
         self.method = config.method
         print(config.method)
         if self.method == 'self':
             self.attn_layer = BertSelfAttention(config)
         elif self.method == 'cross':
             self.attn_layer = BertCrossAttention(config)
         else:
             pdb.set_trace()
             
         self.output = GATSelfOutput(config)
         
     def forward(self, graph_vectors, context_vectors=None, attention_scores=None, drop_first_token=True, sent_ind=None, attention_mask=None):
         
         if isinstance(self.attn_layer, BertSelfAttention):
             graph_vectors = self.attn_layer(graph_vectors, attention_probs=attention_scores)
         elif isinstance(self.attn_layer, BertCrossAttention):
             graph_vectors = self.attn_layer(graph_vectors, context_vectors, drop_first_token=drop_first_token, sent_ind=sent_ind, attention_mask=attention_mask)
             
         graph_vectors = self.output(graph_vectors)
         
         return graph_vectors



class GraphExtractor(nn.Module):
    def __init__(self, config):
        super(GraphExtractor, self).__init__()
        
        self.num_layers = config.n_layer_extractor
        
        self.config = copy.deepcopy(config)
        self.config.method = config.method_extractor
        self.config.layer_norm = config.layer_norm
        self.is_link_prediction = config.link_predict
        
        layer = GATLayer(self.config)
            
        self.extract_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.config.n_layer_extractor)])
        
    def forward(self, sent_ind, start_layer, subsequent_layers=None):
        '''
        start_layer: tensor with shape batch * seq_length * dim
                       seq_length = number of tokens
        subsequent_layers: list, each element is a tensor with shape batch * seq_length * dim
        graph_vectors: tensor with shape batch * seq_length * dim
                       seq_length = number of sentences
        '''
        assert len(subsequent_layers) == self.num_layers
        
        batch = sent_ind.shape[0]
        
        if not self.is_link_prediction:
            num_sent = sent_ind.max().detach().cpu().numpy() + 1
            
            graph_vectors = []
            for n in range(batch):
                graph_vectors_sample = []
                for ith_sent in range(num_sent):
                    if ith_sent != 0:
                        graph_vectors_sample_ith = start_layer[n][sent_ind[n] == ith_sent].mean(0).unsqueeze(0)
                    else:
                        '''
                        The first [CLS] token should not be taken into consideration of the first sentence.
                        '''
                        graph_vectors_sample_ith = start_layer[n][sent_ind[n] == ith_sent][1:].mean(0).unsqueeze(0)
                    graph_vectors_sample.append(graph_vectors_sample_ith)
                try:
                    graph_vectors_sample = torch.cat(graph_vectors_sample).unsqueeze(0)
                except:
                    pdb.set_trace()
                graph_vectors.append(graph_vectors_sample)
            graph_vectors = torch.cat(graph_vectors)
            
            for i in range(self.num_layers):
                graph_vectors = self.extract_layers[i](graph_vectors, subsequent_layers[i],  sent_ind=sent_ind)                
        else:
            #pdb.set_trace()
            graph_vectors = start_layer
            for i in range(self.num_layers):
                graph_vectors = self.extract_layers[i](graph_vectors, subsequent_layers[i])
        
        return graph_vectors
    
    
class AdjacancyApproximator(nn.Module):
    def __init__(self, config):
        super(AdjacancyApproximator, self).__init__()
        
        self.num_layers = config.n_layer_aa - 1
        
        self.config = copy.deepcopy(config)
        self.config.method = 'self'
        self.config.layer_norm = config.layer_norm
        #pdb.set_trace()
        layer = GATLayer(self.config)
        layer.attn_drop = False
        
        self.aa_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_layers)])
        #self.config_final_layer = copy.deepcopy(config)
        self.final_layer = BertSelfAttention(config)
        self.final_layer.output_attentions = True
        self.final_layer.unmix = True
        self.final_layer.attn_drop = False
        
    def forward(self, graph_vectors):

        for i in range(self.num_layers):
            graph_vectors = self.aa_layers[i](graph_vectors) 
        attn_scores, graph_vectors = self.final_layer(graph_vectors)  
        #attn_scores, _ = self.final_layer(graph_vectors)   
        
        return attn_scores, graph_vectors
    
    
class GNN(nn.Module):
    def __init__(self, config):
        super(GNN, self).__init__()
        
        self.config = copy.deepcopy(config)
        #self.num_layers = config.n_layer_gnn - 1
        #temporarily not knowing why minus 1.
        self.num_layers = config.n_layer_gnn
        self.config.layer_norm = config.layer_norm
        
        if config.method_gnn == 'gat':
            self.config.method = 'self'
            layer = GATLayer(self.config)
            layer.attn_drop = False
            self.gnn_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_layers)])
        elif config.method_gnn == 'ggnn':
            raise NotInplementedError
            
    def forward(self, attn_scores, graph_vectors):
        
        if self.config.method_gnn == 'gat':
            graph_vectors = self.gnn_layers[0](graph_vectors, attention_scores=attn_scores)
            for i in range(1, self.num_layers, 1):
                graph_vectors = self.gnn_layers[i](graph_vectors)
        
        elif self.config.method == 'ggnn':
            '''
            for i in range(self.num_layers):
                graph_vectors = self.gnn_layers[i](attn_scores, graph_vectors)
            '''
            raise NotInplementedError    
        return graph_vectors        
    
    
class GATResMergerLayer(nn.Module):
    def __init__(self, config):
        super(GATResMergerLayer, self).__init__()
        self.config = copy.deepcopy(config)
        
        self.config.layer_norm = config.layer_norm
        config.method = 'cross'
        config.layer_norm = False  # !!
        layer = GATLayer(config)
        self.layer = layer
        self.layer.output_attentions = True
        
        if self.config.layer_norm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, context_vectors, graph_vectors, sent_ind, attention_mask=None):
        
        context_vectors_updated = self.layer(context_vectors, graph_vectors, sent_ind, attention_mask=attention_mask, drop_first_token=False)
        if self.config.layer_norm:
            context_vectors_updated = BertLayerNorm(context_vectors + context_vectors_updated)
        else:
            context_vectors_updated = context_vectors + context_vectors_updated
            
        return context_vectors_updated

class AddMergerLayer(nn.Module):
    def __init__(self, config):
        '''
        The res link of add merger is meaningless, since there is not learnable parameters in add merger.
        The layer norm operation is also meaningless.
        '''
        self.config = copy.deepcopy(config)
        
    def forward(self, context_vectors, graph_vectors, sent_ind):

        graph_vectors_spanned = graph_vectors, sent_ind ##
        
        context_vectors_updated = context_vectors + graph_vectors_spanned
            
        return context_vectors_updated
        

class GraphBertEncoder(nn.Module):
    def __init__(self, config, output_attentions, keep_multihead_output):
        super(GraphBertEncoder, self).__init__()

        self.config = config
        self.output_attentions = output_attentions
        
        bert_layer = BertLayer(config, output_attentions=output_attentions,
                                  keep_multihead_output=keep_multihead_output)
        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(config.num_hidden_layers)])
        
        self.graph_extractor = GraphExtractor(config)
        
        self.adjacancy_approximator = AdjacancyApproximator(config)
        
        self.gnn = GNN(config)
        
        self.post_converter = GATSelfOutput(config) # !!!
        
        if config.method_merger == 'gat':
            merger_layer = GATResMergerLayer(config)
            self.merger_layers = nn.ModuleList([merger_layer for _ in range(config.n_layer_merger)])
        elif config.method_merger == 'add':
            merger_layer = AddMergerLayer(config)
            self.merger_layers = nn.ModuleList([merger_layer for _ in range(config.n_layer_merger)])
        
        
    def forward(self, hidden_states, attention_mask, sentence_ind, true_adjacancy_matrix=None, output_all_encoded_layers=True, head_mask=None):
        all_encoder_layers = []
        all_attentions = []
        
        '''
        denote the start layer as sl, merge layer as ml. sl and ml starts from 0. 
        Therefore: 
        '''
        start_layer = self.config.start_layer - 1
        merge_layer = self.config.merge_layer - 1
        
        num_tot_layers = len(self.bert_layers)
        num_sub_layers = self.config.n_layer_extractor
        num_merger_layers = self.config.n_layer_merger
        
        assert start_layer + num_sub_layers <= merge_layer
        
        def append(hidden_states):
            if self.output_attentions:
                attentions, hidden_states = hidden_states
                all_attentions.append(attentions)
            #if output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        for i in range(merge_layer):
            hidden_states = self.bert_layers[i](hidden_states, attention_mask, head_mask[i])
            append(hidden_states)
        #pdb.set_trace()
        context_vector_start = all_encoder_layers[start_layer]
        context_vector_subsequent = all_encoder_layers[(start_layer + 1): (start_layer + 1 + num_sub_layers)]
        
        graph_vectors = self.graph_extractor(sentence_ind, context_vector_start, context_vector_subsequent)
        
        #attn_scores, graph_vectors = self.adjacancy_approximator(graph_vectors)
        attn_scores, _ = self.adjacancy_approximator(graph_vectors)
        
        if true_adjacancy_matrix is not None:
            true_adjacancy_matrix = true_adjacancy_matrix.unsqueeze(1)
            #graph_vectors = self.gnn(true_adjacancy_matrix, graph_vectors) 
            true_adjacancy_matrix = true_adjacancy_matrix.squeeze()
            graph_vectors_post = torch.matmul(true_adjacancy_matrix, graph_vectors) # !!!
            graph_vectors_post = self.post_converter(graph_vectors_post)
            #pdb.set_trace()
            
            graph_vectors = self.gnn(attn_scores, graph_vectors)
            true_adjacancy_matrix = true_adjacancy_matrix.squeeze() 

        else:
            graph_vectors = self.gnn(attn_scores, graph_vectors) 
            
        
        for ith_merger_layer, jth_bert_layer in zip(list(range(num_merger_layers)), list(range(merge_layer, merge_layer + num_merger_layers))):
            if self.config.method_merger != 'combine':
                hidden_states = self.merger_layers[ith_merger_layer](hidden_states, graph_vectors, sentence_ind)
                hidden_states = self.bert_layers[jth_bert_layer](hidden_states, attention_mask, head_mask[jth_bert_layer])
            else:
                #pdb.set_trace()
                attention_mask_tmp = torch.cat([attention_mask, attention_mask[:, :, :, :(graph_vectors.shape[1] + 0)] * 0], -1) 
                hidden_states = torch.cat([hidden_states, graph_vectors], 1)
                hidden_states = self.bert_layers[jth_bert_layer](hidden_states, attention_mask_tmp, head_mask[jth_bert_layer])
                hidden_states = hidden_states[:, :-(graph_vectors.shape[1] + 0), :]
                
            append(hidden_states)
        
        for j in range(merge_layer + num_merger_layers, num_tot_layers):
            hidden_states = self.bert_layers[j](hidden_states, attention_mask, head_mask[j])
            append(hidden_states)
        '''
        if not output_all_encoded_layers:
            all_encoder_layers = all_encoder_layers[-1]
        '''
        if self.output_attentions:
            #return all_attentions, all_encoder_layers, attn_scores
            if true_adjacancy_matrix is not None:
                # return all_attentions, all_encoder_layers, (graph_vectors, graph_vectors_post) # !!!
                return all_attentions, all_encoder_layers, attn_scores
            else:
                # return all_attentions, all_encoder_layers, (graph_vectors, graph_vectors) # !!!
                return all_attentions, all_encoder_layers, attn_scores
        else:
            #return all_encoder_layers, attn_scores
            if true_adjacancy_matrix is not None:
                # return all_encoder_layers, (graph_vectors, graph_vectors_post) # !!!
                return all_encoder_layers, attn_scores
            else:
                # return all_encoder_layers, (graph_vectors, graph_vectors)
                return all_encoder_layers, attn_scores


class GraphBertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(GraphBertModel, self).__init__(config)
        self.config = config
        self.output_attentions = output_attentions
        self.embeddings = BertEmbeddings(config)
        self.encoder = GraphBertEncoder(config, output_attentions=output_attentions,
                                           keep_multihead_output=keep_multihead_output)
        self.is_pretrain = config.pretrain
        if not self.is_pretrain:
            self.pooler = BertPooler(config)
            self.cls = BertOnlyNSPHead(config)
            self.apply(self.init_bert_weights)
        else:
            self.lm_head = BertLMPredictionHead(config, self.embeddings.word_embeddings.weight)

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
    def forward(self, input_ids, sentence_inds=None, graphs=None, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, head_mask=None):
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
        
        encoded_layers = self.encoder(hidden_states = embedding_output,
                                      attention_mask = extended_attention_mask,
                                      sentence_ind = sentence_inds, 
                                      true_adjacancy_matrix = graphs,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers, attn_scores = encoded_layers
        else:
            encoded_layers, attn_scores = encoded_layers
        #attn_scores = attn_scores.squeeze(1) # # !!!
        sequence_output = encoded_layers[-1]
        
        if self.is_pretrain:
            predictions_lm = self.lm_head(sequence_output)
            return encoded_layers, predictions_lm, attn_scores
        else:
            pooled_output = self.pooler(sequence_output)
            cls_scores = self.cls(pooled_output)
            
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            if self.output_attentions:
                return all_attentions, encoded_layers, cls_scores, attn_scores
            else:
                return encoded_layers, cls_scores, attn_scores
      