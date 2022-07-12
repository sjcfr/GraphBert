import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from transformers.models import roberta
from onmt.BertModules import *
from transformers.models.gpt2.modeling_gpt2 import *
from transformers.models.roberta.modeling_roberta import *
from transformers.models.xlnet.modeling_xlnet import *
import pdb, numpy
from onmt.GraphBert import GATResMergerLayer
from torch.utils.checkpoint import checkpoint


class knowledge_integration(nn.Module):
    def __init__(self, config, bert_config, content_embedding):
        super(knowledge_integration, self).__init__()
        self.config = config
        self.bert_config = bert_config
        self.linear = nn.Linear(config.content_dim, bert_config.hidden_size)
        # self.sentinel = nn.Parameter(torch.randn(1, config.content_dim), requires_grad=True)
        self.content_embedding = nn.Embedding.from_pretrained(content_embedding, freeze=config.freeze)
        self.softmax = nn.Softmax(2)
        self.linear2 = nn.Linear(bert_config.hidden_size, config.content_dim)

        # self.slices = torch.ones(self.config.batch_size, self.config.seq_len, 1, 1)

    def forward(self, bert_output, content_ids, content_mask):
        if self.config.model_name == 'xlnet':
            bert_output = bert_output.transpose(0, 1)
        batch_size = bert_output.size()[0]
        contents = self.content_embedding(content_ids)[:, self.config.seq_len-bert_output.size(1):, :, :]  # batch_size * sqe_len * num_content * content_dim

        if self.config.model_name == 'roberta':
            sentinel = bert_output[:, 0:1, :]
            # sentinel = self.linear2(sentinel)
        else:
            sentinel = bert_output[:, -2:-1, :]
            # sentinel = contents[:, :, 0:1, :]

        sentinel = self.linear2(sentinel)
        sentinel = sentinel.expand((batch_size, bert_output.size(1), self.config.content_dim))
        content_all = torch.cat((contents, sentinel.unsqueeze(2)), 2)  # batch_size * sqe_len * num_content+1 * content_dim
        content_trans = self.linear(content_all)

        bert_content = torch.unsqueeze(bert_output, 3)
        if self.config.model_name == 'roberta':
            atten_score = torch.matmul(content_trans, bert_content)
        else:
            # atten_score = torch.relu(torch.matmul(content_trans, bert_content))
            atten_score = torch.matmul(content_trans, bert_content)

        tmp_content_mask = (1.0 - content_mask) * -10000.0
        tmp_content_mask = tmp_content_mask.to(atten_score.dtype)
        content_mask = content_mask.to(atten_score.dtype)

        atten_score = torch.squeeze(atten_score, 3)
        # atten_score = torch.mul(atten_score, content_mask)
        atten_score = atten_score + tmp_content_mask

        atten_weight = self.softmax(atten_score)  # batch_size * sqe_len * num_content+1
        atten_weight = torch.mul(atten_weight, content_mask)
        atten_weight = torch.unsqueeze(atten_weight, 2)
        output = torch.matmul(atten_weight, content_all)
        output = torch.squeeze(output, 2)

        return output


class self_matching(nn.Module):
    def __init__(self, config, bert_config):
        super(self_matching, self).__init__()
        self.config = config
        self.bert_config = bert_config
        if config.wordnet and config.nell:
            self.linear = nn.Linear(3 * (config.content_dim * 2 + bert_config.hidden_size), 1)
        else:
            self.linear = nn.Linear(3 * (config.content_dim + bert_config.hidden_size), 1)
        self.softmax = nn.Softmax(2)

    def forward(self, integration_output, attention_mask):
        batch_size = integration_output.size(0)
        seq_len = integration_output.size(1)
        if self.config.wordnet and self.config.nell:
            embed_dim = self.config.content_dim * 2 + self.bert_config.hidden_size
        else:
            embed_dim = self.config.content_dim + self.bert_config.hidden_size

        integration = torch.unsqueeze(integration_output, 2)
        integration2 = torch.unsqueeze(integration_output, 1)

        expand = integration.expand(batch_size, seq_len, seq_len, embed_dim)
        expand2 = integration2.expand(batch_size, seq_len, seq_len, embed_dim)

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(integration_output.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # attention_mask = torch.unsqueeze(attention_mask, 1)
        # attention_mask = attention_mask.expand(batch_size, self.config.seq_len, self.config.seq_len)
        # attention_mask = attention_mask.to(integration_output.dtype)

        element_wise_mul = torch.mul(expand, expand2)
        final_expand = torch.cat((expand, expand2, element_wise_mul), 3)

        if self.config.model_name == 'roberta':
            atten_score = self.linear(final_expand)
        else:
            atten_score = self.linear(final_expand)
            # atten_score = self.linear(torch.tanh(final_expand))
        atten_score = torch.squeeze(atten_score, 3)

        # atten_score = torch.mul(atten_score, attention_mask)
        atten_score = atten_score + extended_attention_mask
        atten_weight = self.softmax(atten_score)

        V1 = torch.matmul(atten_weight, integration_output)

        atten_weight2 = torch.matmul(atten_weight, atten_weight)
        # atten_weight2 = torch.mul(atten_weight2, attention_mask)

        V2 = torch.matmul(atten_weight2, integration_output)
        # V1,V2: batch_size * seq_len * content_dim + hidden_size

        UV = torch.mul(integration_output, V1)
        final_representation = torch.cat(
            (integration_output, V1, integration_output - V1, UV, V2, integration_output - V2), 2)

        return final_representation


class merge(nn.Module):
    def __init__(self, config, bert_config):
        super(merge, self).__init__()
        self.config = config
        if config.wordnet and config.nell:
            self.linear = nn.Linear(6 * (config.content_dim * 2 + bert_config.hidden_size), bert_config.hidden_size)
        else:
            self.linear = nn.Linear(6 * (config.content_dim + bert_config.hidden_size), bert_config.hidden_size)
        self.multihead_atten = GATResMergerLayer(bert_config)

    def forward(self, match_output, bert_output, attention_mask=None):
        
        match_output = self.linear(match_output)
        if self.config.model_name == 'xlnet':
            bert_output = bert_output.transpose(0, 1)

        merge_output = self.multihead_atten(bert_output, match_output, sent_ind=None, attention_mask=attention_mask)

        if self.config.model_name == 'xlnet':
            merge_output = merge_output.transpose(0, 1)
        return merge_output


class ktnet_encoder(nn.Module):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(ktnet_encoder, self).__init__()
        bert_layer = BertLayer(config, False, False)
        # bert_layer = bert_ignore_last_arg(config)
        self.config = config
        self.bert_config = bert_config
        if config.wordnet and config.nell:
            self.integrate1 = knowledge_integration(config, bert_config, wordnet_embed)
            self.integrate2 = knowledge_integration(config, bert_config, nell_embed)
        elif config.wordnet:
            self.integrate = knowledge_integration(config, bert_config, wordnet_embed)
        elif config.nell:
            self.integrate = knowledge_integration(config, bert_config, nell_embed)
        else:
            raise ValueError('at least one of Wordnet and Nell must be True')

        self.match = self_matching(config, bert_config)
        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(config.num_hidden_layers)])
        self.merge = merge(config, bert_config)

    def forward(self, input_embedding, attention_mask, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None):
        all_hidden_layers = []
        start_layer = self.config.start_layer - 1
        merge_layer = self.config.merge_layer - 1
        num_layers = len(self.bert_layers)


        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(input_embedding.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        hidden_state = input_embedding
        for i in range(merge_layer):
            hidden_state = self.bert_layers[i](hidden_state, extended_attention_mask)
            all_hidden_layers.append(hidden_state)

        start_layer_output = all_hidden_layers[start_layer]

        if self.config.wordnet and self.config.nell:
            integration1 = self.integrate1(start_layer_output, wordnet_content_id, wn_mask)
            integration2 = self.integrate2(start_layer_output, nell_content_id, ne_mask)

            integration = torch.cat((start_layer_output, integration1, integration2), 2)

        elif self.config.wordnet:
            integration = self.integrate(start_layer_output, wordnet_content_id, wn_mask)
            integration = torch.cat((start_layer_output, integration), 2)

        else:
            integration = self.integrate(start_layer_output, nell_content_id, ne_mask)
            integration = torch.cat((start_layer_output, integration), 2)

        matching = self.match(integration, attention_mask)

        merge_output = self.merge(matching, all_hidden_layers[-1], attention_mask=extended_attention_mask)

        hidden_state = merge_output
        for j in range(merge_layer, num_layers):
            hidden_state = self.bert_layers[j](hidden_state, extended_attention_mask)
        return hidden_state


class ktnet_encoder_gpt2_roberta(nn.Module):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(ktnet_encoder_gpt2_roberta, self).__init__()
        self.config = config
        if config.model_name == 'gpt2':
            self.layers = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.n_layer)])
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        elif config.model_name == 'roberta':
            roberta_layer = RobertaLayer(config)
            self.layers = nn.ModuleList([copy.deepcopy(roberta_layer) for _ in range(config.num_layers)])
        else:
            xlnet_layer = XLNetLayer(config)
            self.layers = nn.ModuleList([copy.deepcopy(xlnet_layer) for _ in range(config.num_layers)])
        
        if config.wordnet and config.nell:
            self.integrate1 = knowledge_integration(config, bert_config, wordnet_embed)
            self.integrate2 = knowledge_integration(config, bert_config, nell_embed)
        elif config.wordnet:
            self.integrate = knowledge_integration(config, bert_config, wordnet_embed)
        elif config.nell:
            self.integrate = knowledge_integration(config, bert_config, nell_embed)
        else:
            raise ValueError('at least one of Wordnet and Nell must be True')
        
        self.match  = self_matching(config, bert_config)
        self.merge = merge(config, bert_config)

    def forward(self, input_embedding, attn_mask, attention_mask=None, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None, pos_emb=None, non_tgt_mask=None):   
        all_hidden_layers = []
        start_layer = self.config.start_layer - 1
        merge_layer = self.config.merge_layer - 1
        num_layers = len(self.layers)

        if self.config.model_name == 'roberta':
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(input_embedding.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(input_embedding.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        hidden_state = input_embedding
        for i in range(merge_layer):
            if self.config.model_name == 'xlnet':
                hidden_state = self.layers[i](hidden_state, None, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask, r=pos_emb, seg_mat=None)[0]
            else:
                hidden_state = self.layers[i](hidden_state, attention_mask=extended_attention_mask)[0]
                if self.config.model_name == 'gpt2':
                    hidden_state = self.layer_norm(hidden_state)
            all_hidden_layers.append(hidden_state)

        start_layer_output = all_hidden_layers[start_layer]

        if self.config.wordnet and self.config.nell:
            integration1 = self.integrate1(start_layer_output, wordnet_content_id, wn_mask)
            integration2 = self.integrate2(start_layer_output, nell_content_id, ne_mask)
            if self.config.model_name == 'xlnet':
                integration = torch.cat((start_layer_output.transpose(0, 1), integration1, integration2), 2)
            else:
                integration = torch.cat((start_layer_output, integration1, integration2), 2)

        elif self.config.wordnet:
            integration = self.integrate(start_layer_output, wordnet_content_id, wn_mask)
            if self.config.model_name == 'xlnet':
                integration = torch.cat((start_layer_output.transpose(0, 1), integration), 2)
            else:
                integration = torch.cat((start_layer_output, integration), 2)

        else:
            integration = self.integrate(start_layer_output, nell_content_id, ne_mask)
            integration = torch.cat((start_layer_output, integration), 2)

        matching = self.match(integration, attention_mask)

        merge_output = self.merge(matching, all_hidden_layers[-1], attention_mask=extended_attention_mask)

        hidden_state = merge_output
        for j in range(merge_layer, num_layers):
            if self.config.model_name == 'xlnet':
                hidden_state = self.layers[j](hidden_state, None, attn_mask_h=non_tgt_mask, attn_mask_g=attention_mask, r=pos_emb, seg_mat=None)[0]
            else:
                hidden_state = self.layers[j](hidden_state, attention_mask=extended_attention_mask)[0]
                if self.config.model_name == 'gpt2':
                    hidden_state = self.layer_norm(hidden_state)
        return hidden_state.transpose(0, 1) if self.config.model_name == 'xlnet' else hidden_state


class output_layer(nn.Module):
    def __init__(self, config, bert_config):
        super(output_layer, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(bert_config.hidden_size, 1)
        self.linear2 = nn.Linear(bert_config.hidden_size, 1)


    def forward(self, bert_output, attention_mask=None):
        
        # attention_mask = (1.0 - attention_mask) * -10000.0
        # attention_mask = attention_mask.to(bert_output.dtype)

        pos1 = self.linear1(bert_output).squeeze(2)
        pos2 = self.linear2(bert_output).squeeze(2)

        return pos1, pos2


class ktnet(BertPreTrainedModel):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(ktnet, self).__init__(config)
        self.config = config
        self.bert_config = bert_config
        self.embeddings = BertEmbeddings(bert_config)
        # self.embeddings = embedding_ignore_last_arg(embeddings)

        self.encoder = ktnet_encoder(config, bert_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)
        self.output = output_layer(config, bert_config)

    def forward(self, input_ids, attention_mask, seg_ids, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None):

        embedding_output = self.embeddings(input_ids, token_type_ids=seg_ids)

        encoder = self.encoder(embedding_output, attention_mask, wordnet_content_id=wordnet_content_id,
                               nell_content_id=nell_content_id, wn_mask=wn_mask, ne_mask=ne_mask)

        start_pos, end_pos = self.output(encoder, attention_mask)

        return start_pos, end_pos


class ktnet_gpt2(GPT2PreTrainedModel):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(ktnet_gpt2, self).__init__(config)
        self.config = config
        self.bert_config = bert_config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)

        self.encoder = ktnet_encoder_gpt2_roberta(config, bert_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)
        self.output = output_layer(config, bert_config)

    def forward(self, input_ids, attention_mask, seg_ids, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None):

        input_shape = input_ids.size()
        past_length = 0
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        # embedding_output = self.embeddings(input_ids, token_type_ids=seg_ids)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        embedding_output = inputs_embeds + position_embeds

        encoder = self.encoder(embedding_output, attention_mask, wordnet_content_id=wordnet_content_id,
                               nell_content_id=nell_content_id, wn_mask=wn_mask, ne_mask=ne_mask)

        start_pos, end_pos = self.output(encoder, attention_mask)

        return start_pos, end_pos


class ktnet_roberta(RobertaPreTrainedModel):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(ktnet_roberta, self).__init__(config)
        self.config = config
        self.bert_config = bert_config
        # self.embeddings = BertEmbeddings(bert_config)
        self.embeddings = RobertaEmbeddings(bert_config)
        # self.embeddings = embedding_ignore_last_arg(embeddings)

        self.encoder = ktnet_encoder_gpt2_roberta(config, bert_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)
        self.output = output_layer(config, bert_config)

    def forward(self, input_ids, attention_mask, seg_ids, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None):

        embedding_output = self.embeddings(input_ids)

        encoder = self.encoder(embedding_output, None, attention_mask=attention_mask, wordnet_content_id=wordnet_content_id,
                               nell_content_id=nell_content_id, wn_mask=wn_mask, ne_mask=ne_mask)

        start_pos, end_pos = self.output(encoder, attention_mask)

        return start_pos, end_pos


class ktnet_xlnet(XLNetPreTrainedModel):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(ktnet_xlnet, self).__init__(config)
        self.config = config
        self.bert_config = bert_config
        # self.embeddings = BertEmbeddings(bert_config)
        # self.embeddings = RobertaEmbeddings(bert_config)
        # self.embeddings = embedding_ignore_last_arg(embeddings)
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

        self.encoder = ktnet_encoder_gpt2_roberta(config, bert_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)
        self.output = output_layer(config, bert_config)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None, device=None, dtype=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.config.d_model, 2.0, dtype=dtype)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.config.d_model))

        beg, end = klen, -qlen

        fwd_pos_seq = torch.arange(beg, end, -1.0)
        if self.config.clamp_len > 0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.config.clamp_len, self.config.clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(device)
        return pos_emb

    def forward(self, input_ids, attention_mask, seg_ids, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None):

        input_ids = input_ids.transpose(0, 1).contiguous()
        qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        token_type_ids = seg_ids.transpose(0, 1).contiguous() if seg_ids is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        input_mask = 1.0 - attention_mask
        data_mask = input_mask[None]
        attn_mask = data_mask[:, :, :, None]
        attn_mask = (attn_mask > 0).to(input_mask.dtype)
        non_tgt_mask = -torch.eye(qlen).to(attn_mask)
        non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        embedding_output = self.dropout(self.word_embedding(input_ids))
        pos_emb = self.relative_positional_encoding(qlen, qlen, bsz=bsz, device=input_ids.device, dtype=embedding_output.dtype)
        pos_emb = self.dropout(pos_emb)
        non_tgt_mask = non_tgt_mask.to(pos_emb.dtype)

        encoder = self.encoder(embedding_output, attn_mask, attention_mask=attention_mask.transpose(0, 1), wordnet_content_id=wordnet_content_id,
                               nell_content_id=nell_content_id, wn_mask=wn_mask, ne_mask=ne_mask, pos_emb = pos_emb, non_tgt_mask=non_tgt_mask)

        start_pos, end_pos = self.output(encoder, attention_mask)

        return start_pos, end_pos


class double_base_mix_encoder(nn.Module):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(double_base_mix_encoder, self).__init__()
        bert_layer = BertLayer(config, False, False)

        self.config = config
        self.bert_config = bert_config

        self.integrate1 = knowledge_integration(config, bert_config, wordnet_embed)
        self.integrate2 = knowledge_integration(config, bert_config, nell_embed)

        self.match1 = self_matching(config, bert_config)
        self.match2 = self_matching(config, bert_config)

        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(config.num_hidden_layers)])

        self.merge1 = merge(config, bert_config)
        self.merge2 = merge(config, bert_config)
    
    def forward(self, input_embedding, attention_mask, wordnet_content_id, nell_content_id, wn_mask, ne_mask):
        all_hidden_layers = []
        wn_start = 3
        wn_end = 4
        ne_start = 5
        ne_end = 6
        num_layers = len(self.bert_layers)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(input_embedding.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # integrate wornet
        hidden_state = input_embedding
        for i in range(wn_end):
            hidden_state = self.bert_layers[i](hidden_state, extended_attention_mask)
            all_hidden_layers.append(hidden_state)
        wn_start_output = all_hidden_layers[wn_start]
        integrate1 = self.integrate1(wn_start_output, wordnet_content_id, wn_mask)
        integration1 = torch.cat((wn_start_output, integrate1), 2)

        matching1 = self.match1(integration1, attention_mask)
        hidden_state = self.merge1(matching1, all_hidden_layers[-1], attention_mask=extended_attention_mask)

        # integrate nell
        for j in range(wn_end, ne_end):
            hidden_state = self.bert_layers[j](hidden_state, extended_attention_mask)
            all_hidden_layers.append(hidden_state)
        ne_start_output = all_hidden_layers[ne_start]
        integrate2 = self.integrate2(ne_start_output, nell_content_id, ne_mask)
        integration2 = torch.cat((ne_start_output, integrate2), 2)
        matching2 = self.match2(integration2, attention_mask)
        hidden_state = self.merge2(matching2, all_hidden_layers[-1], attention_mask=extended_attention_mask)

        for k in range(ne_end, num_layers):
            hidden_state = self.bert_layers[k](hidden_state, extended_attention_mask)
        
        return hidden_state


class double_base_mix(BertPreTrainedModel):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(double_base_mix, self).__init__(config)
        self.config = config
        self.bert_config = bert_config
        self.embeddings= BertEmbeddings(bert_config)

        self.encoder = double_base_mix_encoder(config, bert_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)
        self.output = output_layer(config, bert_config)
    
    def forward(self, input_ids, attention_mask, seg_ids, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids=seg_ids)

        encoder = self.encoder(embedding_output, attention_mask, wordnet_content_id=wordnet_content_id,
                               nell_content_id=nell_content_id, wn_mask=wn_mask, ne_mask=ne_mask)
        
        start_pos, end_pos = self.output(encoder, attention_mask)

        return start_pos, end_pos


class ktnet_encoder_baseline(BertPreTrainedModel):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(ktnet_encoder_baseline, self).__init__(config)
        bert_layer = BertLayer(config, False, False)
        # bert_layer = bert_ignore_last_arg(config)
        self.config = config
        self.bert_config = bert_config
        if config.wordnet and config.nell:
            self.integrate1 = knowledge_integration(config, bert_config, wordnet_embed)
            self.integrate2 = knowledge_integration(config, bert_config, nell_embed)
        elif config.wordnet:
            self.integrate = knowledge_integration(config, bert_config, wordnet_embed)
        elif config.nell:
            self.integrate = knowledge_integration(config, bert_config, nell_embed)
        else:
            raise ValueError('at least one of Wordnet and Nell must be True')

        self.match = self_matching(config, bert_config)
        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(config.num_hidden_layers)])

    def forward(self, input_embedding, attention_mask, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None):
        all_hidden_layers = []

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.to(input_embedding.dtype)

        hidden_state = input_embedding
        for i in range(len(self.bert_layers)):
            hidden_state = self.bert_layers[i](hidden_state, extended_attention_mask)
            all_hidden_layers.append(hidden_state)

        start_layer_output = all_hidden_layers[-1]

        if self.config.wordnet and self.config.nell:
            integration1 = self.integrate1(start_layer_output, wordnet_content_id, wn_mask)
            integration2 = self.integrate2(start_layer_output, nell_content_id, ne_mask)
            integration = torch.cat((start_layer_output, integration1, integration2), 2)

        elif self.config.wordnet:
            integration = self.integrate(start_layer_output, wordnet_content_id, wn_mask)
            integration = torch.cat((start_layer_output, integration), 2)

        else:
            integration = self.integrate(start_layer_output, nell_content_id, ne_mask)
            integration = torch.cat((start_layer_output, integration), 2)

        matching = self.match(integration, attention_mask)

        return matching


class output_baseline(nn.Module):
    def __init__(self, config, bert_config):
        super(output_baseline, self).__init__()
        self.config = config
        if config.wordnet and config.nell:
            self.linear1 = nn.Linear(6*(bert_config.hidden_size+2*config.content_dim), 1)
            self.linear2 = nn.Linear(6*(bert_config.hidden_size+2*config.content_dim), 1)
        else:
            self.linear1 = nn.Linear(6*(bert_config.hidden_size+config.content_dim), 1)
            self.linear2 = nn.Linear(6*(bert_config.hidden_size+config.content_dim), 1)
        
        # self.linear3 = nn.Linear(config.seq_len, config.seq_len)
        # self.linear4 = nn.Linear(config.seq_len, config.seq_len)
    
    def forward(self, bert_output, attention_mask=None):

        # attention_mask = (1.0 - attention_mask) * -10000.0
        # attention_mask = attention_mask.to(bert_output.dtype)

        pos1 = self.linear1(bert_output).squeeze(2)
        pos2 = self.linear2(bert_output).squeeze(2)

        # pos1 = self.linear3(pos1)
        # pos2 = self.linear4(pos2)

        return pos1, pos2


class ktnet_baseline(BertPreTrainedModel):
    def __init__(self, config, bert_config, wordnet_embed=None, nell_embed=None):
        super(ktnet_baseline, self).__init__(config)
        self.config = config
        self.bert_config = bert_config
        self.embeddings = BertEmbeddings(bert_config)

        self.encoder = ktnet_encoder_baseline(config, bert_config, wordnet_embed=wordnet_embed, nell_embed=nell_embed)
        self.output = output_baseline(config, bert_config)

    def forward(self, input_ids, attention_mask, seg_ids, wordnet_content_id=None, nell_content_id=None, wn_mask=None,
                ne_mask=None):

        embedding_output = self.embeddings(input_ids, token_type_ids=seg_ids)

        encoder = self.encoder(embedding_output, attention_mask, wordnet_content_id=wordnet_content_id,
                               nell_content_id=nell_content_id, wn_mask=wn_mask, ne_mask=ne_mask)
        
        start_pos, end_pos = self.output(encoder, attention_mask=attention_mask)

        return start_pos, end_pos


class bert_base(BertPreTrainedModel):
    def __init__(self, config, bert_config):
        super(bert_base, self).__init__(config)
        self.config = bert_config
        self.embeddings = BertEmbeddings(bert_config)

        bert_layer = BertLayer(config, False, False)
        self.config = config
        self.bert_config = bert_config

        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(config.num_hidden_layers)])

        self.output1 = nn.Linear(bert_config.hidden_size, 1)
        self.output2 = nn.Linear(bert_config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, seg_ids):

        input_embeddings = self.embeddings(input_ids, token_type_ids=seg_ids)

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extend_mask = (1.0 - attention_mask) * -10000.0
        extend_mask = extend_mask.to(input_ids.dtype)


        hidden_state = input_embeddings
        for layer in self.bert_layers:
            hidden_state = layer(hidden_state, extend_mask)
        
        out_attention_mask = (1.0 - attention_mask.squeeze(1).squeeze(1)) * -10000.0
        out_attention_mask = out_attention_mask.to(input_embeddings.dtype)
        start_pos = self.output1(hidden_state).squeeze(2) + out_attention_mask
        end_pos = self.output2(hidden_state).squeeze(2) + out_attention_mask

        return start_pos, end_pos


class roberta_xlnet_encoder(nn.Module):
    def __init__(self, config, bert_config):
        super(roberta_xlnet_encoder, self).__init__()
        self.config = config

        if config.model_name == "roberta":
            roberta_layer = RobertaLayer(config)
            self.layers = nn.ModuleList([copy.deepcopy(roberta_layer) for _ in range(config.num_layers)])
        else:
            assert config.model_name == "xlnet"
            xlnet_layer = XLNetLayer(config)
            self.layers = nn.ModuleList([copy.deepcopy(xlnet_layer) for _ in range(config.num_layers)])
    
    def forward(self, input_embedding, attn_mask, attention_mask=None, pos_emb=None, non_tgt_mask=None):
        num_layers = len(self.layers)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(input_embedding.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_state = input_embedding
        for i in range(num_layers):
            if self.config.model_name == 'roberta':
                hidden_state = self.layers[i](hidden_state, attention_mask=extended_attention_mask)[0]
            else:
                assert self.config.model_name == 'xlnet'
                hidden_state = self.layers[i](hidden_state, None, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask, r=pos_emb, seg_mat=None)[0]

        return hidden_state.transpose(0, 1) if self.config.model_name == 'xlnet' else hidden_state


class roberta(RobertaPreTrainedModel):
    def __init__(self, config, bert_config):
        super(roberta, self).__init__(config)
        
        self.config = bert_config
        self.embeddings = RobertaEmbeddings(bert_config)
        self.encoder = roberta_xlnet_encoder(config, bert_config)
        self.output = output_layer(config, bert_config)

    def forward(self, input_ids, attention_mask, seg_ids):

        input_embedding = self.embeddings(input_ids)
        encoder = self.encoder(input_embedding, None, attention_mask=attention_mask)

        start_pos, end_pos = self.output(encoder)

        return start_pos, end_pos


class xlnet(XLNetPreTrainedModel):
    def __init__(self, config, bert_config):
        super(xlnet, self).__init__(config)

        self.config = bert_config
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

        self.encoder = roberta_xlnet_encoder(config, bert_config)
        self.output = output_layer(config, bert_config)

        
    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None, device=None, dtype=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.config.d_model, 2.0, dtype=dtype)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.config.d_model))

        beg, end = klen, -qlen

        fwd_pos_seq = torch.arange(beg, end, -1.0)
        if self.config.clamp_len > 0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.config.clamp_len, self.config.clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(device)
        return pos_emb

    def forward(self, input_ids, attention_mask, seg_ids):
        input_ids = input_ids.transpose(0, 1).contiguous()
        qlen, bsz = input_ids.shape[0], input_ids.shape[1]

        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        input_mask = 1.0 - attention_mask
        data_mask = input_mask[None]
        attn_mask = data_mask[:, :, :, None]
        attn_mask = (attn_mask > 0).to(input_mask.dtype)
        non_tgt_mask = -torch.eye(qlen).to(attn_mask)
        non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)

        embedding_output = self.dropout(self.word_embedding(input_ids))
        pos_emb = self.relative_positional_encoding(qlen, qlen, bsz=bsz, device=input_ids.device, dtype=embedding_output.dtype)
        pos_emb = self.dropout(pos_emb)
        non_tgt_mask = non_tgt_mask.to(pos_emb.dtype)

        encoder = self.encoder(embedding_output, attn_mask, attention_mask=attention_mask.transpose(0, 1), pos_emb = pos_emb, non_tgt_mask=non_tgt_mask)

        start_pos, end_pos = self.output(encoder)

        return start_pos, end_pos


