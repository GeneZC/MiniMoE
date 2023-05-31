# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modeling_sparsebert import BertPreTrainedModel, SparseBertModel

import collections
import math


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        

        if config.hidden_size % config.num_relation_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of relation "
                "heads (%d)" % (config.hidden_size, config.num_relation_heads)
            )

        self.num_relation_heads = config.num_relation_heads
        self.relation_head_size = int(config.hidden_size / config.num_relation_heads)
        self.all_head_size = self.num_relation_heads * self.relation_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def transpose(self, x, num_heads, head_size):
        new_x_shape = x.shape[:-1] + (num_heads, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def relation(self, query, key, value, attention_mask):
        query_layer = self.transpose(query, self.num_relation_heads, self.relation_head_size)
        key_layer = self.transpose(key, self.num_relation_heads, self.relation_head_size)
        value_layer = self.transpose(value, self.num_relation_heads, self.relation_head_size)

        # Take the dot product to get the raw scores.
        qq_relation_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))
        kk_relation_scores = torch.matmul(key_layer, key_layer.transpose(-1, -2))
        vv_relation_scores = torch.matmul(value_layer, value_layer.transpose(-1, -2))

        qq_relation_scores = qq_relation_scores / math.sqrt(self.relation_head_size)
        kk_relation_scores = kk_relation_scores / math.sqrt(self.relation_head_size)
        vv_relation_scores = vv_relation_scores / math.sqrt(self.relation_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MiniLMModel forward() function)
            qq_relation_scores = qq_relation_scores + attention_mask
            kk_relation_scores = kk_relation_scores + attention_mask
            vv_relation_scores = vv_relation_scores + attention_mask

        # Normalize the scores to probabilities.
        qq_relation_probs = F.softmax(qq_relation_scores, dim=-1)
        kk_relation_probs = F.softmax(kk_relation_scores, dim=-1)
        vv_relation_probs = F.softmax(vv_relation_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        qq_relation_probs = self.dropout(qq_relation_probs)
        kk_relation_probs = self.dropout(kk_relation_probs)
        vv_relation_probs = self.dropout(vv_relation_probs)
        
        return qq_relation_probs, kk_relation_probs, vv_relation_probs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask.to(query.dtype)) * -10000.0
        
        qq_relations, kk_relations, vv_relations = self.relation(query, key, value, extended_attention_mask)
        return (qq_relations, kk_relations, vv_relations,)


class SparseBertMiniLM(BertPreTrainedModel):
    Output = collections.namedtuple(
        "Output",
        (
            "qq_relations", 
            "kk_relations", 
            "vv_relations",
        )
    )

    def __init__(self, config):
        super().__init__(config)
        self.bert = SparseBertModel(config)
        self.self = SelfAttention(config)
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments = inputs

        # Gather knowledge.
        # Sequentially, num_layers x (batch_size x num_heads x seq_length x seq_length).
        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        qq_relations, kk_relations, vv_relations = self.self(hidden_states, attention_mask=text_mask)
        
        # Mapping: -1th layer.
        # batch_size x num_heads x seq_length x seq_length

        # Mask and reshape.
        # Q, K, V relations.
        logit_size = qq_relations.shape[-1]
        mask = text_mask.unsqueeze(1).unsqueeze(-1).expand_as(qq_relations)
        qq_relations = torch.masked_select(qq_relations, mask)
        qq_relations = qq_relations.reshape(-1, logit_size)
        kk_relations = torch.masked_select(kk_relations, mask)
        kk_relations = kk_relations.reshape(-1, logit_size)
        vv_relations = torch.masked_select(vv_relations, mask)
        vv_relations = vv_relations.reshape(-1, logit_size)

        return SparseBertMiniLM.Output(
            qq_relations=qq_relations, 
            kk_relations=kk_relations, 
            vv_relations=vv_relations,
        )

    @staticmethod
    def loss_fn(t_output, s_output):
        loss = F.kl_div(torch.log(s_output.qq_relations + 1e-7), t_output.qq_relations.detach() + 1e-7, reduction="batchmean") \
            + F.kl_div(torch.log(s_output.kk_relations + 1e-7), t_output.kk_relations.detach() + 1e-7, reduction="batchmean") \
            + F.kl_div(torch.log(s_output.vv_relations + 1e-7), t_output.vv_relations.detach() + 1e-7, reduction="batchmean")
        loss = loss / 3.0
        return loss