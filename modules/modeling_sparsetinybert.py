# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SparseTinyBert Model. """


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
from transformers.utils import logging
from transformers.models.bert.modeling_bert import load_tf_weights_in_bert, BertConfig, BertPreTrainedModel


logger = logging.get_logger(__name__)


def round_to_multiple_of_eight(input_size):
    return round(input_size * 1.0 / 8) * 8


class SparseEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(SparseEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim_origin = embedding_dim
        self.embedding_dim_sparsified = embedding_dim
        # 128 is quite good.
        # The k of svd is min(vocab_size, hidden_size)=hidden_size here.
        u, s, vh = torch.linalg.svd(self.weight.data.clone().detach(), full_matrices=False)
        self.u = nn.Parameter(u)
        self.s = nn.Parameter(s)
        self.vh = nn.Parameter(vh)

    def forward(self, input):
        us = torch.matmul(self.u[:, :self.embedding_dim_sparsified], torch.diag(self.s[:self.embedding_dim_sparsified]))
        vh = self.vh[:self.embedding_dim_sparsified]
        return F.linear(
                F.embedding(
                input, us, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse),
                vh.t(),
                False
            )

    def sparsify(self, num_elements):
        self.embedding_dim_sparsified = self.embedding_dim_origin - round_to_multiple_of_eight(num_elements)

    def densify(self):
        self.embedding_dim = self.embedding_dim_sparsified
        u = self.u[:, :self.embedding_dim_sparsified].clone().detach()
        s = self.s[:self.embedding_dim_sparsified].clone().detach()
        vh = self.vh[:self.embedding_dim_sparsified].clone().detach()
        self.u = nn.Parameter(torch.empty_like(u))
        self.u.requires_grad = False
        self.u.copy_(u.contiguous())
        self.u.requires_grad = True
        self.s = nn.Parameter(torch.empty_like(s))
        self.s.requires_grad = False
        self.s.copy_(s.contiguous())
        self.s.requires_grad = True
        self.vh = nn.Parameter(torch.empty_like(vh))
        self.vh.requires_grad = False
        self.vh.copy_(vh.contiguous())
        self.vh.requires_grad = True


class SparseTinyBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    # def sparsify_embeddings(self, sparsity):
    #     self.word_embeddings.sparsify(sparsity)

    # def densify_embeddings(self):
    #     self.word_embeddings.densify()

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, element_size, bias=True, sparse_dim=0):
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias)
        self.in_features_origin = in_features
        self.out_features_origin = out_features
        self.in_features_sparsified = in_features
        self.out_features_sparsified = out_features
        self.element_size = element_size
        self.sparse_dim = sparse_dim

    def forward(self, input):
        weight = self.weight[:self.out_features_sparsified, :self.in_features_sparsified]
        if self.bias is not None:
            bias = self.bias[:self.out_features_sparsified]
        else:
            bias = self.bias
        return F.linear(input, weight, bias)

    def reorder(self, indices):
        indices = indices.to(self.weight.device)
        weight = self.weight.index_select(1 - self.sparse_dim, indices).clone().detach()
        if self.bias is not None:
            if self.sparse_dim == 0:
                bias = self.bias.clone().detach()
            else:
                bias = self.bias[indices].clone().detach()
        #self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            #self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True

    def sparsify(self, num_elements):
        if self.sparse_dim == 0:
            self.in_features_sparsified = self.in_features_origin - round_to_multiple_of_eight(num_elements * self.element_size)
        if self.sparse_dim == 1:
            self.out_features_sparsified = self.out_features_origin - round_to_multiple_of_eight(num_elements * self.element_size)

    def densify(self):
        self.in_features = self.in_features_sparsified
        self.out_features = self.out_features_sparsified
        weight = self.weight[:self.out_features_sparsified, :self.in_features_sparsified].clone().detach()
        if self.bias is not None:
            bias = self.bias[:self.out_features_sparsified].clone().detach()
        else:
            bias = self.bias.clone().detach()
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True
        

class SparseTinyBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_attention_heads_sparsified = self.num_attention_heads
        self.all_head_size_sparsified = self.all_head_size

        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # Sparse Linear for adaptive width.
        self.query = SparseLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, sparse_dim=1)
        self.key = SparseLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, sparse_dim=1)
        self.value = SparseLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, sparse_dim=1)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose(self, x, num_heads, head_size):
        new_x_shape = x.shape[:-1] + (num_heads, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def attention(self, query, key, value, attention_mask, head_mask):
        query_layer = self.transpose(query, self.num_attention_heads_sparsified, self.attention_head_size)
        key_layer = self.transpose(key, self.num_attention_heads_sparsified, self.attention_head_size)
        value_layer = self.transpose(value, self.num_attention_heads_sparsified, self.attention_head_size)

        # Take the dot product to get the raw scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MiniLMModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # Mask heads if we want to.
        if head_mask is not None:
            context_layer = context_layer * head_mask
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size_sparsified,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs, attention_scores


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self.all_head_size_sparsified = self.query.out_features_sparsified
        self.num_attention_heads_sparsified = int(self.all_head_size_sparsified / self.attention_head_size)

        if self.all_head_size_sparsified < 8:
            context, attentions, raw_attentions = None, None, None
        else:
            query = self.query(hidden_states)
            key = self.key(hidden_states)
            value = self.value(hidden_states)
            context, attentions, raw_attentions = self.attention(query, key, value, attention_mask, head_mask)
        outputs = (context,)
        if output_attentions:
            outputs += (attentions, raw_attentions,)
        return outputs


class SparseTinyBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.dense = SparseLinear(config.hidden_size, config.hidden_size, \
            self.attention_head_size, sparse_dim=0)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if hidden_states is None:
            hidden_states = self.LayerNorm(input_tensor)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SparseTinyBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SparseTinyBertSelfAttention(config)
        self.output = SparseTinyBertSelfOutput(config)

    def forward(self,):
        pass


class SparseTinyBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.neuron_size = config.intermediate_size
        self.dense = SparseLinear(config.hidden_size, config.intermediate_size, \
            1, sparse_dim=1)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        if self.dense.out_features_sparsified < 8:
            return None
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.intermediate_act_fn(hidden_states)
            return hidden_states


class SparseTinyBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SparseLinear(config.intermediate_size, config.hidden_size, \
            1, sparse_dim=0)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if hidden_states is None:
            hidden_states = self.LayerNorm(input_tensor)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SparseTinyBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SparseTinyBertAttention(config)
        self.intermediate = SparseTinyBertIntermediate(config)
        self.output = SparseTinyBertOutput(config)

    def reorder_heads(self, indices):
        n, h = self.attention.self.num_attention_heads, self.attention.self.attention_head_size
        indices = torch.arange(n * h).reshape(n, h)[indices].reshape(-1).contiguous().long()
        self.attention.self.query.reorder(indices)
        self.attention.self.key.reorder(indices)
        self.attention.self.value.reorder(indices)
        self.attention.output.dense.reorder(indices)

    def reorder_neurons(self, indices):
        self.intermediate.dense.reorder(indices)
        self.output.dense.reorder(indices)

    def sparsify_heads(self, num_elements):
        self.attention.self.query.sparsify(num_elements)
        self.attention.self.key.sparsify(num_elements)
        self.attention.self.value.sparsify(num_elements)
        self.attention.output.dense.sparsify(num_elements)

    def sparsify_neurons(self, num_elements):
        self.intermediate.dense.sparsify(num_elements)
        self.output.dense.sparsify(num_elements)
        
    def densify(self):
        self.attention.self.query.densify()
        self.attention.self.key.densify()
        self.attention.self.value.densify()
        self.attention.output.dense.densify()
        self.intermediate.dense.densify()
        self.output.dense.densify()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        neuron_mask=None,
        output_attentions=False,
    ):
        # HACK: weight and bias will be set to nearly 0 elements when the sparsity is nearly 0.
        # This manner is somehow fine for both sparsification and densification ; ).
        # However, we do some tricks here to avoid carrying out actual computation here,
        # which will result in errors.
        
        self_outputs = self.attention.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.attention.output(self_outputs[0], hidden_states)

        if neuron_mask is not None:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_with_mask, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, neuron_mask
            )
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
        outputs = (layer_output,) + self_outputs[1:]
        return outputs

    def feed_forward_chunk_with_mask(self, attention_output, neuron_mask):
        intermediate_output = self.intermediate(attention_output)
        # Mask neurons if we want to.
        intermediate_output = intermediate_output * neuron_mask
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SparseTinyBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SparseTinyBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        neuron_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        if output_attentions:
            all_attentions, all_raw_attentions = (), ()
        else:
            all_attentions, all_raw_attentions = None, None
        if output_hidden_states:
            all_hidden_states = ()
        else:
            all_hidden_states = None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_neuron_mask = neuron_mask[i] if neuron_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    layer_neuron_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    layer_neuron_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                all_raw_attentions = all_raw_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_attentions,
                all_raw_attentions,
            ]
            if v is not None
        )


class SparseTinyBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SparseTinyBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SparseTinyBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = SparseTinyBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SparseTinyBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SparseTinyBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class SparseTinyBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class SparseTinyBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SparseTinyBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class SparseTinyBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = SparseTinyBertEmbeddings(config)
        self.encoder = SparseTinyBertEncoder(config)

        self.pooler = SparseTinyBertPooler(config) if add_pooling_layer else None

        self.init_weights()

        if not hasattr(self.config, "sparsity"):
            self.config.sparsity = "0"
        if not hasattr(self.config, "sparsity_map"):
            self.config.sparsity_map = {"0": {"head": {}, "neuron": {}}}
        self.sparsify(self.config.sparsity)
        self.densify()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        neuron_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head and neuron mask if needed
        # 1.0 in mask indicate we keep the head or neuron neuron
        # input head_mask has shape [num_hidden_layers x num_heads]
        # input neuron_mask has shape [num_hidden_layers x intermediate_size]
        # and head_mask is converted to shape [num_hidden_layers x batch_size (*1) x seq_length (*1) x num_heads x head_size (*1)]
        # similarly neuron_mask is converted to shape [num_hidden_layers x batch_size (*1) x seq_length (*1) x intermediate_size]
        if head_mask is not None:
            head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1).to(self.dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        if neuron_mask is not None:
            neuron_mask = neuron_mask.unsqueeze(1).unsqueeze(1).to(self.dtype)
        else:
            neuron_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            neuron_mask=neuron_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]

    def reorder(self, head_indices, neuron_indices):
        for layer_idx, indices in head_indices.items():
            self.encoder.layer[layer_idx].reorder_heads(indices)
        for layer_idx, indices in neuron_indices.items():
            self.encoder.layer[layer_idx].reorder_neurons(indices)

    def sparsify(self, sparsity):
        assert sparsity in self.config.sparsity_map, f"Sparsity {sparsity} is not in the sparsity map {self.config.sparsity_map}."
        #self.sparsity = sparsity
        head_map, neuron_map = self.config.sparsity_map[sparsity]["head"], self.config.sparsity_map[sparsity]["neuron"]
        #print(head_map)
        #print(neuron_map)
        #print(head_map["11"])
        for layer_idx in range(self.config.num_hidden_layers):
            self.encoder.layer[layer_idx].sparsify_heads(head_map.get(str(layer_idx), 0))
            self.encoder.layer[layer_idx].sparsify_neurons(neuron_map.get(str(layer_idx), 0))

    def densify(self):
        for layer_idx in range(self.config.num_hidden_layers):
            self.encoder.layer[layer_idx].densify()

    def count_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return round(num_params / 10 ** 6, 2)
