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
""" PyTorch SparseBert Model. """


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


class SparseLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.normalized_shape_origin = (normalized_shape,)
        self.normalized_shape_sparsified = (normalized_shape,)

    def forward(self, input, hidden_mask=None):
        if self.elementwise_affine:
            weight = self.weight[:self.normalized_shape_sparsified[0]]
            bias = self.bias[:self.normalized_shape_sparsified[0]]
        else:
            weight = self.weight
            bias = self.bias
        if hidden_mask is not None:
            remain_indices = torch.where(~hidden_mask.squeeze().eq(0))[0]
            output = input.clone()
            input = input.index_select(-1, remain_indices)
            weight = weight[remain_indices]
            bias = bias[remain_indices]
            shape = (len(remain_indices),)
            norm = F.layer_norm(
                input, shape, weight, bias, self.eps)
            output[:, :, remain_indices] = norm
            return output 
        else:
            return F.layer_norm(
                input, self.normalized_shape_sparsified, weight, bias, self.eps)

    def reorder(self, indices):
        if self.elementwise_affine:
            indices = indices.to(self.weight.device)
            weight = self.weight[indices].clone().detach()
            bias = self.bias[indices].clone().detach()
            self.weight.requires_grad = False
            self.weight.copy_(weight.contiguous())
            self.weight.requires_grad = True
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True

    def sparsify(self, num_elements):
        self.normalized_shape_sparsified = (self.normalized_shape_origin[0] - round_to_multiple_of_eight(num_elements),)

    def densify(self):
        self.normalized_shape = self.normalized_shape_sparsified
        if self.elementwise_affine:
            weight = self.weight[:self.normalized_shape_sparsified[0]].clone().detach()
            bias = self.bias[:self.normalized_shape_sparsified[0]].clone().detach()
            self.weight = nn.Parameter(torch.empty_like(weight))
            self.weight.requires_grad = False
            self.weight.copy_(weight.contiguous())
            self.weight.requires_grad = True
            self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True


class SparseEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim_origin = embedding_dim
        self.embedding_dim_sparsified = embedding_dim

    def forward(self, input):
        weight = self.weight[:, :self.embedding_dim_sparsified]
        #return F.linear(
        return F.embedding(input, weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse)
        #        vh.t(),
        #        False
        #    )

    def reorder(self, indices):
        indices = indices.to(self.weight.device)
        weight = self.weight.index_select(1, indices).clone().detach()
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True

    def sparsify(self, num_elements):
        self.embedding_dim_sparsified = self.embedding_dim_origin - round_to_multiple_of_eight(num_elements)

    def densify(self):
        self.embedding_dim = self.embedding_dim_sparsified
        weight = self.weight[:, :self.embedding_dim_sparsified].clone().detach()
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True


class SparseBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = SparseEmbedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = SparseEmbedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = SparseEmbedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = SparseLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def reorder_embeddings(self, indices):
        self.word_embeddings.reorder(indices)
        self.position_embeddings.reorder(indices)
        self.token_type_embeddings.reorder(indices)
        self.LayerNorm.reorder(indices)

    def sparsify_embeddings(self, num_elements):
        self.word_embeddings.sparsify(num_elements)
        self.position_embeddings.sparsify(num_elements)
        self.token_type_embeddings.sparsify(num_elements)
        self.LayerNorm.sparsify(num_elements)

    def densify(self):
        self.word_embeddings.densify()
        self.position_embeddings.densify()
        self.token_type_embeddings.densify()
        self.LayerNorm.densify()

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, hidden_mask=None
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
        if hidden_mask is not None:
            embeddings = embeddings * hidden_mask
        embeddings = self.dropout(embeddings)
        return embeddings


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, element_size=1, dim=0):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features_origin = in_features
        self.out_features_origin = out_features
        self.in_features_sparsified = in_features
        self.out_features_sparsified = out_features
        self.element_size = element_size
        self.dim = dim

    def forward(self, input):
        weight = self.weight[:self.out_features_sparsified, :self.in_features_sparsified]
        if self.bias is not None:
            bias = self.bias[:self.out_features_sparsified]
        else:
            bias = self.bias
        return F.linear(input, weight, bias)

    def reorder(self, indices, for_hidden=False):
        if for_hidden:
            self.dim = 1 - self.dim

        indices = indices.to(self.weight.device)
        weight = self.weight.index_select(1 - self.dim, indices).clone().detach()
        if self.bias is not None:
            if self.dim == 0:
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

        if for_hidden:
            self.dim = 1 - self.dim

    def sparsify(self, num_elements, for_hidden=False):
        if for_hidden:
            self.dim = 1 - self.dim
            cache_element_size = self.element_size
            self.element_size = 1

        if self.dim == 0:
            self.in_features_sparsified = self.in_features_origin - round_to_multiple_of_eight(num_elements * self.element_size)
        if self.dim == 1:
            self.out_features_sparsified = self.out_features_origin - round_to_multiple_of_eight(num_elements * self.element_size)

        if for_hidden:
            self.dim = 1 - self.dim
            self.element_size = cache_element_size

    def densify(self):
        self.in_features = self.in_features_sparsified
        self.out_features = self.out_features_sparsified
        weight = self.weight[:self.out_features_sparsified, :self.in_features_sparsified].clone().detach()
        if self.bias is not None:
            bias = self.bias[:self.out_features_sparsified].clone().detach()
        # else:
        #     bias = self.bias.clone().detach()
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True
        

class SparseBertSelfAttention(nn.Module):
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
        self.query = SparseLinear(config.hidden_size, self.all_head_size, element_size=self.attention_head_size, dim=1)
        self.key = SparseLinear(config.hidden_size, self.all_head_size, element_size=self.attention_head_size, dim=1)
        self.value = SparseLinear(config.hidden_size, self.all_head_size, element_size=self.attention_head_size, dim=1)

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
        return context_layer, attention_probs

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
            context, attentions = None, None
        else:
            query = self.query(hidden_states)
            key = self.key(hidden_states)
            value = self.value(hidden_states)
            context, attentions = self.attention(query, key, value, attention_mask, head_mask)
        outputs = (context,)
        if output_attentions:
            outputs += (attentions,)
        return outputs


class SparseBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.dense = SparseLinear(config.hidden_size, config.hidden_size, \
            element_size=self.attention_head_size, dim=0)
        self.LayerNorm = SparseLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, hidden_mask=None):
        if hidden_states is None:
            if hidden_mask is not None:
                hidden_states = hidden_states * hidden_mask
            hidden_states = self.LayerNorm(input_tensor, hidden_mask=hidden_mask)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if hidden_mask is not None:
                hidden_states = hidden_states * hidden_mask
            hidden_states = self.LayerNorm(hidden_states + input_tensor, hidden_mask=hidden_mask)
        if hidden_mask is not None:
            hidden_states = hidden_states * hidden_mask
        return hidden_states


class SparseBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SparseBertSelfAttention(config)
        self.output = SparseBertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class SparseBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.neuron_size = config.intermediate_size
        self.dense = SparseLinear(config.hidden_size, config.intermediate_size, \
            element_size=1, dim=1)
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


class SparseBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SparseLinear(config.intermediate_size, config.hidden_size, \
            element_size=1, dim=0)
        self.LayerNorm = SparseLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, hidden_mask=None):
        if hidden_states is None:
            if hidden_mask is not None:
                hidden_states = hidden_states * hidden_mask
            hidden_states = self.LayerNorm(input_tensor, hidden_mask=hidden_mask)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if hidden_mask is not None:
                hidden_states = hidden_states * hidden_mask
            hidden_states = self.LayerNorm(hidden_states + input_tensor, hidden_mask=hidden_mask)
        if hidden_mask is not None:
            hidden_states = hidden_states * hidden_mask
        return hidden_states


class SparseBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SparseBertAttention(config)
        self.intermediate = SparseBertIntermediate(config)
        self.output = SparseBertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        neuron_mask=None,
        hidden_mask=None,
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
        attention_output = self.attention.output(self_outputs[0], hidden_states, hidden_mask)

        layer_output = self.feed_forward(attention_output, neuron_mask, hidden_mask)

        outputs = (layer_output,) + self_outputs[1:]
        return outputs

    def feed_forward(self, attention_output, neuron_mask=None, hidden_mask=None):
        intermediate_output = self.intermediate(attention_output)
        if neuron_mask is not None:
            # Mask neurons if we want to.
            intermediate_output = intermediate_output * neuron_mask
        layer_output = self.output(intermediate_output, attention_output, hidden_mask)
        return layer_output

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

    def reorder_hiddens(self, indices):
        self.attention.self.query.reorder(indices, for_hidden=True)
        self.attention.self.key.reorder(indices, for_hidden=True)
        self.attention.self.value.reorder(indices, for_hidden=True)
        self.attention.output.dense.reorder(indices, for_hidden=True)
        self.attention.output.LayerNorm.reorder(indices)
        self.intermediate.dense.reorder(indices, for_hidden=True)
        self.output.dense.reorder(indices, for_hidden=True)
        self.output.LayerNorm.reorder(indices)

    def sparsify_heads(self, num_elements):
        self.attention.self.query.sparsify(num_elements)
        self.attention.self.key.sparsify(num_elements)
        self.attention.self.value.sparsify(num_elements)
        self.attention.output.dense.sparsify(num_elements)

    def sparsify_neurons(self, num_elements):
        self.intermediate.dense.sparsify(num_elements)
        self.output.dense.sparsify(num_elements)
    
    def sparsify_hiddens(self, num_elements):
        self.attention.self.query.sparsify(num_elements, for_hidden=True)
        self.attention.self.key.sparsify(num_elements, for_hidden=True)
        self.attention.self.value.sparsify(num_elements, for_hidden=True)
        self.attention.output.dense.sparsify(num_elements, for_hidden=True)
        self.attention.output.LayerNorm.sparsify(num_elements)
        self.intermediate.dense.sparsify(num_elements, for_hidden=True)
        self.output.dense.sparsify(num_elements, for_hidden=True)
        self.output.LayerNorm.sparsify(num_elements)
        
    def densify(self):
        self.attention.self.query.densify()
        self.attention.self.key.densify()
        self.attention.self.value.densify()
        self.attention.output.dense.densify()
        self.attention.output.LayerNorm.densify()
        self.intermediate.dense.densify()
        self.output.dense.densify()
        self.output.LayerNorm.densify()


class SparseBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SparseBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        neuron_mask=None,
        hidden_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        if output_attentions:
            all_attentions = ()
        else:
            all_attentions = None
        if output_hidden_states:
            all_hidden_states = ()
        else:
            all_hidden_states = None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_neuron_mask = neuron_mask[i] if neuron_mask is not None else None
            layer_hidden_mask = hidden_mask[i] if hidden_mask is not None else None

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
                    layer_hidden_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    layer_neuron_mask,
                    layer_hidden_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_attentions,
            ]
            if v is not None
        )


class SparseBertPooler(nn.Module):
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


class SparseBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if not hasattr(self.config, "sparsity"):
            self.config.sparsity = "0"
        if not hasattr(self.config, "sparsity_map"):
            self.config.sparsity_map = {"0": {"hidden": {}, "head": {}, "neuron": {}}}

        self.embeddings = SparseBertEmbeddings(config)
        self.encoder = SparseBertEncoder(config)

        self.pooler = SparseBertPooler(config) if add_pooling_layer else None

        self.init_weights()

        self.sparsify(self.config.sparsity)
        self.densify()
        self.recover_indices = nn.Parameter(torch.arange(self.config.hidden_size).float())

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
        hidden_mask=None,
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
        # hidden_mask is converted to shape [num_hidden_layers x batch_size (*1) x seq_length (*1) x hidden_size]
        if head_mask is not None:
            head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1).to(self.dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        if neuron_mask is not None:
            neuron_mask = neuron_mask.unsqueeze(1).unsqueeze(1).to(self.dtype)
        else:
            neuron_mask = [None] * self.config.num_hidden_layers
        if hidden_mask is not None:
            hidden_mask = [hidden_mask.unsqueeze(0).unsqueeze(1).to(self.dtype)]  * (self.config.num_hidden_layers + 1)
        else:
            hidden_mask = [None] * (self.config.num_hidden_layers + 1)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            hidden_mask=hidden_mask[0],
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            neuron_mask=neuron_mask,
            hidden_mask=hidden_mask[1:],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        #sequence_output = encoder_outputs[0].index_select(2, self.recover_indices.long())
        sequence_output_shape = encoder_outputs[0].shape[:-1] + (self.config.hidden_size,)
        sequence_output = torch.zeros(*sequence_output_shape).to(encoder_outputs[0].device)
        sequence_output.index_copy_(2, self.recover_indices.long()[:encoder_outputs[0].shape[-1]], encoder_outputs[0])
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]

    def reorder(self, head_indices, neuron_indices, hidden_indices):
        for layer_idx, indices in head_indices.items():
            self.encoder.layer[layer_idx].reorder_heads(indices)
        for layer_idx, indices in neuron_indices.items():
            self.encoder.layer[layer_idx].reorder_neurons(indices)
        for layer_idx, indices in hidden_indices.items():
            if layer_idx == -1:
                self.embeddings.reorder_embeddings(indices)
                recover_indices = indices.to(self.recover_indices.device)
                self.recover_indices.requires_grad = False
                self.recover_indices.copy_(recover_indices.contiguous())
                self.recover_indices.requires_grad = True
            else:
                self.encoder.layer[layer_idx].reorder_hiddens(indices)

    def sparsify(self, sparsity):
        assert sparsity in self.config.sparsity_map, f"Sparsity {sparsity} is not in the sparsity map {self.config.sparsity_map}."
        #self.sparsity = sparsity
        head_map, neuron_map, hidden_map = \
            self.config.sparsity_map[sparsity]["head"], self.config.sparsity_map[sparsity]["neuron"], self.config.sparsity_map[sparsity]["hidden"]
        
        for layer_idx in range(self.config.num_hidden_layers):
            self.encoder.layer[layer_idx].sparsify_heads(head_map.get(str(layer_idx), 0))
            self.encoder.layer[layer_idx].sparsify_neurons(neuron_map.get(str(layer_idx), 0))
            self.encoder.layer[layer_idx].sparsify_hiddens(hidden_map.get(str(layer_idx), 0))
        self.embeddings.sparsify_embeddings(hidden_map.get("-1", 0))

    def densify(self):
        for layer_idx in range(self.config.num_hidden_layers):
            self.encoder.layer[layer_idx].densify()
        self.embeddings.densify()

    def count_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return round(num_params / 10 ** 6, 2)
