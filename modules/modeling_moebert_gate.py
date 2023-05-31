# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" PyTorch MoEBertv2 Model. """


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


class MoEBertEmbeddings(nn.Module):
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

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MoELinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, element_size=1, moe_dim=0):
        super().__init__(in_features, out_features, bias=bias)
        self.num_experts = 1
        self.element_size = element_size
        self.moe_dim = moe_dim

    def forward(self, input, expert_idx=None):
        if self.num_experts > 1:
            # Common forward with indexing.
            if expert_idx is not None:
                return F.linear(input, self.weight[expert_idx], self.bias[expert_idx])
            # Sparse forward.
            # TODO: Find more efficient way to achieve so.
            # An alternative that is slower:
            # torch.nn.functional.conv1d(input.transpose(1,2).reshape(1,-1,3),weight.transpose(1,2).reshape(-1,3).unsqueeze(-1),bias.reshape(-1),groups=2).reshape(2,4,-1).transpose(1,2)
            if self.bias is not None:
                return torch.matmul(input, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)
            else:
                return torch.matmul(input, self.weight.transpose(1, 2))
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}, bias={}".format(
            self.num_experts,
            self.in_features,
            self.out_features,
            self.bias is not None,
        )

    def moefy(self, num_experts, num_expert_elements, num_shared_elements):
        self.num_experts = num_experts
        # NOTE: moe_dim should be either 0 or 1.
        # moefy() should only work on reordered modules.
        expert_size = num_expert_elements * self.element_size
        shared_size = num_shared_elements * self.element_size
        if self.moe_dim == 0:
            self.in_features = expert_size
            shared_weight = self.weight[:, :shared_size]
            specific_weight = self.weight[:, shared_size:].reshape(self.out_features, -1, self.element_size)
            if self.bias is not None:
                shared_bias = self.bias
            weight, bias = [], []
            for expert_i in range(num_experts):
                weight_i = torch.cat((shared_weight, specific_weight[:, expert_i::num_experts].reshape(self.out_features, -1).contiguous()), dim=1)[:, :expert_size].clone().detach()
                weight.append(weight_i)
                if self.bias is not None:
                    bias_i = shared_bias.clone().detach()
                    bias.append(bias_i)
        if self.moe_dim == 1:
            self.out_features = expert_size
            shared_weight = self.weight[:shared_size, :]
            specific_weight = self.weight[shared_size:, :].reshape(-1, self.element_size, self.in_features)
            if self.bias is not None:
                shared_bias = self.bias[:shared_size]
                specific_bias = self.bias[shared_size:].reshape(-1, self.element_size)
            weight, bias = [], []
            for expert_i in range(num_experts):
                weight_i = torch.cat((shared_weight, specific_weight[expert_i::num_experts, :].reshape(-1, self.in_features).contiguous()), dim=0)[:expert_size, :].clone().detach()
                weight.append(weight_i)
                if self.bias is not None:
                    bias_i = torch.cat((shared_bias, specific_bias[expert_i::num_experts].reshape(-1).contiguous()), dim=0)[:expert_size].clone().detach()
                    bias.append(bias_i)
        weight = torch.stack(weight, dim=0)
        if self.bias is not None:
            bias = torch.stack(bias, dim=0)
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True


class MoEBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) if getattr(config, "attention_head_size", None) is None else config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_attention_heads_moefied = self.num_attention_heads
        self.all_head_size_moefied = self.all_head_size

        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = MoELinear(config.hidden_size, self.all_head_size, element_size=self.attention_head_size, moe_dim=1)
        self.key = MoELinear(config.hidden_size, self.all_head_size, element_size=self.attention_head_size, moe_dim=1)
        self.value = MoELinear(config.hidden_size, self.all_head_size, element_size=self.attention_head_size, moe_dim=1)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def transpose(self, x, num_heads, head_size):
        new_x_shape = x.shape[:-1] + (num_heads, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def attention(self, query, key, value, attention_mask, head_mask):
        query_layer = self.transpose(query, self.num_attention_heads_moefied, self.attention_head_size)
        key_layer = self.transpose(key, self.num_attention_heads_moefied, self.attention_head_size)
        value_layer = self.transpose(value, self.num_attention_heads_moefied, self.attention_head_size)

        # Take the dot product to get the raw scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MoEBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to.
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size_moefied,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        pass
        # query = self.query(hidden_states)
        # key = self.key(hidden_states)
        # value = self.value(hidden_states)
        
        # context, attention_probs = self.attention(query, key, value, attention_mask, head_mask)
        # outputs = (context,)
        # if output_attentions:
        #     outputs += (attention_probs,)
        # return outputs


class MoEBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) if getattr(config, "attention_head_size", None) is None else config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dense = MoELinear(self.all_head_size, config.hidden_size, element_size=self.attention_head_size, moe_dim=0)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, expert_idx=None, expert_gate=None):
        hidden_states = self.dense(hidden_states, expert_idx=expert_idx)
        hidden_states = self.dropout(hidden_states)
        if expert_gate is not None:
            hidden_states = expert_gate * hidden_states
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MoEBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MoEBertSelfAttention(config)
        self.output = MoEBertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        pass
        # self_outputs = self.self(
        #     hidden_states,
        #     attention_mask,
        #     head_mask,
        #     output_attentions,
        # )
        # attention_output = self.output(self_outputs[0], hidden_states)
        # outputs = (attention_output,) + self_outputs[1:]
        # return outputs


class MoEBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = MoELinear(config.hidden_size, config.intermediate_size, element_size=1, moe_dim=1)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, expert_idx=None):
        hidden_states = self.dense(hidden_states, expert_idx=expert_idx)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MoEBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = MoELinear(config.intermediate_size, config.hidden_size, element_size=1, moe_dim=0)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, expert_idx=None, expert_gate=None):
        hidden_states = self.dense(hidden_states, expert_idx=expert_idx)
        hidden_states = self.dropout(hidden_states)
        if expert_gate is not None:
            hidden_states = expert_gate * hidden_states
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def one_hot(indices, num_classes):
    """Workaround for https://github.com/pytorch/pytorch/issues/55579"""
    assert num_classes > 0, "num_classes must be a positive integer"
    ret = torch.zeros(indices.shape + (num_classes,), device=indices.device, dtype=indices.dtype)
    ret.scatter_(-1, indices.unsqueeze(-1), 1)
    return ret


class GateLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def moefy(self, num_experts):
        self.out_features = num_experts
        weight = torch.empty(self.out_features, self.in_features).to(self.weight.device)
        if self.bias is not None:
            bias = torch.empty(self.out_features).to(self.weight.device)
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True
        self.reset_parameters()

class MoEBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MoEBertAttention(config)
        self.intermediate = MoEBertIntermediate(config)
        self.output = MoEBertOutput(config)
        
        self.attn_gate = GateLinear(config.hidden_size, 1)
        self.ffn_gate = GateLinear(config.hidden_size, 1)
        self.epsilon = config.hidden_dropout_prob
        self.num_attn_experts = 1
        self.num_ffn_experts = 1

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        layer_attn_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        layer_attn_output, layer_attn_load_balance = layer_attn_outputs[:2]
        outputs = layer_attn_outputs[2:]

        layer_ffn_output, layer_ffn_load_balance = self.feed_forward(layer_attn_output)

        outputs = (layer_ffn_output, layer_attn_load_balance, layer_ffn_load_balance,) + outputs
        return outputs

    def self_attention(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        if self.num_attn_experts > 1:
            batch_size, seq_len, hidden_size = hidden_states.shape
            reshaped_hidden_states = hidden_states.reshape(-1, hidden_size)
            gates = self.attn_gate(reshaped_hidden_states)
            # Add randomness to gates for better exploration with a dropout.
            if self.training:
                # Uniform number from [1 - eps, 1 + eps].
                noises = torch.rand_like(gates)
                noises = 1.0 + noises * 2 * self.epsilon - self.epsilon
                gates += noises
            gates = F.softmax(gates, -1)
            indices = torch.argmax(gates, -1)
            mask = one_hot(indices, self.num_attn_experts)

            me = torch.mean(gates, 0)
            #ce = torch.mean(mask.float(), 0)
            te = torch.sum(mask.float(), 0)
            ce = te / te.sum(0, keepdim=True)
            attn_load_balance = self.num_attn_experts * me * ce

            token_dist = mask.sum(0).tolist()

            positions = indices.argsort(0)
            reshaped_hidden_states = reshaped_hidden_states.index_select(0, positions) # Reorder.
            reshaped_hidden_states = reshaped_hidden_states.split(token_dist, 0)

            gates = gates.gather(-1, index=indices.unsqueeze(-1))
            gates = gates.index_select(0, positions)
            gates = gates.split(token_dist, 0)

            query, key, value = [], [], []
            for i in range(self.num_attn_experts):
                query.append(self.attention.self.query(reshaped_hidden_states[i], expert_idx=i))
                key.append(self.attention.self.key(reshaped_hidden_states[i], expert_idx=i))
                value.append(self.attention.self.value(reshaped_hidden_states[i], expert_idx=i))
            query = torch.cat(query, 0)
            query = query.index_select(0, positions.argsort(0)) # Restore the order.
            query = query.reshape(batch_size, seq_len, self.attention.self.all_head_size_moefied).contiguous()
            key = torch.cat(key, 0)
            key = key.index_select(0, positions.argsort(0))
            key = key.reshape(batch_size, seq_len, self.attention.self.all_head_size_moefied).contiguous()
            value = torch.cat(value, 0)
            value = value.index_select(0, positions.argsort(0))
            value = value.reshape(batch_size, seq_len, self.attention.self.all_head_size_moefied).contiguous()
            
            context, attn_probs = self.attention.self.attention(query, key, value, attention_mask, head_mask)
            reshaped_context = context.reshape(-1, self.attention.self.all_head_size_moefied)
            reshaped_context = reshaped_context.index_select(0, positions) # Reorder.
            reshaped_context = reshaped_context.split(token_dist, 0)
            attn_output = []
            for i in range(self.num_attn_experts):
                attn_output.append(self.attention.output(reshaped_context[i], reshaped_hidden_states[i], expert_idx=i, expert_gate=gates[i]))
            attn_output = torch.cat(attn_output, 0)
            attn_output = attn_output.index_select(0, positions.argsort(0)) # Restore the order.
            attn_output = attn_output.reshape(batch_size, seq_len, hidden_size).contiguous()
            outputs = (attn_output, attn_load_balance,) 
            if output_attentions:
                outputs += (attn_probs,)
        else:
            attn_load_balance = torch.ones(1,).to(hidden_states.device)
            query = self.attention.self.query(hidden_states)
            key = self.attention.self.key(hidden_states)
            value = self.attention.self.value(hidden_states)
            
            context, attn_probs = self.attention.self.attention(query, key, value, attention_mask, head_mask)
            attn_output = self.attention.output(context, hidden_states)
            outputs = (attn_output, attn_load_balance,) 
            if output_attentions:
                outputs += (attn_probs,)
        return outputs

    def feed_forward(self, attn_output):
        if self.num_ffn_experts > 1:
            batch_size, seq_len, hidden_size = attn_output.shape
            reshaped_attn_output = attn_output.reshape(-1, hidden_size)
            gates = self.ffn_gate(reshaped_attn_output)
            # Add randomness to gates for better exploration with a dropout.
            if self.training:
                # Uniform number from [1 - eps, 1 + eps].
                noises = torch.rand_like(gates)
                noises = 1.0 + noises * 2 * self.epsilon - self.epsilon
                gates += noises
            gates = F.softmax(gates, -1)
            indices = torch.argmax(gates, -1)
            # print(torch.sum(indices == 0))
            # print(torch.sum(indices == 1))
            # print(torch.sum(indices == 2))
            # print(torch.sum(indices == 3))
            mask = one_hot(indices, self.num_ffn_experts)

            me = torch.mean(gates, 0)
            #ce = torch.mean(mask.float(), 0)
            te = torch.sum(mask.float(), 0)
            ce = te / te.sum(0, keepdim=True)
            ffn_load_balance = self.num_ffn_experts * me * ce

            token_dist = mask.sum(0).tolist()

            positions = indices.argsort(0)
            reshaped_attn_output = reshaped_attn_output.index_select(0, positions) # Reorder.
            reshaped_attn_output = reshaped_attn_output.split(token_dist, 0)

            gates = gates.gather(-1, index=indices.unsqueeze(-1))
            gates = gates.index_select(0, positions)
            gates = gates.split(token_dist, 0)

            # TODO: This can be improved with kernel.
            ffn_output = []
            for i in range(self.num_ffn_experts):
                o = self.intermediate(reshaped_attn_output[i], expert_idx=i)
                o = self.output(o, reshaped_attn_output[i], expert_idx=i, expert_gate=gates[i])
                ffn_output.append(o)
            ffn_output = torch.cat(ffn_output, 0)
            ffn_output = ffn_output.index_select(0, positions.argsort(0)) # Restore the order.
            ffn_output = ffn_output.reshape(batch_size, seq_len, hidden_size).contiguous()
        else:
            ffn_load_balance = torch.ones(1,).to(attn_output.device)
            intermediate_output = self.intermediate(attn_output)
            ffn_output = self.output(intermediate_output, attn_output)
            
        return (ffn_output, ffn_load_balance,)

    def moefy_heads(self, num_experts, num_expert_elements, num_shared_elements):
        self.num_attn_experts = num_experts
        self.attn_gate.moefy(num_experts)
        self.attention.self.query.moefy(num_experts, num_expert_elements, num_shared_elements)
        self.attention.self.key.moefy(num_experts, num_expert_elements, num_shared_elements)
        self.attention.self.value.moefy(num_experts, num_expert_elements, num_shared_elements)
        self.attention.output.dense.moefy(num_experts, num_expert_elements, num_shared_elements)
        self.attention.self.all_head_size_moefied = self.attention.self.attention_head_size * num_expert_elements
        self.attention.self.num_attention_heads_moefied = num_expert_elements
    
    def moefy_neurons(self, num_experts, num_expert_elements, num_shared_elements):
        self.num_ffn_experts = num_experts
        self.ffn_gate.moefy(num_experts)
        self.intermediate.dense.moefy(num_experts, num_expert_elements, num_shared_elements)
        self.output.dense.moefy(num_experts, num_expert_elements, num_shared_elements)

class MoEBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MoEBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_load_balances = ()
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

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, 
                            output_attentions,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            all_load_balances = all_load_balances + (layer_outputs[1], layer_outputs[2])
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_load_balances,
                all_hidden_states,
                all_attentions,
            ]
            if v is not None
        )


class MoEBertPooler(nn.Module):
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


class MoEBertPredictionHeadTransform(nn.Module):
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


class MoEBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MoEBertPredictionHeadTransform(config)

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


class MoEBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MoEBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MoEBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class MoEBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MoEBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MoEBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if not hasattr(self.config, "moe"):
            self.config.moe = "1,1"
            
        self.embeddings = MoEBertEmbeddings(config)
        self.encoder = MoEBertEncoder(config)

        self.pooler = MoEBertPooler(config) if add_pooling_layer else None

        self.init_weights()

        self.moefy(self.config.moe)

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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output, load_balances = encoder_outputs[:2]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, load_balances, pooled_output) + encoder_outputs[2:]

    def moefy(self, moe):
        num_attn_experts, num_ffn_experts = (int(n) for n in moe.split(","))
        assert self.config.num_attention_heads % num_attn_experts == 0
        num_attn_expert_elements = int(self.config.num_attention_heads / num_attn_experts)
        assert self.config.num_attention_heads % (num_attn_experts * 3 / 2) == 0
        num_attn_shared_elements = int(self.config.num_attention_heads / num_attn_experts * 2 / 3)
        assert self.config.intermediate_size % num_ffn_experts == 0
        num_ffn_expert_elements = int(self.config.intermediate_size / num_ffn_experts)
        assert self.config.intermediate_size % (num_ffn_experts * 3 / 2) == 0
        num_ffn_shared_elements = int(self.config.intermediate_size / num_ffn_experts * 2 / 3)
        for layer_idx in range(self.config.num_hidden_layers):
            if num_attn_experts > 1:
                self.encoder.layer[layer_idx].moefy_heads(num_attn_experts, num_attn_expert_elements, num_attn_shared_elements)
            if num_ffn_experts > 1:
                self.encoder.layer[layer_idx].moefy_neurons(num_ffn_experts, num_ffn_expert_elements, num_ffn_shared_elements)
