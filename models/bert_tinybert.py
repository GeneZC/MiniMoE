# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modeling_sparsetinybert import BertPreTrainedModel, SparseTinyBertModel

import collections


class BertTinyBert(BertPreTrainedModel):
    Output = collections.namedtuple(
        "Output",
        (
            "hidden_states", 
            "attentions",
        )
    )

    def __init__(self, config):
        super().__init__(config)
        self.bert = SparseTinyBertModel(config)
        self.init_weights()
        self.hidden_size = config.hidden_size
        if self.hidden_size < 768:
            self.hidden_transform = nn.Linear(config.hidden_size, 768)

    def forward(self, inputs):
        text_indices, text_mask, text_segments = inputs

        # Gather knowledge.
        # Sequentially, num_layers x (batch_size x num_heads x seq_length x seq_length).
        all_hidden_states, _, all_attentions \
            = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments, output_hidden_states=True, output_attentions=True)[2:]
        
        # Mapping: -1th layer.
        # batch_size x num_layers x num_heads x seq_length x seq_length
        hidden_states = torch.stack((all_hidden_states[0], all_hidden_states[-1],), dim=1)
        attentions = torch.stack((all_attentions[-1],), dim=1)

        # Mask and reshape.
        # Hidden states & attentions.
        hidden_size = hidden_states.shape[-1]
        mask = text_mask.unsqueeze(1).unsqueeze(-1).expand_as(hidden_states)
        hidden_states = torch.masked_select(hidden_states, mask)
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.hidden_size < 768:
            hidden_states = self.hidden_transform(hidden_states)
        logit_size = attentions.shape[-1]
        mask = text_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand_as(attentions)
        attentions = torch.masked_select(attentions, mask)
        attentions = attentions.reshape(-1, logit_size)
        # attentions = None

        return BertTinyBert.Output(
            hidden_states=hidden_states, 
            attentions=attentions,
        )

    @staticmethod
    def loss_fn(t_output, s_output):
        loss = F.mse_loss(s_output.hidden_states, t_output.hidden_states.detach(), reduction="mean")
        loss += F.mse_loss(s_output.attentions, t_output.attentions.detach(), reduction="mean")
        loss /= 2
        return loss