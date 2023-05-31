# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead

import collections


class BertMLM(BertPreTrainedModel):
    Output = collections.namedtuple(
        "Output",
        (   
            "mlm_logits",
            "mlm_labels",
        )
    )

    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self, inputs):
        text_indices, text_mask, text_segments, mlm_positions, mlm_mask, mlm_labels = inputs

        # Gather knowledge.
        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]

        # Mapping.
        # Logit.
        mlm_hidden_states = torch.gather(hidden_states, 1, mlm_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]))
        mlm_logits = self.cls(mlm_hidden_states)

        # Mask and reshape.
        # MLM logits.
        logit_size = mlm_logits.shape[-1]
        mask = mlm_mask.unsqueeze(-1).expand_as(mlm_logits)
        mlm_logits = torch.masked_select(mlm_logits, mask)
        mlm_logits = mlm_logits.reshape(-1, logit_size)
        mlm_labels = torch.masked_select(mlm_labels, mlm_mask)
        mlm_labels = mlm_labels.reshape(-1)

        return BertMLM.Output(
            mlm_logits=mlm_logits,
            mlm_labels=mlm_labels,
        )

    @staticmethod
    def loss_fn(t_output, s_output):
        loss = F.cross_entropy(s_output.mlm_logits, s_output.mlm_labels, reduction="mean")
        return loss 