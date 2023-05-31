# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modeling_roberta import RobertaPreTrainedModel, RobertaModel

import collections


class RobertaMiniLM(RobertaPreTrainedModel):
    Output = collections.namedtuple(
        "Output",
        (
            "qq_relations", 
            "kk_relations", 
            "vv_relations",
        )
    )

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments = inputs

        # Gather knowledge.
        # Sequentially, num_layers x (batch_size x num_heads x seq_length x seq_length).
        all_qq_relations, all_kk_relations, all_vv_relations \
            = self.roberta(text_indices, attention_mask=text_mask, output_relations=True)[2:]
        
        # Mapping: -1th layer.
        # batch_size x num_heads x seq_length x seq_length
        qq_relations = all_qq_relations[-1]
        kk_relations = all_kk_relations[-1]
        vv_relations = all_vv_relations[-1]


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

        return RobertaMiniLM.Output(
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