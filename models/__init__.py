# -*- coding: utf-8 -*-

from transformers import (
    MegatronBertConfig,
    BertTokenizer,
    BertConfig,
    RobertaTokenizer,
    RobertaConfig,
)

from models.megatronbert_minilm import MegatronBertMiniLM

from models.bert_minilm import BertMiniLM
from models.bert_tinybert import BertTinyBert
from models.sparsebert_minilm import SparseBertMiniLM
from models.sparsebert_mlm import SparseBertMLM
from models.ofabert_minilm import OFABertMiniLM
from models.moebert_minilm import MoEBertMiniLM

from models.roberta_minilm import RobertaMiniLM


def get_model_class(model_type):
    if model_type == "megatronbert_minilm":
        tokenizer_class = BertTokenizer
        config_class = MegatronBertConfig
        model_class = MegatronBertMiniLM
    elif model_type == "bert_minilm":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = BertMiniLM
    elif model_type == "bert_tinybert":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = BertTinyBert
    elif model_type == "sparsebert_minilm":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = SparseBertMiniLM
    elif model_type == "sparsebert_mlm":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = SparseBertMLM
    elif model_type == "ofabert_minilm":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = OFABertMiniLM
    elif model_type == "moebert_minilm":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = MoEBertMiniLM
    elif model_type == "roberta_minilm":
        tokenizer_class = RobertaTokenizer
        config_class = RobertaConfig
        model_class = RobertaMiniLM
    else:
        raise KeyError(f"Unknown model type {model_type}.")

    return tokenizer_class, config_class, model_class