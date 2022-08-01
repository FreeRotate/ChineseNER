#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : LSTM.py
# @Author: LauTrueYes
# @Date  : 2022/8/1 9:11
import torch
import torch.nn as nn
from torchcrf import CRF

class Model(nn.Module):
    def __init__(self, vocab_len, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.lstm = nn.LSTM(input_size=config.embed_dim, hidden_size=config.embed_dim, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(config.embed_dim * 2)
        self.classifier = nn.Linear(config.embed_dim * 2, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, word_ids, label_ids=None, label_mask=None, use_crf=True):
        word_embed = self.embed(word_ids)
        word_embed = self.dropout(word_embed)
        sequence_output, _ = self.lstm(word_embed)
        sequence_output = self.layer_norm(sequence_output)
        logits = self.classifier(sequence_output)
        if label_ids != None:
            if use_crf:
                loss = self.crf(logits, label_ids)
                loss = -1 * loss
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(label_mask.view(-1), label_ids.view(-1), self.loss_fct.ignore_index)
                loss = self.loss_fct(active_logits, active_labels)

        else:
            loss = None

        return loss, logits.argmax(dim=-1)
