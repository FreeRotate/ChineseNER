#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : LSTM.py
# @Author: LauTrueYes
# @Date  : 2022/8/1 9:11
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_len, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=config.embed_dim)
        self.lstm = nn.LSTM(input_size=config.embed_dim, hidden_size=config.embed_dim, bidirectional=True)
        self.fc = nn.Linear(config.embed_dim * 2, config.num_labels) #分类
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ln = nn.LayerNorm(config.embed_dim * 2)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, word_ids, label_ids=None, label_mask=None):
        word_embed = self.embed(word_ids.permute(1,0))
        lstm_embed, _ = self.lstm(word_embed)
        lstm_embed = self.dropout(lstm_embed)
        lstm_embed = self.ln(lstm_embed)
        label_predict = self.fc(lstm_embed)
        if label_ids != None:
            active_logits = label_predict.view(-1, self.num_labels)
            active_labels = torch.where(label_mask.view(-1), label_ids.view(-1), self.loss_fct.ignore_index)
            loss = self.loss_fct(active_logits, active_labels)
        else:
            loss = None

        return loss, label_predict.argmax(dim=-1)
