#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: LauTrueYes
# @Date  : 2022/7/30 20:35
import json
import torch
import pandas as pd

class ContentLabel(object):
    def __init__(self, content, label):
        self.content = content
        self.label = label
    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            item = json.loads(line)
            dataset.append(item)
    return dataset

class Vocab(object):
    def __init__(self):
        self.id2word = None
        self.word2id = {'PAD':0}

    def add(self, dataset, test_file=False):
        id = len(self.word2id)
        for item in dataset:
            for word in item['text']:
                if word not in self.word2id:
                    self.word2id.update({word: id})
                    id += 1
        self.id2word = {j: i for i, j in self.word2id.items()}
    def __len__(self):
        return len(self.word2id)


class DataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for index in range(len(self.dataset)):
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch):
            yield batch
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def batch_variable(batch_data, vocab, config):
    batch_size = len(batch_data)
    max_seq_len = max(len(insts['text']) for insts in batch_data)
    word_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for index, item in enumerate(batch_data):
        sentence = item['text']
        seq_len = len(sentence)
        label_mask[index, :seq_len].fill_(1)
        word_ids[index, :seq_len] = torch.tensor([vocab.word2id[word] for word in sentence])

        for label_name, tag in item['label'].items():
            for words, posies in tag.items():
                for posi in posies:
                    start, end = posi[0], posi[1]
                    label_ids[index, start] = config.label2id['B-'+label_name]
                    label_ids[index, start+1:end+1] = config.label2id['I-'+label_name]
    return word_ids.to(config.device), label_ids.to(config.device), label_mask.to(config.device)
