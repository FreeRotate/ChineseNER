#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: LauTrueYes
# @Date  : 2022/7/30 20:35
import json
import torch

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

    def add(self, dataset):
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
    sentence_list = []
    labels_list = []

    for index, item in enumerate(batch_data):
        sentence = item['text']
        labels = ['O'] * len(sentence)

        for label_name, tag in item['label'].items():
            for sub_name, sub_index in tag.items():
                for start_index, end_index in sub_index:
                    assert ''.join(sentence[start_index:end_index + 1]) == sub_name
                    if start_index == end_index:
                        labels[start_index] = 'B-' + label_name
                    else:
                        labels[start_index] = 'B-' + label_name
                        labels[start_index + 1:end_index + 1] = ['I-' + label_name] * (len(sub_name) - 1)
        sentence_list.append(sentence)
        labels_list.append(labels)

    batch_size = len(batch_data)
    max_seq_len = max(len(insts['text']) for insts in batch_data)
    word_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    index = 0
    for sentence, labels in zip(sentence_list, labels_list):
        word_ids[index, :len(sentence)] = torch.tensor([vocab.word2id[word] for word in sentence])
        label_ids[index, :len(labels)] = torch.tensor([config.label2id[label] for label in labels])
        label_mask[index, :len(labels)].fill_(1)
        index += 1
    return word_ids.to(config.device), label_ids.to(config.device), label_mask.to(config.device)
