#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: LauTrueYes
# @Date  : 2022/7/22 15:35
import argparse
from utils import load_dataset, Vocab, DataLoader
from config import Config
from train import train
from test import test
from importlib import import_module

parser = argparse.ArgumentParser(description='NER')
parser.add_argument('--model', type=str, default='LSTM', help='CNN, GRU, LSTM, TransformerEncoder')  #在defaule中修改所需的模型
args = parser.parse_args()

if __name__ == '__main__':
    dataset = './data/CLUENER/'
    config = Config(dataset=dataset)

    train_CL = load_dataset(config.train_path)
    dev_CL = load_dataset(config.dev_path)
    test_CL = load_dataset(config.test_path)

    vocab = Vocab()
    vocab.add(dataset=train_CL)
    vocab.add(dataset=dev_CL)
    vocab.add(dataset=test_CL)

    train_loader = DataLoader(train_CL, config.batch_size)
    dev_loader = DataLoader(dev_CL, config.batch_size)
    test_loader = DataLoader(test_CL, config.batch_size)

    model_name = args.model
    lib = import_module('models.'+model_name)
    model = lib.Model(len(vocab), config).to(config.device)

    train(model=model, train_loader=train_loader, dev_loader=dev_loader, config=config, vocab=vocab)
    test(model=model, test_loader=dev_loader, config=config, vocab=vocab)