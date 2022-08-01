#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: LauTrueYes
# @Date  : 2022/7/30 20:34
import torch

class Config(object):
    def __init__(self, dataset):
        self.train_path = dataset + 'train.json'     #训练集
        self.dev_path = dataset + 'dev.json'         #验证集
        self.test_path = dataset + 'test.json'       #测试集
        self.class_path = dataset + 'class.txt'       #测试集
        self.predict_path = dataset + '/saved_data/' + 'predict.json'    #预测结果
        self.value_path = dataset + '/saved_data/' + 'value.csv'        #评价效果
        self.save_path = dataset + '/saved_data/' + 'model.ckpl'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.epochs = 1
        self.batch_size = 64
        self.max_seq = 50
        self.lr = 1e-3
        self.dropout_rate = 0.1
        self.require_improvement = 2

        self.label_list = [x.strip() for x in open(self.class_path, encoding='utf-8').readlines()]
        self.num_labels = len(self.label_list)                     #类别数量
        self.id2label = dict(enumerate(self.label_list))            #标号转类别
        self.label2id = {j: i for i, j in self.id2label.items()}    #类别转标号

        self.kernal_sizes = (2, 3, 4)
        self.kernel_nums =(50, 100, 150)
        self.num_filters = 128
        self.embed_dim = 200
