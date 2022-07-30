#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: LauTrueYes
# @Date  : 2022/7/30 20:45
import torch
import numpy as np
import torch.optim as optim
from utils import batch_variable
from seqeval.metrics import accuracy_score, classification_report, f1_score

def train(model, train_loader, dev_loader, config, vocab):

    loss_all = np.array([], dtype=float)
    label_all = np.array([], dtype=float)
    predict_all = np.array([], dtype=float)
    dev_best_f1 = float('-inf')

    optimizer = optim.AdamW(params=model.parameters(), lr=config.lr)
    for epoch in range(0, config.epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            model.train()   #训练模型
            word_ids, label_ids = batch_variable(batch_data, vocab, config)
            loss, label_predict = model(word_ids, label_ids)

            loss_all = np.append(loss_all, loss.data.item())
            label_all = np.append(label_all, label_ids.data.cpu().numpy())
            predict_all = np.append(predict_all, label_predict.data.cpu().numpy())
            acc = accuracy_score(predict_all, label_all)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("Epoch:{}--------Iter:{}--------train_loss:{:.3f}--------train_acc:{:.3f}".format(epoch+1, batch_idx+1, loss_all.mean(), acc))
        dev_loss, dev_acc, dev_f1, dev_report = evaluate(model, dev_loader, config, vocab)
        msg = "Dev Loss:{}--------Dev Acc:{}--------Dev F1:{}"
        print(msg.format(dev_loss, dev_acc, dev_f1))
        print("Dev Report")
        print(dev_report)

        if dev_best_f1 < dev_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), config.save_path)
            print("***************************** Save Model *****************************")

def evaluate(config, model, dev_loader, vocab, output_dict=False):
    model.eval()    #评价模式
    loss_all = np.array([], dtype=float)
    predict_all = []
    label_all = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dev_loader):
            word_ids, label_ids = batch_variable(batch_data, vocab, config)
            loss, label_predict = model(word_ids, label_ids)

            loss_all = np.append(loss_all, loss.data.item())
            predict_all.append(label_predict.data)
            label_all.append(label_ids.data)
    acc = accuracy_score(label_all, predict_all)
    f1 = f1_score(label_all, predict_all, average='macro')
    report = classification_report(label_all, predict_all, digits=3, output_dict=output_dict)

    return loss.mean(), acc, f1, report