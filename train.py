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

    dev_best_f1 = float('-inf')
    avg_loss = []
    optimizer = optim.AdamW(params=model.parameters(), lr=config.lr)
    for epoch in range(0, config.epochs):
        train_right, train_total = 0, 0
        for batch_idx, batch_data in enumerate(train_loader):
            model.train()   #训练模型
            word_ids, label_ids, label_mask = batch_variable(batch_data, vocab, config)
            loss, predicts = model(word_ids, label_ids, label_mask)

            avg_loss.append(loss.data.item())

            batch_right = ((predicts == label_ids) * label_mask).sum().item()
            batch_total = label_mask.sum().item()
            train_right += batch_right
            train_total += batch_total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if batch_idx % 10 == 0:
                print("Epoch:{}--------Iter:{}--------train_loss:{:.3f}--------train_acc:{:.3f}".format(epoch+1, batch_idx+1, np.array(avg_loss).mean(), train_right/train_total))
        dev_loss, dev_acc, dev_f1, dev_report = evaluate(model, dev_loader, config, vocab)
        msg = "Dev Loss:{:.3f}--------Dev Acc:{:.3f}--------Dev F1:{:.3f}"
        print(msg.format(dev_loss, dev_acc, dev_f1))
        print(dev_report)

        if dev_best_f1 < dev_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), config.save_path)
            print("***************************** Save Model *****************************")

def evaluate(model, one_loader, config, vocab, output_dict=False):
    model.eval()    #评价模式
    loss_total = 0
    predict_all = []
    label_all = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(one_loader):
            word_ids, label_ids, label_mask = batch_variable(batch_data, vocab, config)
            loss, predicts = model(word_ids, label_ids, label_mask)

            loss_total = loss_total + loss

            for i, sen_mask in enumerate(label_mask):
                for j, word_mask in enumerate(sen_mask):
                    if word_mask.item() == False:
                        predicts[i][j] = 0
            labels_list = []
            for index_i, ids in enumerate(label_ids):
                labels_list.append([config.id2label[id.cpu().item()] for index_j, id in enumerate(ids)])
            predicts_list = []
            for index_i, pres in enumerate(predicts):
                predicts_list.append([config.id2label[pre.cpu().item()] for index_j, pre in enumerate(pres)])

            label_all += labels_list
            predict_all += predicts_list

    acc = accuracy_score(label_all, predict_all)
    f1 = f1_score(label_all, predict_all, average='micro')
    report = classification_report(label_all, predict_all, digits=3, output_dict=output_dict)

    return loss_total/len(one_loader), acc, f1, report
