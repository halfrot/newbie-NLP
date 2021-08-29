import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import time
import os
from MyDataset import *
from File2Data import *
from Attention_bilstm import *
from macro_f1 import cal_F1_measure
from cnn import *

emb = []
word_to_idx = {}
cnt = 0


def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x), y)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)


def data2list(model, valid_dl):
    predict = []
    relation = []
    for x, y in valid_dl:
        predict.extend(model(x).argmax(dim=1))
        relation.extend(y)
    return predict, relation


def fit(epochs, model, loss_func, train_dl, valid_dl, opt, exp_lr, f1file, lossfile):
    for epoch in range(epochs):
        print("epoch start")
        time_start = time.time()
        model.train()
        for x, y in train_dl:
            loss_batch(model, loss_func, x, y, opt)
        model.eval()
        file_f1 = open(f1file, "a+", encoding="UTF-8")
        file_loss = open(lossfile, "a+", encoding="UTF-8")
        sum_loss = 0
        sum_num = 0
        with torch.no_grad():
            for x, y in valid_dl:
                val_loss, loss_num = loss_batch(model, loss_func, x, y)
                sum_loss += val_loss
                sum_num += loss_num
            predict, relation = data2list(model, valid_dl)
            f1 = cal_F1_measure(predict, relation)
            file_f1.write(str(f1) + '\n')
            loss = sum_loss / sum_num
            file_loss.write(str(loss) + '\n')
        file_f1.close()
        file_loss.close()
        print("epoch : %d  loss : %f  valid_F1 : %f" % (epoch, loss, f1))

        # lstm
        torch.save(obj=model.state_dict(), f="Attention_bilstm.pth")

        # cnn
        # torch.save(obj=model.state_dict(), f="cnn.pth")
        exp_lr.step()
        time_end = time.time()
        print("time cost : %f" % (time_end - time_start))

    print("fit ends")


read_glove("glove.6B.50d.txt", emb, word_to_idx, cnt)

train_ds = MyDataSet("Mytrain.txt", "Answer.txt", word_to_idx, emb)
valid_ds = MyDataSet("Myvalid.txt", "Answer.txt", word_to_idx, emb)
train_dl = DataLoader(train_ds, batch_size=1)
valid_dl = DataLoader(valid_ds, batch_size=1)

loss_func = F.cross_entropy

# for lstm

lstm, opt, exp_lr = get_LSTM("Attention_bilstm.pth", 0.001)
fit(10, lstm, loss_func, train_dl, valid_dl, opt, exp_lr, "valid_f1.txt", "loss.txt")

# for cnn

# cnn, opt, exp_lr = get_CNN("cnn.pth", 0.001)
# fit(10, cnn, loss_func, train_dl, valid_dl, opt, exp_lr, "valid_f1.txt", "loss.txt")
