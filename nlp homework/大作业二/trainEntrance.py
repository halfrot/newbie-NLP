import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from constant import *
import model
import dataset
from train import fit
import torch
import torch.nn as nn
from prework import read_word2vec

"""
模型训练的入口
"""
model, opt, exp_lr = model.get_Attention_BiLSTM()
loss_func = F.cross_entropy

train_ds = dataset.MyDataSet(TRAIN_SET_FILE)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
dict_list, word_to_idx = read_word2vec(WORD_TO_VECTOR_FILE)
embed = nn.Embedding.from_pretrained(torch.FloatTensor(dict_list))

fit(EPOCHS, model, loss_func, train_dl, opt, exp_lr, VALID_PRECISION_FILE, embed, dict_list, word_to_idx)
