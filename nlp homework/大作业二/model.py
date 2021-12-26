import torch
import torch.nn as nn
from torch import optim
import os
from constant import *

"""
class MyAttentionBiLSTM
继承自nn.Module
具有Attention层+BiLSTM层的神经网络模型
"""


class MyAttentionBiLSTM(nn.Module):
    def __init__(self):
        super(MyAttentionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.attentionW = nn.Parameter(torch.randn(BATCH_SIZE, HIDDEN_SIZE * 2))
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(HIDDEN_SIZE * 2, 2)

    def attentionNet(self, input):
        # input(batch_size,sequence_length,hidden_size*2)
        H = input.view(-1, HIDDEN_SIZE * 2)
        # H(batch_size*sequence_length,hidden_size*2)
        M = H.tanh()
        # M(batch_size*sequence_length,hidden_size*2)
        alpha = self.softmax(self.attentionW.mm(M.t()))
        # alpha(batch_size,batch_size*sequence_length)
        r = alpha.mm(M)
        # r(batch_size,hidden_size*2)
        rt = r.tanh()
        # rt(batch_size,hidden_size*2)
        return rt

    def forward(self, x):
        # x(batch_size,sequence_length,word2vec_length)
        h, (hidden_state, cell_state) = self.lstm(x)
        del (hidden_state, cell_state)
        # h(batch_size,sequence_length,hidden_size*2)
        rt = self.attentionNet(h)
        # rt(batch_size,hidden_size*2)
        ret = self.linear(rt)
        # ret(batch_size,2)
        return ret


"""
function get_Attention_BiLSTM
用于获取模型，并给模型绑定对应的optimizer和lr_scheduler

deploy表示是否为部署模型，默认为False
False代表模型需要进行训练
True代表模型已完成训练，需要部署到应用层中
"""


def get_Attention_BiLSTM(deploy=False):
    model = MyAttentionBiLSTM().to(DEVICE)
    if os.path.exists(STATE_FILE):
        model.load_state_dict(torch.load(STATE_FILE))
    if deploy:
        return model
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    exp_lr = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    return model, opt, exp_lr
