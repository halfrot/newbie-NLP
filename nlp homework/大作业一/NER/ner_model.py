import torch
import torch.nn as nn
from torch import optim
import os
from ner_constant import *

"""
class MyAttentionBiLSTM
继承自nn.Module
具有Attention层+BiLSTM层的神经网络模型
"""


class MyAttentionBiLSTM(nn.Module):
    def __init__(self):
        super(MyAttentionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=SIZE, hidden_size=100, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.w = nn.Parameter(torch.randn(10, 100 * 2))
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(100 * 2, 3)

    def forward(self, input):
        h, c = self.lstm(input)
        # print(h.shape)
        h = h.squeeze(0)
        H = h.t()
        M = H.tanh()
        w_M = self.w.mm(M)
        alpha = self.softmax(w_M)
        r = H.mm(alpha.t())
        rt = r.tanh()
        # print(rt.shape)
        ret = self.linear(rt.t())
        return ret


"""
class MyBiLSTM
继承自nn.Module
BiLSTM层的神经网络模型
"""


class MyBiLSTM(nn.Module):
    def __init__(self):
        super(MyBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=SIZE, hidden_size=100, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(100 * 2, 3)

    def forward(self, input):
        # print(type(input), input)
        h, c = self.lstm(input)
        ret = self.linear(h)
        return ret


"""
function get_LSTM
用于获取模型，并给模型绑定对应的optimizer和lr_scheduler
attention用于表示是否添加Attention层，默认为False

attention=False，返回MyBiLSTM()
attention=True，返回MyAttentionBiLSTM()
"""


def get_LSTM(attention=False):
    if not attention:
        lstm = MyBiLSTM()
    else:
        lstm = MyAttentionBiLSTM()
    if os.path.exists(STATE_FILE):
        lstm.load_state_dict(torch.load(STATE_FILE))
    opt = optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
    exp_lr = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    return lstm, opt, exp_lr
