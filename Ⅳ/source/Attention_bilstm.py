import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import time
import os


class MyLSTM(nn.Module):
    def __init__(self, size):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=size, hidden_size=100, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.w = nn.Parameter(torch.randn(1, 100 * 2))
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(100 * 2, 10)

    def forward(self, input):
        h, c = self.lstm(input)
        h = h.squeeze(0)
        H = h.t()
        M = H.tanh()
        w_M = self.w.mm(M)
        alpha = self.softmax(w_M)
        r = H.mm(alpha.t())
        rt = r.tanh()
        ret = self.linear(rt.t())
        return ret


def get_LSTM(state_file, learning_rate):
    lstm = MyLSTM(56)
    if os.path.exists(state_file):
        print("loaded state")
        lstm.load_state_dict(torch.load(state_file))
    opt = optim.Adam(lstm.parameters(), lr=learning_rate)
    exp_lr = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    return lstm, opt, exp_lr
