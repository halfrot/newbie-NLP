import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import time
import os


class CNN(nn.Module):
    def __init__(self, size=56):
        super(CNN, self).__init__()
        self.covn = nn.Sequential(
            nn.Conv2d(size, 10, 5, 1, 2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2))
        self.linear = nn.Linear(10, 10)

    def forward(self, input):
        out = self.covn(input)
        out = out.view(-1)
        out = self.linear(out)
        return out


def get_CNN(state_file, learning_rate):
    lstm = CNN(56)
    if os.path.exists(state_file):
        lstm.load_state_dict(torch.load(state_file))
    opt = optim.Adam(lstm.parameters(), lr=learning_rate)
    exp_lr = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    return lstm, opt, exp_lr
