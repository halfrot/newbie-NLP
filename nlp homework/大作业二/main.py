import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from constant import *
import model
import dataset
from train import fit

"""
测试
"""

if __name__ == "__main__":
    lstm, opt, exp_lr = model.get_LSTM()
    loss_func = F.cross_entropy

    train_ds = dataset.MyDataSet(TRAIN_SET_FILE)
    valid_ds = dataset.MyDataSet(VALID_SET_FILE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)

    fit(EPOCHS, lstm, loss_func, train_dl, valid_dl, opt, exp_lr, VALID_PRECISION_FILE, LOSS_FILE)

    test_ds = dataset.MyDataSet("test_set.txt")
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
