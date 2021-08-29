import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import time
import os

state_file = "protected data/softmax.pth"
word_vector_file = "ctb.50d.vec"
train_set_file = "train_set.txt"
valid_set_file = "validation_set.txt"
valid_f1_measure_file = "valid_f1_measure.txt"
loss_file = "loss.txt"

size = 50
batch_size = 256
learning_rate = 0.005
epochs = 700
dict = {}


def read_vector(file_path):
    f = open(file_path, "r", encoding="UTF-8")
    for s in f:
        vec = s.split()
        for i in range(1, len(vec)):
            vec[i] = float(vec[i])
        dict[vec[0]] = vec[1:]
    f.close()


class MyDataSet(TensorDataset):
    def __init__(self, file_path):
        super(MyDataSet, self).__init__()
        self.word = []
        self.label = []
        self.len = 0
        f = open(file_path, "r", encoding="UTF-8")
        for s in f:
            raw = s.split()
            if raw[0] in dict:
                self.word.append(raw[0])
            else:
                self.word.append("-unknown-")
            self.label.append(raw[1])
            self.len += 1
        f.close()

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        zero = torch.zeros(size)
        if item == 0:
            x = torch.cat((zero, torch.Tensor(dict[self.word[item]]), torch.Tensor(dict[self.word[item + 1]])))
        elif 1 <= item < self.len - 1:
            x = torch.cat((torch.Tensor(dict[self.word[item - 1]]), torch.Tensor(dict[self.word[item]]),
                           torch.Tensor(dict[self.word[item + 1]])))
        else:
            x = torch.cat((torch.Tensor(dict[self.word[item - 1]]), torch.Tensor(dict[self.word[item]]), zero))
        y = int(self.label[item])
        return x.t(), y


class SoftMax(nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.layer = nn.Sequential(nn.Linear(size * 3, 3))

    def forward(self, input):
        return self.layer(input)


def get_softmax():
    softmax = SoftMax()
    if os.path.exists(state_file):
        softmax.load_state_dict(torch.load(state_file))
    return softmax, optim.SGD(softmax.parameters(), lr=learning_rate)


def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x), y)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)


def fit(epochs, model, loss_func, train_dl, valid_dl, opt, valid_ds, f1file, lossfile):
    for epoch in range(epochs):
        torch.save(obj=model.state_dict(), f="protected data/softmax.pth")
        print("epoch start")
        time_start = time.time()
        model.train()
        for x, y in train_dl:
            loss_batch(model, loss_func, x, y, opt)
        model.eval()
        sum_loss = 0
        sum_num = 0
        with torch.no_grad():
            for x, y in valid_dl:
                val_loss, loss_num = loss_batch(model, loss_func, x, y)
                sum_loss += val_loss
                sum_num += loss_num
        file_f1 = open(f1file, "a+", encoding="UTF-8")
        file_loss = open(lossfile, "a+", encoding="UTF-8")
        f1 = cal_F1_measure(model, valid_ds)
        file_f1.write(str(f1) + '\n')
        loss = sum_loss / sum_num
        file_loss.write(str(loss) + '\n')
        file_f1.close()
        file_loss.close()
        print("epoch : %d  loss : %f  valid_F1 : %f" % (epoch, loss, f1))
        time_end = time.time()
        print("time cost : %f" % (time_end - time_start))
    print("fit ends")


def cal_F1_measure(model, valid_ds):
    predict = []
    for x, y in valid_ds:
        print(model(x))
        exit(0)
        predict.append(model(x).argmax(dim=0))
    y = valid_ds.label
    TP = TN = FP = FN = 0
    i = 0
    while i < len(y):
        if y[i] == '0':
            if predict[i] == 0:
                TN += 1
            else:
                FP += 1
        elif y[i] == '1':
            if predict[i] != 1:
                FN += 1
            else:
                flag = True
                i += 1
                while y[i] == '2':
                    if predict[i] != 2:
                        flag = False
                        break
                    i += 1
                i -= 1
                if flag:
                    TP += 1
                else:
                    FN += 1
        i += 1
    if TP > 0:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        return 2 * P * R / (P + R)
    else:
        return 0


read_vector(word_vector_file)
train_ds = MyDataSet(train_set_file)
valid_ds = MyDataSet(valid_set_file)
train_dl = DataLoader(train_ds, batch_size=batch_size)
valid_dl = DataLoader(valid_ds, batch_size=batch_size * 4)

softmax, opt = get_softmax()
loss_func = F.cross_entropy

fit(epochs, softmax, loss_func, train_dl, valid_dl, opt, valid_ds, valid_f1_measure_file, loss_file)

# test_ds = MyDataSet("test_set.txt")
# f1 = cal_F1_measure(softmax, test_ds)
# print(str(f1))
