import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import time
import os

state_file = "protected data/lstm.pth"
word_vector_file = "ctb.50d.vec"
train_set_file = "train_set.txt"
valid_set_file = "validation_set.txt"
valid_f1_measure_file = "valid_f1_measure.txt"
loss_file = "loss.txt"

size = 50
batch_size = 5
learning_rate = 0.001
epochs = 20
dict = []
word_to_idx = {}


def read_vector(file_path):
    f = open(file_path, "r", encoding="UTF-8")
    cnt = 0
    for s in f:
        vec = s.split()
        word_to_idx[vec[0]] = cnt
        dict.append([])
        for i in range(1, len(vec)):
            dict[cnt].append(float(vec[i]))
        cnt += 1
    dict.append([])
    for i in range(0, len(dict[0])):
        dict[cnt].append(float(0))
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
            if raw[0] in word_to_idx:
                self.word.append(word_to_idx[raw[0]])
            else:
                self.word.append(word_to_idx["-unknown-"])
            self.label.append(int(raw[1]))
            self.len += 1
            # if self.len % 200 == 0:
            #     print(self.len)
        f.close()
        # print("?")
        self.senten_len = 10

    def __len__(self):
        return (self.len - 1) // self.senten_len + 1

    def __getitem__(self, item):
        x = []
        y = []
        s = item * self.senten_len
        e = min((item + 1) * self.senten_len, self.len)
        for i in range(s, e):
            x.append(self.word[i])
            y.append(self.label[i])
        for i in range(e - s, self.senten_len):
            x.append(int(len(dict) - 1))
            y.append(int(0))
        # print(x)
        if len(x) > self.senten_len:
            print(e, s)
        x = torch.LongTensor(x)
        # print(x)
        y = torch.LongTensor(y)
        return x, y


class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=size, hidden_size=100, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(100 * 2, 3)
        # self.embed = nn.Embedding(len(dict), size)
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(dict))

    def forward(self, input):
        # print(type(input), input)
        x = self.embed(input)
        h, c = self.lstm(x)
        return self.linear(h)


def get_LSTM():
    lstm = MyLSTM()
    if os.path.exists(state_file):
        lstm.load_state_dict(torch.load(state_file))
    opt = optim.Adam(lstm.parameters(), lr=learning_rate)
    exp_lr = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    return lstm, opt, exp_lr


def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x).view(-1, 3), y.view(-1).to(int))
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)


def fit(epochs, model, loss_func, train_dl, valid_dl, opt, exp_lr, f1file, lossfile):
    for epoch in range(epochs):
        print("epoch start")
        time_start = time.time()
        model.train()
        for x, y in train_dl:
            # print(x.shape)
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
            f1 = cal_F1_measure(model, valid_dl)
            file_f1.write(str(f1) + '\n')
            loss = sum_loss / sum_num
            file_loss.write(str(loss) + '\n')
        file_f1.close()
        file_loss.close()
        print("epoch : %d  loss : %f  valid_F1 : %f" % (epoch, loss, f1))
        torch.save(obj=model.state_dict(), f=state_file)
        exp_lr.step()
        time_end = time.time()
        print("time cost : %f" % (time_end - time_start))

    print("fit ends")


def cal_F1_measure(model, valid_dl):
    predict = []
    lable = []
    for x, y in valid_dl:
        # print(model(x).view(-1, 3))
        predict.extend(model(x).view(-1, 3).argmax(dim=1))
        lable.extend(y.view(-1))
    # print(lable)
    # print(predict)
    TP = TN = FP = FN = 0
    i = 0
    while i < len(lable):
        if lable[i] == 0:
            if predict[i] == 0:
                TN += 1
            else:
                FP += 1
        elif lable[i] == 1:
            if predict[i] != 1:
                FN += 1
            else:
                flag = True
                i += 1
                while lable[i] == 2:
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
    # print(TP)
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

# _x = valid_ds.word
# _y = valid_ds.label
# cnt = 0
# for i in range(0, valid_ds.len):
#     print(cnt)
#     cnt += 1

lstm, opt, exp_lr = get_LSTM()
loss_func = F.cross_entropy

fit(epochs, lstm, loss_func, train_dl, valid_dl, opt, exp_lr, valid_f1_measure_file, loss_file)

test_ds = MyDataSet("test_set.txt")
test_dl = DataLoader(test_ds, batch_size=batch_size * 4)
f1 = cal_F1_measure(lstm, test_dl)
print(str(f1))
