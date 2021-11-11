import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import time
from ner_constant import *
import ner_dataset
import ner_F1
import ner_model

"""
function loss_batch
对一个batch的数据进行传播迭代，训练参数

model为模型
loss_func为损失函数
x为预测标签
y为标准标签
opt为模型optimizer
"""


def loss_batch(model, loss_func, x, y, opt=None):
    # print(model(x).view(-1, 3))
    # print(y.size())
    loss = loss_func(model(x).view(-1, 3), y.view(-1).to(int))
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)


"""
function fit
模型训练函数，使用数据集对模型进行训练和效率跟踪

epochs为总训练轮数
model为模型
loss_func为损失函数
train_dl为训练集
valid_dl为验证集
opt为模型optimizer
exp_lr为lr_scheduler
f1file、loss_file为保存f1、loss的文件
"""


def fit(epochs, model, loss_func, train_dl, valid_dl, opt, exp_lr, f1file, loss_file):
    for epoch in range(epochs):
        print("epoch start")
        time_start = time.time()
        model.train()
        for x, y in train_dl:
            # print(x.shape)
            # print(y.shape)
            loss_batch(model, loss_func, x, y, opt)
        model.eval()
        file_f1 = open(f1file, "a+", encoding="UTF-8")
        file_loss = open(loss_file, "a+", encoding="UTF-8")
        sum_loss = 0
        sum_num = 0
        with torch.no_grad():
            for x, y in valid_dl:
                val_loss, loss_num = loss_batch(model, loss_func, x, y)
                sum_loss += val_loss
                sum_num += loss_num
            f1 = ner_F1.cal_F1_measure(model, valid_dl)
            file_f1.write(str(f1) + '\n')
            loss = sum_loss / sum_num
            file_loss.write(str(loss) + '\n')
        file_f1.close()
        file_loss.close()
        print("epoch : %d  loss : %f  valid_F1 : %f" % (epoch, loss, f1))
        torch.save(obj=model.state_dict(), f=STATE_FILE)
        exp_lr.step()
        time_end = time.time()
        print("time cost : %f" % (time_end - time_start))

    print("fit ends")


"""
测试

Attention Based Bi_LSTM
"
epoch : 0  loss : 0.039942  valid_F1 : 0.615137
time cost : 2028.711767
epoch : 1  loss : 0.041523  valid_F1 : 0.635550
time cost : 2031.803820
epoch : 2  loss : 0.036702  valid_F1 : 0.688464
time cost : 2473.315319
epoch : 3  loss : 0.039811  valid_F1 : 0.695464
time cost : 2379.467796
epoch : 4  loss : 0.038918  valid_F1 : 0.700959
time cost : 1725.222917
fit ends
test_set
P :0.7216699801192843 R: 0.6703601108033241
F1:0.6950694112015319
"

Bi_LSTM
"
epoch : 0  loss : 0.007826  valid_F1 : 0.616168
time cost : 285.198637
epoch : 1  loss : 0.006314  valid_F1 : 0.692699
time cost : 507.337503
epoch : 2  loss : 0.006052  valid_F1 : 0.702760
time cost : 494.685615
epoch : 3  loss : 0.006439  valid_F1 : 0.701031
time cost : 489.333487
epoch : 4  loss : 0.007330  valid_F1 : 0.710609
time cost : 504.607309
fit ends
P :0.7351778656126482 R :0.6869806094182825
F1:0.7102625298329357
"

"""
if __name__ == "__main__":
    lstm, opt, exp_lr = ner_model.get_LSTM()
    loss_func = F.cross_entropy

    # train_ds = ner_dataset.MyDataSet(TRAIN_SET_FILE)
    # valid_ds = ner_dataset.MyDataSet(VALID_SET_FILE)
    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
    # valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)

    # fit(EPOCHS, lstm, loss_func, train_dl, valid_dl, opt, exp_lr, VALID_F1_MEASURE_FILE, LOSS_FILE)

    test_ds = ner_dataset.MyDataSet("test_set.txt")
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    f1 = ner_F1.cal_F1_measure(lstm, test_dl)
    print(str(f1))
