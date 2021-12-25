import torch
import time
from constant import *
from evaluate import trainPrecision

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


def fit(epochs, model, loss_func, train_dl, valid_dl, opt, exp_lr, precision_file, loss_file):
    for epoch in range(epochs):
        print("epoch start")
        time_start = time.time()
        model.train()
        for x, y in train_dl:
            # print(x.shape)
            # print(y.shape)
            loss_batch(model, loss_func, x, y, opt)
        model.eval()
        file_precision = open(precision_file, "a+", encoding="UTF-8")
        file_loss = open(loss_file, "a+", encoding="UTF-8")
        sum_loss = 0
        sum_num = 0
        with torch.no_grad():
            for x, y in valid_dl:
                val_loss, loss_num = loss_batch(model, loss_func, x, y)
                sum_loss += val_loss
                sum_num += loss_num
            precision = trainPrecision(model, valid_dl)
            file_precision.write(str(precision) + '\n')
            loss = sum_loss / sum_num
            file_loss.write(str(loss) + '\n')
        file_precision.close()
        file_loss.close()
        print("epoch : %d  loss : %f  precision : %f" % (epoch, loss, precision))
        # print("epoch : %d  loss : %f" % (epoch, loss))
        torch.save(obj=model.state_dict(), f=STATE_FILE)
        exp_lr.step()
        time_end = time.time()
        print("time cost : %f" % (time_end - time_start))

    print("fit ends")
