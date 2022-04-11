import torch
import numpy
import time
import os

size = 5000 + 1
batch = 256
learn_rate = 0.005
epochs = 200

train_set_name = "train_set.txt"
validation_set_name = "validation_set.txt"
test_set_name = "test_set.txt"
theta_name = "theta.txt"
dictionary_name = "dictionary.txt"
v_f1_measure_name = "vf1_measure.txt"
t_f1_measure_name = "tf1_measure.txt"
P_name = "P_measure.txt"
R_name = "R_measure.txt"


def read_theta():
    if os.path.exists(theta_name):
        file_theta = open(theta_name, "r", encoding="UTF-8")
    else:
        file_theta = open(theta_name, "w", encoding="UTF-8")
        tmp = torch.randn(1, size * 3, requires_grad=False)
        for i in range(size * 3):
            file_theta.write("%f\n" % tmp[0, i].item())
        file_theta.close()
        file_theta = open(theta_name, "r", encoding="UTF-8")
    theta = torch.zeros(1, size * 3, requires_grad=False)
    ptr = 0
    for s in file_theta:
        theta[0, ptr] += float(s)
        # print(theta[0, ptr].item())
        ptr += 1
    del ptr
    file_theta.close()
    return theta


def read_dictionary():
    file_dictionary = open(dictionary_name, "r", encoding="UTF-8")
    dist = {}
    index = 0
    for s in file_dictionary:
        # print(s[0:-1])
        dist[s[0:-1]] = index
        index += 1
    return dist


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + numpy.exp(-x))
    else:
        return numpy.exp(x) / (1 + numpy.exp(x))


x = []
y = []
theta = read_theta()
dictionary = read_dictionary()


def read_train_set():
    file_set = open(train_set_name, "r", encoding="UTF-8")
    cnt = 0
    for s in file_set:
        cnt += 1
        vector = s.split()[0:-1]
        data = s.split()[-1]
        xi = []
        for ele in vector:
            # print(ele)
            if ele in dictionary:
                xi.append(dictionary[ele])
            else:
                xi.append(size - 1)
        x.append(xi)
        y.append(data)
    file_set.close()
    return cnt


vx = []
vy = []


def read_validation_set():
    file_set = open(validation_set_name, "r", encoding="UTF-8")
    cnt = 0
    for s in file_set:
        cnt += 1
        vector = s.split()[0:-1]
        data = s.split()[-1]
        xi = []
        for ele in vector:
            if ele in dictionary:
                xi.append(dictionary[ele])
            else:
                xi.append(size - 1)
        vx.append(xi)
        vy.append(data)
    file_set.close()
    return cnt


count_validation = read_validation_set()

tx = []
ty = []


def read_test_set():
    file_set = open(test_set_name, "r", encoding="UTF-8")
    cnt = 0
    for s in file_set:
        cnt += 1
        vector = s.split()[0:-1]
        data = s.split()[-1]
        xi = []
        for ele in vector:
            if ele in dictionary:
                xi.append(dictionary[ele])
            else:
                xi.append(size - 1)
        tx.append(xi)
        ty.append(data)
    file_set.close()
    return cnt


count_test = read_test_set()


def v_cal_f1_measure():
    TP = FP = FN = TN = 0
    for i in range(0, count_validation):
        threewordsv = torch.zeros(1, size * 3)
        if i > 0:
            for ele in vx[i - 1]:
                threewordsv[0, ele] = 1
        for ele in vx[i]:
            threewordsv[0, ele + size] = 1
        if i + 1 < count_validation:
            for ele in vx[i + 1]:
                threewordsv[0, ele + size * 2] = 1
        temp = theta.mm(threewordsv.t())
        p = sigmoid(temp[0, 0].item())
        del threewordsv
        del temp
        if p >= 0.5:
            y_predict = '1'
        else:
            y_predict = '0'
        if y_predict == '1' and vy[i] == '1':
            TP += 1
        elif y_predict == '1' and vy[i] == '0':
            FP += 1
        elif y_predict == '0' and vy[i] == '1':
            FN += 1
        else:
            TN += 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print("P:" + str(P) + " R:" + str(R))
    return 2 * P * R / (P + R), P, R


def t_cal_f1_measure():
    TP = FP = FN = TN = 0
    for i in range(0, count_test):
        threewordsv = torch.zeros(1, size * 3)
        if i > 0:
            for ele in tx[i - 1]:
                threewordsv[0, ele] = 1
        for ele in tx[i]:
            threewordsv[0, ele + size] = 1
        if i + 1 < count_test:
            for ele in tx[i + 1]:
                threewordsv[0, ele + size * 2] = 1
        temp = theta.mm(threewordsv.t())
        p = sigmoid(temp[0, 0].item())
        del threewordsv
        del temp
        if p >= 0.5:
            y_predict = '1'
        else:
            y_predict = '0'
        if y_predict == '1' and ty[i] == '1':
            TP += 1
        elif y_predict == '1' and ty[i] == '0':
            FP += 1
        elif y_predict == '0' and ty[i] == '1':
            FN += 1
        else:
            TN += 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    return 2 * P * R / (P + R)


count = read_train_set()

for epoch in range(epochs):
    print("epoch start\n")
    time_start = time.time()
    l = 1
    r = 0
    while l < count - 1:
        r = min(l + batch, count - 1)
        grad = torch.zeros(1, size * 3)
        for i in range(l, r):
            threewordsv = torch.zeros(1, size * 3)
            for ele in x[i - 1]:
                threewordsv[0, ele] = 1
            for ele in x[i]:
                threewordsv[0, ele + size] = 1
            for ele in x[i + 1]:
                threewordsv[0, ele + size * 2] = 1
            temp = theta.mm(threewordsv.t())
            grad = torch.add(grad, float(learn_rate * (float(y[i]) - sigmoid(temp[0, 0].item()))) * threewordsv)
            del threewordsv
            del temp
        theta = torch.add(theta, grad)
        del grad
        l += batch

    file_theta = open(theta_name, 'w', encoding='UTF-8')
    for i in range(size * 3):
        file_theta.write(str(theta[0, i].item()) + '\n')
    file_theta.close()

    vf1, P, R = v_cal_f1_measure()
    tf1 = t_cal_f1_measure()
    print("vf1 : %f  tf1 : %f\n" % (vf1, tf1))
    file_vf1 = open(v_f1_measure_name, 'a+', encoding='UTF-8')
    file_vf1.write(str(vf1) + '\n')
    file_vf1.close()
    file_tf1 = open(t_f1_measure_name, 'a+', encoding='UTF-8')
    file_tf1.write(str(tf1) + '\n')
    file_tf1.close()
    file_P = open(P_name, 'a+', encoding='UTF-8')
    file_P.write(str(P) + '\n')
    file_P.close()
    file_R = open(R_name, 'a+', encoding='UTF-8')
    file_R.write(str(R) + '\n')
    file_R.close()
    time_end = time.time()
    print("%d epoch time cost : %f\n" % (epoch, time_end - time_start))
