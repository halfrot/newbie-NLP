from constant import *

"""
function trainPrecision
用于评估模型训练过程中的准确率
model为输入模型
valid_dl为验证数据集

返回准确率
"""


def calPrecision(model, embed, dict_list, word_to_idx):
    fileValid = open(VALID_SET_FILE, "r", encoding="UTF-8")
    predict = []
    x = []
    y = []
    cnt = 0
    total = 0
    correct = 0
    for s in fileValid:
        s = s.split()
        if s[0] == '$':
            total += 1
            for i in range(0, cnt):
                predict.append(model(x[i].unsqueeze(0)).view(-1, 2)[0][1])
            max_idx = 0
            for i in range(0, cnt):
                if predict[i] > predict[max_idx]:
                    max_idx = i
            if y[max_idx] == 1:
                correct += 1
            predict.clear()
            x.clear()
            y.clear()
            cnt = 0
        else:
            y.append(int(s[0]))
            s = s[1:]
            u = list()
            for word in s:
                if word in word_to_idx:
                    u.append(word_to_idx[word])
                else:
                    u.append(word_to_idx["-unknown-"])
            x.append(embed(torch.tensor(u)))
            cnt += 1
    fileValid.close()
    return correct / total
