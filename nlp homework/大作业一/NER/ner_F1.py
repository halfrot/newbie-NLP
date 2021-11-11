"""
function cal_F1_measure
用于评估模型的输出结果
model为输入模型
valid_dl为验证数据集

打印P和R并返回F1值
"""


def cal_F1_measure(model, valid_dl):
    predict = []
    label = []
    for x, y in valid_dl:
        # print(model(x).view(-1, 3))
        predict.extend(model(x).view(-1, 3).argmax(dim=1))
        label.extend(y.view(-1))
    # print(label)
    # print(predict)
    TP = TN = FP = FN = 0
    i = 0
    while i < len(label):
        if label[i] == 0:
            if predict[i] == 0:
                TN += 1
            else:
                FP += 1
        elif label[i] == 1:
            if predict[i] != 1:
                FN += 1
            else:
                flag = True
                i += 1
                while label[i] == 2:
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
        print("P :", P)
        print("R :", R)
        return 2 * P * R / (P + R)
    else:
        return 0
