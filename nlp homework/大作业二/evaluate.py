def trainPrecision(model, valid_dl):
    predict = []
    label = []
    for x, y in valid_dl:
        # print(model(x).view(-1, 2))
        predict.extend(model(x).view(-1, 2).argmax(dim=1))
        label.extend(y.view(-1))
    # print(label)
    # print(predict)
    total = len(label)
    correct = 0
    for i in range(0, total):
        if predict[i] == label[i]:
            correct += 1
    return correct / total


"""
function trainPrecision
用于评估模型训练过程中的准确率
model为输入模型
valid_dl为验证数据集

返回准确率
"""
