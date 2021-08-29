def cal_F1_measure(prediction, truth):
    TP = [0] * 10
    FP = [0] * 10
    FN = [0] * 10
    for i in range(0, len(prediction)):
        predict = prediction[i]
        correct = truth[i]
        if predict == correct:
            TP[predict] += 1
        else:
            FP[predict] += 1
            FN[correct] += 1
    macro_f1 = 0
    typ_occur = 10
    for i in range(0, 10):
        if TP[i] != 0:
            P = TP[i] / (TP[i] + FP[i])
            R = TP[i] / (TP[i] + FN[i])
            macro_f1 += (2 * P * R / (P + R))
        else:
            typ_occur -= 1
    macro_f1 /= typ_occur
    return macro_f1
