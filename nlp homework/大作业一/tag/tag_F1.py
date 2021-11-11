from tag_main import ViterbiModel

"""
function getF1
用于评估模型的输出结果
out_path为模型输出
ans_path为标准输出

打印P和R并返回F1值
"""


def getF1(out_path, ans_path):
    out = open(out_path, "r", encoding="UTF-8")
    ans = open(ans_path, "r", encoding="UTF-8")
    correct_cnt = 0
    total = 0
    for out_line in out:
        ans_line = ans.readline()
        out_words = out_line.split()
        ans_words = ans_line.split()
        words_cnt = len(out_words)
        for i in range(words_cnt):
            if out_words[i] == ans_words[i]:
                correct_cnt += 1
            total += 1
    P = R = F1 = correct_cnt / total
    print("P: " + str(P))
    print("R: " + str(R))
    return F1


"""
测试

"
total time:  6.286844000000166
P: 0.9555757034256943
R: 0.9555757034256943
0.9555757034256943
"
"""
if __name__ == "__main__":
    test = ViterbiModel()
    fin = open("WashedInput", "r", encoding="UTF-8")
    out = open("tag_out", "w", encoding="UTF-8")
    total_cost = 0
    for s in fin:
        s = s[:-1]
        test.setSentence(s)
        total_cost += test.train()
        out.write(test.return_s())
        out.write("\n")
    fin.close()
    out.close()
    print("total time: ", total_cost)
    print(getF1("tag_out", "WashedAnswer"))
