from segment_main import DictBasedSPM

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
    out_cnt = 0
    ans_cnt = 0
    both_cnt = 0
    d = set()
    for out_line in out:
        ans_line = ans.readline()
        out_words = out_line.split()
        ans_words = ans_line.split()
        out_cnt += len(out_words)
        ans_cnt += len(ans_words)
        d.clear()
        p = 1
        for ele in out_words:
            d.add((p, p + len(ele) - 1))
            p += len(ele)
        p = 1
        for ele in ans_words:
            if (p, p + len(ele) - 1) in d:
                both_cnt += 1
            p += len(ele)
    P = both_cnt / out_cnt
    R = both_cnt / ans_cnt
    print("P: " + str(P))
    print("R: " + str(R))
    F1 = 2 * P * R / (P + R)
    return F1


"""
测试

flag = True
"
total time:  2.1166435999999136
P: 0.9815469981416267
R: 0.9740780415641468
0.9777982570695761
"

flag = False
"
total time:  2.2259895999999477
P: 0.8451154517196089
R: 0.8933228343335312
0.8685507408873829
"

"""
if __name__ == "__main__":
    test = DictBasedSPM()
    fin = open("segment_nospace", "r", encoding="UTF-8")
    out = open("segment_out", "w", encoding="UTF-8")
    total_cost = 0
    for s in fin:
        s = s[:-1]
        # test.setSentence(s, flag=True)
        test.setSentence(s, flag=False)
        total_cost += test.train()
        out.write(test.return_s())
        out.write("\n")
    fin.close()
    out.close()
    print("total time: ", total_cost)
    print(getF1("segment_out", "segment_space"))
