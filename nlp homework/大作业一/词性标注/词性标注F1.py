from 词性标注 import ViterbiModel


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
        # if words_cnt != len(ans_words):
        #     print(out_line)
        #     print(ans_line)
        #     break
        for i in range(words_cnt):
            # print(out_words[i] + " " + ans_words[i])
            if out_words[i] == ans_words[i]:
                correct_cnt += 1
            total += 1
    P = R = F1 = correct_cnt / total
    print("P: " + str(P))
    print("R: " + str(R))
    return F1


"""
测试
"""
if __name__ == "__main__":
    test = ViterbiModel()
    fin = open("WashedInput", "r", encoding="UTF-8")
    out = open("tag_out", "w", encoding="UTF-8")
    for s in fin:
        s = s[:-1]
        test.setSentence(s)
        test.train()
        out.write(test.return_s())
        out.write("\n")
    fin.close()
    out.close()
    print(getF1("tag_out", "WashedAnswer"))
