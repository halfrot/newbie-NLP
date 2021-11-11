from 词性标注prework import getFrequency
import time


class MyDict:
    def __init__(self):
        self.cnt, self.mat, self.type_cnt = getFrequency()
        self.abs = 0
        self.cnt["unk"] = {}
        self.cnt["unk"]["?"] = self.abs
        self.type_cnt["?"] = self.abs

    def getCnt(self, word):
        if word not in self.cnt:
            # print(word)
            word = "unk"
        return self.cnt[word]

    def getMat(self, p1, p2):
        if p1 not in self.mat or p2 not in self.mat[p1]:
            return self.abs
        return self.mat[p1][p2] + self.abs

    def getTypeCnt(self, part_of_speech):
        # if part_of_speech not in self.type_cnt:
        #     return 0
        return self.type_cnt[part_of_speech]


class Token:
    def __init__(self):
        self.delta = {}
        self.word = ""


class ViterbiModel:
    def __init__(self):
        self.sentence = ""
        self.len = 0
        self.token_list = []
        self.dict = MyDict()
        self.answer = []

    def setSentence(self, s):
        self.sentence = s.split()
        self.len = len(self.sentence)
        self.token_list.clear()
        self.answer.clear()

    def train(self):
        print("---------ViterbiModel train starts---------")
        begin_time = time.clock()
        self.token_list.append(Token())
        # print(self.sentence)
        self.token_list[0].word = self.sentence[0]
        temp_cnt = self.dict.getCnt(self.token_list[0].word)
        for ele in temp_cnt:
            # self.token_list[0].delta[ele] = (temp_cnt[ele] / self.dict.getTypeCnt(ele), "#")
            self.token_list[0].delta[ele] = (1, "#")
        pre_cnt = temp_cnt
        del temp_cnt
        for i in range(1, self.len):
            # print("{:d}".format(i))
            self.token_list.append(Token())
            self.token_list[i].word = self.sentence[i]
            now_cnt = self.dict.getCnt(self.token_list[i].word)
            for now_ele in now_cnt:
                for pre_ele in pre_cnt:
                    value = self.token_list[i - 1].delta[pre_ele][0] * (
                            self.dict.getMat(pre_ele, now_ele) / self.dict.getTypeCnt(pre_ele)) * (
                                    now_cnt[now_ele] / self.dict.getTypeCnt(now_ele))
                    # print(pre_ele, now_ele, self.token_list[i - 1].delta[pre_ele][0],
                    #       self.dict.getMat(pre_ele, now_ele), self.dict.getTypeCnt(pre_ele),
                    #       now_cnt[now_ele], self.dict.getTypeCnt(now_ele), value)
                    if now_ele not in self.token_list[i].delta:
                        self.token_list[i].delta[now_ele] = (value, pre_ele)
                    elif self.token_list[i].delta[now_ele][0] < value:
                        self.token_list[i].delta[now_ele] = (value, pre_ele)
            pre_cnt = now_cnt
        u = "NULL"
        # print(self.token_list[self.len - 1].word)
        for ele in self.token_list[self.len - 1].delta:
            # print(ele)
            # print(self.token_list[self.len - 1].delta[ele])
            if u == "NULL":
                u = ele
            elif self.token_list[self.len - 1].delta[ele][0] > self.token_list[self.len - 1].delta[u][0]:
                u = ele
        self.answer.append(u)
        p = self.len - 1
        while p > 0:
            u = self.token_list[p].delta[u][1]
            self.answer.append(u)
            # print(u)
            p -= 1
        self.answer.reverse()
        # print(self.answer)
        end_time = time.clock()
        print("---------ViterbiModel train ends---------")
        print("time cost:  {:f}".format(end_time - begin_time))

    def return_s(self):
        s = ""
        for i in range(self.len):
            s += self.sentence[i] + "/" + self.answer[i]
            s += " "
        return s


if __name__ == "__main__":
    a = ViterbiModel()
    # a.setSentence("迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ） ")
    a.setSentence("我 今 晚 做 完 了 一 项 作业")
    a.train()
    print(a.return_s())
