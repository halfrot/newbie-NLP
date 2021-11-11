from tag_prework import getFrequency
import time

"""
class MyDict
自定义词典类型

cnt表示词语/词性频度表
mat表示词性转移矩阵
type_cnt表示词性频度表
abs为绝对增量
"""


class MyDict:
    """
    function __init__

    初始化
    """

    def __init__(self):
        self.cnt, self.mat, self.type_cnt = getFrequency()
        self.abs = 0
        self.cnt["unk"] = {}
        self.cnt["unk"]["?"] = self.abs
        self.type_cnt["?"] = self.abs + 1

    """
    function getCnt

    未登录词返回cnt["unk"]，否则返回对应cnt
    """

    def getCnt(self, word):
        if word not in self.cnt:
            # print(word)
            word = "unk"
        return self.cnt[word]

    """
    function getMat

    未登录词性转移返回abs，否则返回对应mat
    """

    def getMat(self, p1, p2):
        if p1 not in self.mat or p2 not in self.mat[p1]:
            return self.abs
        return self.mat[p1][p2] + self.abs

    """
    function getMat

    未登录词性返回type_cnt["?"]，否则返回对应type_cnt
    """

    def getTypeCnt(self, part_of_speech):
        if part_of_speech not in self.type_cnt:
            part_of_speech = "?"
        return self.type_cnt[part_of_speech]


"""
class Token
自定义词类型

delta存放当前词性最大概率
元素格式（到当前词性的概率，上一个词的词性）
word表示词内容
"""


class Token:
    """
    function __init__

    初始化
    """

    def __init__(self):
        self.delta = {}
        self.word = ""


"""
class ViterbiModel
Viterbi算法训练HMM模型

sentence表示待标注句子
len表示句子中词的个数
token_list为存放Token类型的List，将句子中的词作为Token依次存放
dict为MyDict类型的词典
answer表示词性标注结果
"""


class ViterbiModel:
    """
    function __init__

    初始化
    """

    def __init__(self):
        self.sentence = ""
        self.len = 0
        self.token_list = []
        self.dict = MyDict()
        self.answer = []

    """
    function setSentence
    设置待标注的句子
    
    初始化token_list、answer
    """

    def setSentence(self, s):
        self.sentence = s.split()
        self.len = len(self.sentence)
        self.token_list.clear()
        self.answer.clear()

    """
    function train
    对模型进行训练，训练后的标注结果保存在self.answer中
    返回train花费的时间
    """

    def train(self):
        print("---------ViterbiModel train starts---------")
        begin_time = time.clock()
        self.token_list.append(Token())
        self.token_list[0].word = self.sentence[0]
        temp_cnt = self.dict.getCnt(self.token_list[0].word)
        for ele in temp_cnt:
            # self.token_list[0].delta[ele] = (temp_cnt[ele] / self.dict.getTypeCnt(ele), "#")
            self.token_list[0].delta[ele] = (1, "#")
        pre_cnt = temp_cnt
        del temp_cnt
        for i in range(1, self.len):
            self.token_list.append(Token())
            self.token_list[i].word = self.sentence[i]
            now_cnt = self.dict.getCnt(self.token_list[i].word)
            for now_ele in now_cnt:
                for pre_ele in pre_cnt:
                    value = self.token_list[i - 1].delta[pre_ele][0] * (
                            self.dict.getMat(pre_ele, now_ele) / self.dict.getTypeCnt(pre_ele)) * (
                                    now_cnt[now_ele] / self.dict.getTypeCnt(now_ele))
                    if now_ele not in self.token_list[i].delta:
                        self.token_list[i].delta[now_ele] = (value, pre_ele)
                    elif self.token_list[i].delta[now_ele][0] < value:
                        self.token_list[i].delta[now_ele] = (value, pre_ele)
            pre_cnt = now_cnt
        u = "NULL"
        for ele in self.token_list[self.len - 1].delta:
            if u == "NULL":
                u = ele
            elif self.token_list[self.len - 1].delta[ele][0] > self.token_list[self.len - 1].delta[u][0]:
                u = ele
        self.answer.append(u)
        p = self.len - 1
        while p > 0:
            u = self.token_list[p].delta[u][1]
            self.answer.append(u)
            p -= 1
        self.answer.reverse()
        end_time = time.clock()
        print("---------ViterbiModel train ends---------")
        print("time cost:  {:f}".format(end_time - begin_time))
        return end_time - begin_time

    """
    function return_s
    以字符串形式返回词语/词性标注结果
    """

    def return_s(self):
        s = ""
        for i in range(self.len):
            s += self.sentence[i] + "/" + self.answer[i]
            s += " "
        return s


if __name__ == "__main__":
    a = ViterbiModel()
    # a.setSentence("迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ） ")
    # a.train()
    a.setSentence("我 今 晚 做 完 了 一 项 作业")
    a.train()
    print(a.return_s())
