from segment_prework import getDict
import os
import collections
import numpy as np
import time

# 默认词典文本路径
DICTIONARY_PATH = "dictionary.utf8"

"""
class DictBasedSPM
最短路径分词模型

该模型可根据需要分别运行以下两种算法
基于词典的最少分词算法
基于统计的最大路径概率算法

dag存放图的信息

max_dis存放最大概率/最短路径
元素格式（到当前字最大概率/最短路径，当前词的起点）

sentence表示待分词句子
len表示待分词句子的长度s
dict表示词典
max_word表示最长词的长度

answer表示分词结果
元素格式（词首位置，词尾位置）
"""


class DictBasedSPM:
    """
    function __init__

    初始化
    """

    def __init__(self):

        self.dag = {}
        self.max_dis = {}
        self.sentence = ""
        self.len = 0
        self.dict = {}
        self.max_word = 22
        self.answer = []
        self.setDict()

    """
    function setDict
    设置词典，读入到模型中
    """

    def setDict(self):
        if os.path.exists("myDict"):
            f = open("myDict", "r", encoding="UTF-8")
            for line in f:
                word = line.split()
                self.dict[word[0]] = int(word[1])
            f.close()
        else:
            self.dict = getDict(DICTIONARY_PATH)

    """
    function setSentence
    设置待分词的句子
    flag用于选择模型算法，默认值False
    flag=False，则采用最大路径概率法
    flag=True，则采用最少分词法
    
    初始化dag、max_dis、answer
    """

    def setSentence(self, s, flag=False):
        self.sentence = s
        self.len = len(s)
        self.dag.clear()
        self.max_dis.clear()
        self.answer.clear()
        self.setDag(flag)

    """
    function add
    连接一条a->b的边，边权为c
    """

    def add(self, a, b, c, flag):
        # 最少分词法
        if flag:
            c = -1
        if a not in self.dag:
            self.dag[a] = collections.deque()
        self.dag[a].append((b, c))

    """
    function setDag
    对文本构图
    """

    def setDag(self, flag):
        for i in range(self.len):
            for word_len in range(0, self.max_word):
                if i + word_len >= self.len:
                    break
                word = self.sentence[i:i + word_len + 1]
                if word in self.dict:
                    self.add(i, i + word_len + 1, np.log(1.0 / self.dict.get(word)).item(), flag)
                elif word_len == 0:
                    self.add(i, i + word_len + 1, 0, flag)

    """
    function train
    对模型进行训练，训练后的分词结果保存在self.answer中
    返回train花费的时间
    """

    def train(self):
        print("---------DictBasedSPM train starts---------")
        begin_time = time.clock()
        vis = dict()
        self.max_dis[0] = (0, -1)
        vis[0] = True
        for u in range(self.len):
            while bool(self.dag[u]):
                v, w = self.dag[u].pop()
                if v not in vis:
                    vis[v] = True
                    self.max_dis[v] = (self.max_dis[u][0] + w, u)
                    continue
                if self.max_dis[v][0] < (self.max_dis[u][0] + w):
                    self.max_dis[v] = (self.max_dis[u][0] + w, u)
        u = self.len
        while u >= 0:
            self.answer.append((self.max_dis[u][1], u - 1))
            u = self.max_dis[u][1]
        self.answer.reverse()
        end_time = time.clock()
        print("---------DictBasedSPM train ends---------")
        print("time cost:  {:f}".format(end_time - begin_time))
        return end_time - begin_time

    """
    function return_s
    以字符串形式返回分词结果
    """

    def return_s(self):
        s = ""
        for ele in self.answer:
            s += self.sentence[ele[0]:ele[1] + 1]
            s += " "
        return s

    """
    function write
    将分词结果写入到filepath文件中
    """

    def write(self, filepath):
        f = open(filepath, "a+", encoding="UTF-8")
        for ele in self.answer:
            f.write(self.sentence[ele[0]:ele[1] + 1])
            f.write(" ")
        f.write("\n")
        f.close()


"""
测试
"""

if __name__ == "__main__":
    test = DictBasedSPM()
    test.setDict()
    test.setSentence("颜老师好，西北农林科技大学收到的包裹里只有3个队的物品，少一支队伍的物品。请您看是不是漏发了。")
    test.train()
    # test.setSentence("１２月３１日，中共中央总书记、国家主席江泽民发表１９９８年新年讲话《迈向充满希望的新世纪》。（新华社记者兰红光摄")
    # test.train()
    print(test.return_s())
