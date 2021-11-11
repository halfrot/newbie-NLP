import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from ner_prework import read_vector
from ner_constant import *

"""
class MyDataSet
继承自torch.utils.data.dataset.TensorDataset
自定义数据集

word，label存放词和标准标签
len表示word中词的数量
dict_list，word_to_idx为预训练的embedding表示
embed为nn.Embedding类型
sentence_len为单位数据的句子长度
"""


class MyDataSet(TensorDataset):
    """
    function __init__
    初始化
    """

    def __init__(self, file_path):
        super(MyDataSet, self).__init__()
        self.word = []
        self.label = []
        self.len = 0
        self.dict_list, self.word_to_idx = read_vector(WORD_VECTOR_FILE)
        # self.embed = nn.Embedding(len(self.dict_list), size)
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(self.dict_list))
        f = open(file_path, "r", encoding="UTF-8")
        for s in f:
            raw = s.split()
            if raw[0] in self.word_to_idx:
                self.word.append(self.word_to_idx[raw[0]])
            else:
                self.word.append(self.word_to_idx["-unknown-"])
            self.label.append(int(raw[1]))
            self.len += 1
        f.close()
        self.sentence_len = 10

    """
    function __len__
    重载数据组数
    """

    def __len__(self):
        return (self.len - 1) // self.sentence_len + 1

    """
    function __getitem__
    返回单位数据
    """

    def __getitem__(self, item):
        x = []
        y = []
        s = item * self.sentence_len
        e = min((item + 1) * self.sentence_len, self.len)
        for i in range(s, e):
            x.append(self.word[i])
            y.append(self.label[i])
        for i in range(e - s, self.sentence_len):
            x.append(int(len(self.dict_list) - 1))
            y.append(int(0))
        if len(x) > self.sentence_len:
            print(e, s)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        # print(x.shape)
        # print(y.shape)
        return self.embed(x), y
