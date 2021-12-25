import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from constant import *
from prework import read_word2vec

"""
class MyDataSet
继承自torch.utils.data.dataset.TensorDataset
自定义数据集

sentence，label存放词和标准标签
len表示句子总数
dict_list，word_to_idx为预训练的embedding表示
embed为nn.Embedding类型
"""


class MyDataSet(TensorDataset):
    """
    function __init__
    初始化
    """

    def __init__(self, file_path):
        super(MyDataSet, self).__init__()
        self.sentence = []
        self.label = []
        self.len = 0
        self.dict_list, self.word_to_idx = read_word2vec(WORD_TO_VECTOR_FILE)
        # self.embed = nn.Embedding(len(self.dict_list), size)
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(self.dict_list))
        f = open(file_path, "r", encoding="UTF-8")
        for s in f:
            raw = s.split()
            self.label.append(int(raw[0]))
            raw = raw[:1]
            u = list()
            for word in raw:
                if word in self.word_to_idx:
                    u.append(self.word_to_idx[word])
                else:
                    u.append(self.word_to_idx["-unknown-"])
            self.sentence.append(u)
            self.len += 1
        f.close()

    """
    function __len__
    重载数据组数
    """

    def __len__(self):
        return self.len

    """
    function __getitem__
    返回单位数据
    """

    def __getitem__(self, item):
        x = torch.LongTensor(self.sentence[item])
        y = torch.LongTensor(self.label[item])
        # print(x.shape)
        # print(y.shape)
        return self.embed(x), y
