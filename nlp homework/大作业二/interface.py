import torch
from torch import nn
from model import get_Attention_BiLSTM
from constant import *
from prework import read_word2vec


class MyWSD:
    def __init__(self):
        super(MyWSD, self).__init__()
        self.myNet = get_Attention_BiLSTM(deploy=True)
        self.dict_list, self.word_to_idx = read_word2vec(WORD_TO_VECTOR_FILE)
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(self.dict_list))
        self.answer = -1

    def setSentence(self, input):
        cnt = len(input)
        x = []
        for s in input:
            u = list()
            for word in s:
                if word in self.word_to_idx:
                    u.append(self.word_to_idx[word])
                else:
                    u.append(self.word_to_idx["-unknown-"])
            x.append(self.embed(torch.tensor(u)))
        predict = []
        for i in range(0, cnt):
            predict.append(self.myNet(x[i].unsqueeze(0)).view(-1, 2)[0][1])
        max_idx = 0
        for i in range(0, cnt):
            if predict[i] > predict[max_idx]:
                max_idx = i
        self.answer = max_idx

    def getAnswer(self):
        return self.answer
