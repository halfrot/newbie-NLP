import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset


# get word2vec
def read_glove(file, emb, word_to_idx, cnt):
    f = open(file, "r", encoding="UTF-8")
    for s in f:
        vec = s.split()
        word_to_idx[vec[0]] = cnt
        emb.append([])
        for i in range(1, len(vec)):
            emb[cnt].append(float(vec[i]))
        cnt += 1
    word_to_idx["-unknown-"] = cnt
    emb.append([])
    for i in range(0, len(emb[0])):
        emb[cnt].append(float(0))
    f.close()


# define class MyDataset
class MyDataSet(TensorDataset):
    def __init__(self, file_x, file_y, word_to_idx, emb):
        super(MyDataSet, self).__init__()
        # read x
        self.identify = {"e11", "e12", "e21", "e22"}
        self.text = []
        self.num = []
        self.pos = []
        self.len = 0
        f = open(file_x, "r", encoding="UTF-8")
        for s in f:
            sentence = []
            raw = s.split()
            self.num += [int(raw[0])]
            spos = {}
            i = 0
            for word in raw[1:]:
                if word in self.identify:
                    spos[word] = i
                if word in word_to_idx:
                    sentence.append(word_to_idx[word])
                else:
                    sentence.append(word_to_idx["-unknown-"])
                i += 1
            self.text.append(sentence)
            self.pos.append(spos)
            self.len += 1
        f.close()

        # read y
        self.relation = {}
        self.relation2idx = {}
        self.relation_cnt = 0
        f = open(file_y, "r", encoding="UTF-8")
        for s in f:
            raw = s.split()
            if raw[1] in self.relation2idx:
                self.relation[int(raw[0])] = self.relation2idx[raw[1]]
            else:
                self.relation2idx[raw[1]] = self.relation_cnt
                self.relation[int(raw[0])] = self.relation2idx[raw[1]]
                self.relation_cnt += 1

        f.close()

        self.Emb = nn.Embedding.from_pretrained(torch.FloatTensor(emb), freeze=True)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        x = []
        # word feature dim=50
        wf_sentence = self.Emb(torch.LongTensor(self.text[item]))

        # final dim=56
        for i in range(0, len(wf_sentence)):
            x.append([])
            x[i] = torch.cat((wf_sentence[i], torch.zeros(6)))

        # position identify dim=54
        x[self.pos[item]["e11"]] = torch.cat((torch.zeros(50), torch.LongTensor([1, 0, 0, 0, 0, 0])))
        x[self.pos[item]["e12"]] = torch.cat((torch.zeros(50), torch.LongTensor([0, 1, 0, 0, 0, 0])))
        x[self.pos[item]["e21"]] = torch.cat((torch.zeros(50), torch.LongTensor([0, 0, 1, 0, 0, 0])))
        x[self.pos[item]["e22"]] = torch.cat((torch.zeros(50), torch.LongTensor([0, 0, 0, 1, 0, 0])))

        # position feature dim=56
        for i in range(0, len(x)):
            x[i][54] = i - self.pos[item]["e11"]
            x[i][55] = i - self.pos[item]["e21"]
            x[i] = x[i].numpy().tolist()

        x = torch.Tensor(x)
        y = self.relation[self.num[item]]
        return x, y
