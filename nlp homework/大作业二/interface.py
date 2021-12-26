import random
import copy
from typing import Tuple
from torch import nn
from model import get_Attention_BiLSTM
from constant import *
from prework import read_word2vec

"""
class MyWSD
提供了调用模型的接口
"""


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
        self.answer = input[max_idx]

    def getAnswer(self):
        return self.answer


"""
class SentenceSubstitution
用于将句子中的歧义词替换，并生成多个句子
"""


class SentenceSubstitution:

    def __init__(self, cilin_path: str):
        self.syn_dict = {}
        self.syn_sense = {}
        self.cilin_process(cilin_path)

    def do_substitution(self, index: int, sentence: list) -> Tuple[int, list]:
        random.seed(0)
        lines = []
        synonym = sentence[index]
        if synonym not in self.syn_dict:
            return -1, []
        senses = self.syn_dict[synonym]
        for sense in senses:
            if sense not in self.syn_sense:
                continue
            new_sentence = copy.deepcopy(sentence)
            new_sentence[index] = SentenceSubstitution.random_pick_ex(self.syn_sense[sense], synonym)
            lines.append(new_sentence)
        return 0, lines

    def cilin_process(self, cilin_path: str):
        cilin = open(cilin_path, 'r', encoding='utf-8')

        t_syn_dict = {}

        for line in cilin:
            # Skip the empty line
            if line == '':
                continue
            # Split the text by [space]
            sep = line.strip().split(' ')
            # Skip if not the synonym
            if not sep[0].endswith('='):
                continue
            # Read the type and the words
            sense, words = sep[0], sep[1:]
            # Trim the sense for each word from length of 8 to length
            # of 4 in order to align with the sense format of `Sense_POS.txt`
            sense = sense[0:4]

            # Everytime a new sense come in, update the `syn_sense`
            if sense in self.syn_sense:
                for word in words:
                    if self.syn_sense[sense].count(word) == 0:
                        self.syn_sense[sense].append(word)
            else:
                self.syn_sense[sense] = words

            # Iterate through the `words` with sense of `sense`,
            # and insert the sense into the dictionary respectively.
            for word in words:
                if word in t_syn_dict:
                    # Word has been added to the dictionary with 
                    # different sense, so add the new sense to the 
                    # word.
                    if t_syn_dict[word].count(sense) == 0:
                        t_syn_dict[word].append(sense)
                else:
                    # Word hasn't been found before, so we add it to 
                    # the dictionary and create a sense list for it.
                    t_syn_dict[word] = [sense]

        for key in t_syn_dict:
            if len(t_syn_dict[key]) > 1:
                self.syn_dict[key] = t_syn_dict[key]
        cilin.close()

    @staticmethod
    def random_pick(collection: list) -> str:
        return collection[random.randint(0, len(collection) - 1)]

    # Exclusively picking another item from list
    @staticmethod
    def random_pick_ex(collec: list, ex_item: str) -> str:
        offset = random.randint(1, len(collec) - 1)
        ex_index = collec.index(ex_item)
        return collec[(ex_index + offset) % len(collec)]


"""
function WSD
用于调用接口
"""


def WSD(wordList, index, model):
    t = SentenceSubstitution(r'./HIT-IRLab-Cilin_extended_full_2005.3.3.utf8.txt')
    r = t.do_substitution(index, wordList)
    print(r[1])
    if r[0] == -1:
        return -1
    model.setSentence(r[1])
    print(model.getAnswer())
    return model.getAnswer()[index]
