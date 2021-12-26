import os
import jieba
from math import log2


class lesk(object):
    def __init__(self, word, sent):
        self.stopwords = [word, '我', '你', '它', '他', '她', '了', '是', '的', '啊', '谁', '什么', '都',
                          '很', '个', '之', '人', '在', '上', '下', '左', '右', '。', '，', '！', '？']  # 停用词
        self.word = word
        self.sent = sent
        self.total_document = 0  # 整个语料库中句子条数
        self.sent_cut = []  # 待消歧句子预处理的结果
        self.wsd_dict = {}  # wsd_dict[歧义词词义] = [语料]
        self.tf_dict = {}  # tf_dict[歧义词词义] = [(词，词频)，……]
        self.idf_dict = {}  # idf_dict[词] = 所有语料中有该词语的句子个数
        self.mean_tf_idf = []  # 各个词义下tf-idf值

    # jieba分词，并去除结束词，结果存在sent_cut中
    def fen_ci(self):
        jieba.add_word(self.word)
        sent_to_words = list(jieba.cut(self.sent, cut_all=False))
        for word in sent_to_words:
            if word not in self.stopwords:
                self.sent_cut.append(word)

    # 计算sent_cut中词的tf值
    def cal_tf(self):
        for meaning, sents in self.wsd_dict.items():
            self.tf_dict[meaning] = []
            for word in self.sent_cut:
                word_count = 0
                for sent in sents:
                    example = list(jieba.cut(sent, cut_all=False))
                    word_count += example.count(word)

                if word_count:
                    self.tf_dict[meaning].append((word, word_count))

    # 计算sent_cut中词的idf值
    def cal_idf(self):
        for word in self.sent_cut:
            document_count = 0
            for meaning, sents in self.wsd_dict.items():
                for sent in sents:
                    if word in sent:
                        document_count += 1
            self.idf_dict[word] = document_count

    # 统计语料库句子总条数
    def cal_sent_num(self):
        for meaning, sents in self.wsd_dict.items():
            self.total_document += len(sents)

    # 计算每个义项的tf-idf值
    def cal_tf_idf(self):
        for k, v in self.tf_dict.items():
            tf_idf_sum = 0
            for item in v:
                word = item[0]
                tf_idf = item[1] * log2(self.total_document / (1 + self.idf_dict[word]))
                tf_idf_sum += tf_idf
            self.mean_tf_idf.append((k, tf_idf_sum))

    # 对每个义项的tf-idf值排序，选取最大值对应的义项作为该歧义词的词义
    def show_result(self):
        sort_array = sorted(self.mean_tf_idf, key=lambda x: x[1], reverse=True)
        print(sort_array)
        true_meaning = sort_array[0][0]
        print('\n经过词义消岐，%s在该句子中的意思为 %s .' % (self.word, true_meaning))
        return true_meaning

    def run(self, Dict):
        self.wsd_dict = Dict
        self.fen_ci()
        self.cal_tf()
        self.cal_idf()
        self.cal_sent_num()
        self.cal_tf_idf()
        return self.show_result()


def Work(sent, word, Dict):
    return lesk(word, sent).run(Dict)
