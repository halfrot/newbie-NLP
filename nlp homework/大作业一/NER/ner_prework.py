"""
function read_vector
读入预训练word embedding矩阵
dict_list存放编号到词向量的映射
word_to_idx存放词到编号的映射
"""


def read_vector(file_path):
    dict_list = []
    word_to_idx = {}
    f = open(file_path, "r", encoding="UTF-8")
    cnt = 0
    for s in f:
        vec = s.split()
        word_to_idx[vec[0]] = cnt
        dict_list.append([])
        for i in range(1, len(vec)):
            dict_list[cnt].append(float(vec[i]))
        cnt += 1
    # dict_list.append([])
    # for i in range(0, len(dict_list[0])):
    #     dict_list[cnt].append(float(0))
    f.close()
    return dict_list, word_to_idx
