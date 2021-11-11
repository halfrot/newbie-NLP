"""
function getDict
用于提取文本中的词，填充词典
返回值 字典类型dict
dict中key为词，value为词频

将dict按value排序后保存在文件"myDict"中
格式为"word word_frequency"
"""


def getDict(filepath):
    dict = {}
    f = open(filepath, "r", encoding="UTF-8")
    for s in f:
        words = s.split()
        for word in words:
            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1
    f.close()
    tofile = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    out = open("myDict", "w", encoding="UTF-8")
    for ele in tofile:
        out.write(str(ele[0]) + " " + str(ele[1]) + '\n')
    out.close()
    return dict


"""
测试
"""

if __name__ == "__main__":
    Dict = getDict("dictionary.utf8")
    print(Dict)
