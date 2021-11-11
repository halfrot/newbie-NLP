"""
function getFrequency
用于提取文本中的词，填充词典
返回值 字典类型cnt,mat,type_cnt
cnt中key为词，value为字典类型（key为词性，value为频度），存有词语/词性频度
mat中key为词性，value为字典类型（key为词性，value为频度），存有词性转移频度
type_cnt中key为词性，value为词性频率
"""


def getFrequency():
    f = open("1998-01-2003版-带音.txt", "r", encoding="UTF-8")
    pre = "NULL"
    cnt = {}
    mat = {}
    type_cnt = {}
    for s in f:
        words = s.split()
        for word in words:
            if '[' in word:
                word = word.split('[')[1]
            if ']' in word:
                word = word.split(']')[0]
            word = word.split('/')
            # 词语/词性频度表
            if word[0] not in cnt:
                cnt[word[0]] = dict()
                cnt[word[0]][word[1]] = 1
            elif word[1] not in cnt[word[0]]:
                cnt[word[0]][word[1]] = 1
            else:
                cnt[word[0]][word[1]] += 1
            # 词性转移矩阵
            if pre == "NULL":
                pre = word[1]
                continue
            if pre not in mat:
                mat[pre] = dict()
                mat[pre][word[1]] = 1
            elif word[1] not in mat[pre]:
                mat[pre][word[1]] = 1
            else:
                mat[pre][word[1]] += 1
            pre = word[1]
            # 词性频度表
            if word[1] not in type_cnt:
                type_cnt[word[1]] = 1
            else:
                type_cnt[word[1]] += 1
    return cnt, mat, type_cnt


"""
function inputWash
清洗input文本，整理格式，去掉空行和复合词
"""


def inputWash(filepath):
    fin = open(filepath, "r", encoding="UTF-8")
    fout = open("WashedInput", "w", encoding="UTF-8")
    for s in fin:
        words = s.split()
        if len(words) == 0:
            continue
        for word in words:
            if "[" in word:
                word = word.split("[")[1]
            if "]" in word:
                word = word.split("]")[0]
            word = word.split("/")[0]
            fout.write(word + " ")
        fout.write("\n")
    fin.close()
    fout.close()


"""
function answerWash
清洗answer文本，整理格式，去掉空行和复合词
"""


def answerWash(filepath):
    fin = open(filepath, "r", encoding="UTF-8")
    fout = open("WashedAnswer", "w", encoding="UTF-8")
    for s in fin:
        words = s.split()
        if len(words) == 0:
            continue
        for word in words:
            if "[" in word:
                word = word.split("[")[1]
            if "]" in word:
                word = word.split("]")[0]
            fout.write(word + " ")
        fout.write("\n")
    fin.close()
    fout.close()


"""
测试
"""
if __name__ == "__main__":
    cnt, mat, type_cnt = getFrequency()
    out = open("tmp.txt", "w", encoding="UTF-8")
    for ele in cnt:
        out.write(ele + " " + str(cnt[ele]) + "\n")
    out.close()

    answerWash("1998-01-2003版-带音.txt")

    inputWash("1998-01-2003版-带音.txt")
