file_origin = open("people_daily.txt", "r", encoding="UTF-8")
file_train_set = open("train_set.txt", "w", encoding="UTF-8")
file_validation_set = open("validation_set.txt", "w", encoding="UTF-8")
file_test_set = open("test_set.txt", "w", encoding="UTF-8")
# file_temp = open("debug.txt", "w", encoding="UTF-8")

"""
ner_collect_set.py
独立于其他代码
用于切分、清洗文本，整理格式，获得可供使用的语料
初始文本people_daily.txt
切分得到训练集train_set.txt，验证集validation_set.txt，测试集test_set.txt
"""


def SetVector(file_name, s):
    words = s.split()[1:]
    inbracket = False
    vector_dim = []
    for word in words:
        vector_data = 0
        if not inbracket:
            if word.find('[') >= 0:
                inbracket = True
                vector_dim.append(word.split('/', 1)[0][1:])
            else:
                if word.split('/', 1)[1] == 'nt':
                    vector_data = 1
                vector_dim.append(word.split('/', 1)[0])
        else:
            if word.find(']') >= 0:
                inbracket = False
                if word.split(']', 1)[1] == 'nt':
                    vector_data = 1
            vector_dim.append(word.split('/', 1)[0])
        if not inbracket:
            if len(vector_dim) == 1:
                file_name.write(vector_dim[0] + ' ' + str(vector_data) + '\n')
            elif vector_data == 0:
                for element in vector_dim:
                    file_name.write(element + ' 0\n')
            else:
                file_name.write(vector_dim[0] + ' 1\n')
                for i in range(1, len(vector_dim)):
                    file_name.write(vector_dim[i] + ' 2\n')
            vector_dim.clear()


for s in file_origin:
    Time = s[0:8]
    if Time <= '19980120':
        SetVector(file_train_set, s)
    elif Time <= '19980126':
        SetVector(file_validation_set, s)
    else:
        SetVector(file_test_set, s)
file_origin.close()
file_train_set.close()
file_validation_set.close()
file_test_set.close()
