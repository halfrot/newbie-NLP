file_origin = open("people_daily.txt", "r", encoding="UTF-8")
file_train_set = open("train_set.txt", "w", encoding="UTF-8")
file_validation_set = open("validation_set.txt", "w", encoding="UTF-8")
file_test_set = open("test_set.txt", "w", encoding="UTF-8")


# file_temp = open("debug.txt", "w", encoding="UTF-8")


def SetVector(file_name, s):
    words = s.split()[1:]
    inbracket = False
    vectordim = []
    # cnt = 0
    for word in words:
        # cnt += 1
        # if cnt > 1:
        #     break
        vectordata = 0
        if not inbracket:
            if word.find('[') >= 0:
                inbracket = True
                vectordim.append(word.split('/', 1)[0][1:])
            else:
                if word.split('/', 1)[1] == 'nt':
                    vectordata = 1
                vectordim.append(word.split('/', 1)[0])
        else:
            if word.find(']') >= 0:
                inbracket = False
                if word.split(']', 1)[1] == 'nt':
                    vectordata = 1
            vectordim.append(word.split('/', 1)[0])
        if not inbracket:
            if len(vectordim) == 1:
                file_name.write(vectordim[0] + ' ' + str(vectordata) + '\n')
            elif vectordata == 0:
                for element in vectordim:
                    file_name.write(element + ' 0\n')
            else:
                file_name.write(vectordim[0] + ' 1\n')
                for i in range(1, len(vectordim)):
                    file_name.write(vectordim[i] + ' 2\n')
            vectordim.clear()


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
