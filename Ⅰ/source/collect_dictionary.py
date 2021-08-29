from collections import Counter

DictionarySize = 5000

file_origin = open("people_daily.txt", "r", encoding="UTF-8")
file_dictionary = open("dictionary.txt", "w", encoding="UTF-8")

count = Counter()
for s in file_origin:
    words = s.split()[1:]
    for word in words:
        vectordim = list(word.split('/', 1)[0])
        if '[' in vectordim:
            vectordim.remove('[')
        vectordim = ''.join(vectordim)
        if word.split('/', 1)[1] == 'nt':
            # print(vectordim)
            if vectordim in count:
                count[vectordim] += 1
            else:
                count[vectordim] = 2
        else:
            count[vectordim] = 1

temp_count = 0
for (vectordim, cnt) in count.most_common(DictionarySize):
    file_dictionary.write(vectordim + '\n')
    temp_count += 1
print(temp_count)
file_origin.close()
file_dictionary.close()
