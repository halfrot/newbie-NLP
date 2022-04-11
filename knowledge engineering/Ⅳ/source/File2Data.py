import random
import re


# get the train_set and valid_set
def GetDataset(file):
    f = open(file, "r", encoding="UTF-8")
    t = open("Mytrain.txt", "w", encoding="UTF-8")
    v = open("Myvalid.txt", "w", encoding="UTF-8")
    for s in f:
        s = ClearString(s) + '\n'
        p = random.random()
        if p > 0.1:
            t.write(s)
        else:
            v.write(s)
    f.close()
    t.close()
    v.close()


# get y
def GetAnswer(file):
    f = open(file, "r", encoding="UTF-8")
    a = open("Answer.txt", "w", encoding="UTF-8")
    for s in f:
        a.write(s)
    f.close()
    a.close()


# Clean the String
def ClearString(text):
    text = text.lower()
    text = text.replace('<e1>', ' _e11_')
    text = text.replace('</e1>', ' _e12_')
    text = text.replace('<e2>', ' _e21_')
    text = text.replace('</e2>', ' _e22_')
    text = text.replace(';', ' ;')
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\s0", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# when running the model for the first time, run the function blow

# GetDataset("train.txt")
# GetAnswer("train_result.txt")
