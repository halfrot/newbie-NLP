from typing import Dict
import requests
import re
from bs4 import BeautifulSoup


class WebScrape(object):
    def __init__(self, word, url):
        self.url = url
        self.word = word

    # 爬取百度百科页面
    def web_parse(self):
        # HTTP请求头
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 \
                                             (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36'}
        # 获取网页，对应HTTP中的GET方法
        req = requests.get(url=self.url, headers=headers)

        # 解析网页，定位到main-content部分
        if req.status_code == 200:
            soup = BeautifulSoup(req.text.encode(req.encoding), 'lxml')
            return soup
        return None

    # 获取该词语的义项和url
    def get_gloss(self):
        soup = self.web_parse()
        if soup:
            gloss = []  # 存储词的义项
            url = []  # 存储每个义项的网址

            # BeautifulSoup.find()在树形结构中查找第一个标签入口
            # 百度百科网页有两类组织方式，多义词义项分别放在类
            # ”polysemantList-wrapper cmn-clearfix“、”custom_dot para-list list-paddingleft-1“
            # 分别定位两个类名，若均未定位到则为非歧义词
            lis = soup.find('ul', class_="polysemantList-wrapper cmn-clearfix")
            if lis == None:
                lis = soup.find('ul', class_="custom_dot para-list list-paddingleft-1")

            if lis:
                # 获取该词语的义项和url
                for li in lis('li'):
                    print(str(li))
                    gloss.append(li.text.replace('▪', ''))
                    if '<a' in str(li):
                        l = str(li).find("item/") + 5
                        # ”polysemantList-wrapper cmn-clearfix“的网址右边界
                        r = str(li).find("#viewPageContent") + 16
                        if r == 15:
                            # ”custom_dot para-list list-paddingleft-1“的网址右边界
                            r = str(li).find("\" target=\"_blank\"")
                        # 拼接义项的网页网址
                        url.append(self.url[: -len(self.word)] + str(li)[l: r])
                    else:
                        url.append(self.url)
                # 该词为歧义词，返回所有义项和每个义项的网页
                return gloss, url
            else:
                # 该词为非歧义词，返回当前词义和网页
                return [self.word], [self.url]

    # 获取该义项的语料，以句子为单位
    def get_content(self):
        # 发送HTTP请求
        result = []
        soup = self.web_parse()
        if soup:
            # 定位到“main-content”部分，提取全部文本内容
            paras = soup.find('div', class_='main-content').text.split('\n')
            for para in paras:
                if self.word in para:
                    # 对文本按句子分割，将包含歧义词的句子作为该义项的语料
                    sents = re.split("[。？！]+", para)
                    for sent in sents:
                        if self.word in sent:
                            sent = sent.replace('\xa0', '').replace('\u3000', '')
                            result.append(sent)

        result = list(set(result))

        return result

    # 将该义项的语料存入Dict字典，字典结构为：Dict[歧义词义项]=[语料]
    def get_dict(self):
        gloss, url = self.get_gloss()
        Dict = {}
        for i in range(len(gloss)):
            self.url = url[i]
            result = self.get_content()
            if result and gloss[i]:
                Dict[gloss[i]] = result
        return Dict

    def run(self):
        return self.get_dict()


def WSD_word(keyword):
    url = 'https://baike.baidu.com/item/' + keyword
    return WebScrape(keyword, url).run()
