from typing import Dict
import myUi  # 引入designer生成的py文件
import interface  # 引入Attention Bi-LSTM 模型
import sys
import Lesk_bug  # 引入Lesk模型
import Lesk_main  # 引入Lesk模型
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton
import jieba


class MyMainWindow(myUi.Ui_MainWindow):
    def __init__(self, MainWindow):
        super().setupUi(MainWindow)
        self.split.clicked.connect(self.sentenceSplit)  # 给split按钮绑定sentenceSplit按钮，点击可以进行分词
        self.WSD.clicked.connect(self.WordSentenceDisambiguation)  # 给WSD按钮绑定WordSentenceDisambiguation，点击进行ABD消歧
        self.Lesk.clicked.connect(self.WSD_Lesk)  # 给Lesk按钮绑定WSD_Lesk函数，点击进行Lesk消歧
        # 将15个按钮存入列表中
        self.ButtonList = [self.pushButton, self.pushButton_2, self.pushButton_3, self.pushButton_4, self.pushButton_5,
                           self.pushButton_6, self.pushButton_7, self.pushButton_8, self.pushButton_9,
                           self.pushButton_10,
                           self.pushButton_11, self.pushButton_12, self.pushButton_13, self.pushButton_14,
                           self.pushButton_15]
        # 变量初始化
        self.wordList = []
        self.button2id = {}
        for i in range(15):
            self.ButtonList[i].setFlat(True)
            self.button2id[self.ButtonList[i]] = i
        self.label_2.setText('')
        self.label_3.setText('')
        self.LastNum = 0  # 上一个选中词的标号
        self.LastButton = 0  # 上一个选中的按钮
        self.WSD.setFlat(True)
        self.WSD.setText('')
        self.Lesk.setFlat(True)
        self.Lesk.setText('')
        self.MyModel = interface.MyWSD()

    def sentenceSplit(self):
        # ---------每一次分词对界面进行初始化--------
        # item_list = list(range(self.words.count()))
        for i in range(15):
            self.ButtonList[i].setFlat(True)
            self.ButtonList[i].setText('')
        self.label_2.setText('')
        self.label_3.setText('')
        self.wordSelected.setText('')
        for i in range(self.LastNum):
            self.ButtonList[i].clicked.disconnect()
        if self.LastButton != 0:
            op = QtWidgets.QGraphicsOpacityEffect()
            op.setOpacity(1)
            self.LastButton.setGraphicsEffect(op)
            self.LastButton.setAutoFillBackground(True)
        self.LastNum = 0
        self.LastButton = 0
        self.WSD.setFlat(True)
        self.WSD.setText('')
        self.Lesk.setFlat(True)
        self.Lesk.setText('')
        self.word_WSD.setText('')
        self.mean_WSD.setText('')
        # --------------初始化结束-----------------
        if str(self.sentence.text()) == '':  # 如果没有输入句子进行报错
            reply = QMessageBox.information(QMainWindow(), '注意', '请输入有效的句子', QMessageBox.Yes)
        else:  # 输入了就进行切分
            self.label_3.setText('您选择了:')
            self.label_2.setText('请选择您觉得有歧义的词：')
            self.WSD.setFlat(False)
            self.WSD.setText('ABL消歧')
            self.Lesk.setFlat(False)
            self.Lesk.setText('Lesk消歧')
            # 将界面更新
            List = list(jieba.cut(self.sentence.text()))
            self.wordList = List
            self.LastButton = 0
            for i in range(len(List)):
                self.ButtonList[i].setFlat(False)
                self.ButtonList[i].setText(List[i])
                self.ButtonList[i].clicked.connect(self.wordSelect)  # 将按钮绑定选中模块
            self.LastNum = len(List)

    def wordSelect(self):
        op = QtWidgets.QGraphicsOpacityEffect()  # 设置透明度
        if self.LastButton != 0:  # 如果之前选过一个按钮了，将他透明度复原
            op.setOpacity(1)
            self.LastButton.setGraphicsEffect(op)
            self.LastButton.setAutoFillBackground(True)
        sender = QMainWindow().sender()
        op.setOpacity(0.5)
        sender.setGraphicsEffect(op)
        sender.setAutoFillBackground(True)  # 将选中的按钮透明度改变
        self.LastButton = sender
        self.wordSelected.setText(sender.text())  # 同时显示选择的词

    def WordSentenceDisambiguation(self):  # Attention Bi-LSTM模型
        if self.LastButton == 0:  # 如果没有选择歧义词
            reply = QMessageBox.information(QMainWindow(), '注意', '请选择一个歧义词', QMessageBox.Yes)
        # print(self.button2id[self.LastButton])
        meaning = interface.WSD(self.wordList, self.button2id[self.LastButton], self.MyModel)
        if (meaning == -1):  # 如果词库中没有找到歧义词，返回错误
            reply = QMessageBox.information(QMainWindow(), '抱歉', '词库中并无该歧义词', QMessageBox.Yes)
        else:  # 成功获取答案后，将答案显示
            self.word_WSD.setText(self.wordSelected.text() + "的意思为:")
            self.mean_WSD.setText(str(meaning))

    def WSD_Lesk(self):
        if self.LastButton == 0:
            reply = QMessageBox.information(QMainWindow(), '注意', '请选择一个歧义词', QMessageBox.Yes)
            return
        #        print(self.wordSelected.text())
        Dict = Lesk_bug.WSD_word(self.wordSelected.text())  # 从百度百科中获取词的意思
        meaning = Lesk_main.Work(self.sentence.text(), self.wordSelected.text(), Dict)  # 获取词的含义
        self.word_WSD.setText(self.wordSelected.text() + "的意思为:")
        self.mean_WSD.setText(meaning)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = MyMainWindow(mainWindow)
    mainWindow.show()

    sys.exit(app.exec_())

# 诸葛亮卒于2000年前
