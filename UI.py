
import re
from PyQt5.QtWidgets import (QMainWindow,QMessageBox,  QFileDialog, QTextEdit)
from PyQt5.QtGui import  QStandardItemModel, QStandardItem
import os
from langchain_community.document_loaders import PyPDFLoader
import sys
from datetime import datetime
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QMessageBox
from ui_aipilot import Ui_AIPILOT
import sys
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QComboBox, QVBoxLayout, QDialogButtonBox

import jieba.posseg as pseg
import pandas as pd
import logging
from openpyxl import Workbook
logging.getLogger().setLevel(logging.WARNING)# 设置日志级别为 WARNING
# 设置jieba库的日志级别为 WARNING
import jieba
jieba.setLogLevel(logging.WARNING)
import subprocess
import matplotlib.pyplot as plt
from langchain_community.llms import OpenAI
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #正常显示负号
import extractGraph5
import combine_like_items2
import toNeo4j


class WorkerThread(QThread):
    # 通过信号进行线程间通信
    finished = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
    def run(self):
        # 调用耗时函数
        self.main_window.EXTRACT_fun()
        # 发射信号表示操作完成
        self.finished.emit()

class WorkerThread2(QThread):
    # 通过信号进行线程间通信
    finished = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
    def run(self):
        # 调用耗时函数
        self.main_window.combine_fun()
        # 发射信号表示操作完成
        self.finished.emit()


class EncodingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择编码类型")
        layout = QVBoxLayout()
        label = QLabel("请选择编码类型:")
        layout.addWidget(label)
        combobox = QComboBox()
        combobox.addItem("utf-8")
        combobox.addItem("gbk")
        layout.addWidget(combobox)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.setLayout(layout)


    def get_encoding(self):
        if self.exec_() == QDialog.Accepted:
            combobox = self.findChild(QComboBox)
            return combobox.currentText()
        else:
            return None
class OutputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输出对话框")
        self.output_text_edit = QTextEdit(self)
        self.output_text_edit.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(self.output_text_edit)
        self.setLayout(layout)
        self.resize(700,700)
    def print_text(self, text):
        self.output_text_edit.append(text)

class QmyUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)  # 调用父类构造函数
        self.ui = Ui_AIPILOT()  # 创建UI对象
        self.ui.setupUi(self)  # 构造UI界面
        self.setWindowTitle("AIPILOT")
        self.showMaximized()
        self.ui.te_remind.verticalScrollBar().setValue(self.ui.te_remind.verticalScrollBar().maximum())
        self.ui.pbt_importdata.clicked.connect(self.openfile_fun)
        self.ui.pbt_cleandata.clicked.connect(self.washData_fun)
        self.ui.pbt_tagStatistics.clicked.connect(self.tagStatistics_fun)
        self.myOutputDialog = OutputDialog()
        self.ui.pbt_drawcixing.clicked.connect(self.drawcixing_fun)
        self.ui.rb_remind.setChecked(True)
        self.ui.rb_remind.toggled.connect(self.print_change)
        self.ui.pbt_clear_remind.clicked.connect(self.ui.te_remind.clear)
        self.ui.pbt_extract.clicked.connect(self.run_long_operation)
        self.ui.pbt_combine.clicked.connect(self.run_long_operation2)
        self.ui.pbt_run_ollama.clicked.connect(self.test)
        self.ui.pbt_savetoexcel.clicked.connect(self.save2path)
        self.ui.pbt_toNeoj.clicked.connect(self.ToNeoj)
        self.ui.pbt_openPrompt.clicked.connect(self.open_prompt_fun)
        self.ui.pbt_save_prompt.clicked.connect(self.save_prompt_fun)
        self.ui.pbt_openModelList.clicked.connect(self.open_modelList_fun)
        self.ui.le_result_extractPath.setPlaceholderText('知识提取结果路径，用于相似词合并')
        self.ui.le_result_scorePath.setPlaceholderText('打分结果路径')
        self.ui.le_neo4j_excelPath.setPlaceholderText('相似词合并结果路径，同时用于neo4j可视化')

        self.ui.pbt_save_clean.clicked.connect(self.save_clean_fun)
        # self.ui.cobo_ciChoose.addItems(['n','nz','a','m','c','PER','f','v','ad','q','u','LOC','s','vd','an','r','xc','ORG','nw','vn','d','p','w','TIME'])

        self.ui.cobo_ciChoose.addItems(['v','q','f','n','nz','x','d','vn','m','c','eng','I','a','r','p'])   #需要修改

        self.ui.tab_ollama.setVisible(False)  #设置ollama界面不可见
        self.ui.tab_ollama.setEnabled(False)
        # 保存默认的sys.stdout
        self.default_stdout = sys.stdout
        self.redirect_output = True
        # 重定向sys.stdout到自定义的输出函数
        sys.stdout = self
        #定义常量
        self.defultFolderPath=os.getcwd()

    def save_clean_fun(self):
        text = self.ui.te_after.toPlainText()
        if not text:
            QMessageBox.warning(self, 'Warning', 'No text to save!')
            return
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;All Files (*)",
                                                  options=options)
        if filePath:
            try:
                # 将文本写入txt文件
                with open(filePath, 'w') as file:
                    file.write(text)
                QMessageBox.information(self, 'Success', 'Text saved to TXT successfully!')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error occurred while saving: {str(e)}')
    def run_long_operation(self):
        # 创建并启动工作线程
        self.worker_thread = WorkerThread(self)
        self.worker_thread.finished.connect(self.on_thread_finished)
        self.worker_thread.start()
        # 禁用按钮防止重复点击
        self.ui.pbt_extract.setEnabled(False)
    def on_thread_finished(self):
        # 操作完成后恢复按钮状态
        self.ui.pbt_extract.setEnabled(True)
        QMessageBox.information(self, "提示-线程", "知识提取已完成")


    def run_long_operation2(self):
        # 创建并启动工作线程
        self.worker_thread2 = WorkerThread2(self)
        self.worker_thread2.finished.connect(self.on_thread_finished2)
        self.worker_thread2.start()
        # 禁用按钮防止重复点击
        self.ui.pbt_combine.setEnabled(False)
    def on_thread_finished2(self):
        # 操作完成后恢复按钮状态
        self.ui.pbt_combine.setEnabled(True)
        QMessageBox.information(self, "提示-线程", "相似词合并已完成")



    def combine_fun(self):
        # source_file=self.knowledge_path
        source_file=self.ui.le_result_extractPath.text()
        if len(source_file)<3:
            print('知识路径为空')
            return
        try:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
            filename = os.path.join(self.defultFolderPath, formatted_datetime + "-相似词合并结果.xlsx")
            save_file = filename
            self.ui.le_neo4j_excelPath.setText(save_file)
            # promptStatement_replaceTwoSentenceWithOne = "Generate a very short sentence to replace two short sentences with the same meaning"
            prompt=self.prompt_dict['promptStatement_replaceTwoSentenceWithOne']
            score_usedtoJudegeSimilarity = float(self.ui.le_similarity.text())
            combine_like_items2.combine(source_file, save_file, self.model_gpt4, score_usedtoJudegeSimilarity,prompt)
            print('相似词合并完成')
        except:
            print('相似词合并失败')
    def open_modelList_fun(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*);;Excel Files (*.xls)")
        self.ui.le_modelListPath.setText(fileName1)
        df = pd.read_excel(fileName1)
        self.score_columnName = ['text', 'dict'] + df['modelName'].values.tolist() + ['average']

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(df.columns)
        for row in range(df.shape[0]):
            items = [QStandardItem(str(df.iloc[row, col])) for col in range(df.shape[1])]
            model.appendRow(items)
        # 设置第一列不可编辑
        for row in range(model.rowCount()):
            item = model.item(row, 0)
            item.setEditable(False)
        self.model_prompt = model
        self.ui.tb_modelList.setModel(model)

    def open_prompt_fun(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*);;Excel Files (*.xls)")
        self.ui.le_promptPath.setText(fileName1)
        df=pd.read_excel(fileName1)
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(df.columns)
        for row in range(df.shape[0]):
            items = [QStandardItem(str(df.iloc[row, col])) for col in range(df.shape[1])]
            model.appendRow(items)
        # 设置第一列不可编辑
        for row in range(model.rowCount()):
            item = model.item(row, 0)
            item.setEditable(False)
        print('第一列不可修改，其余双击可编辑，保存后更新原模板文件。')
        self.model_prompt=model
        self.ui.tb_prompt.setModel(model)
    def save_prompt_fun(self):
        num_cols = self.model_prompt.columnCount()
        column_names = [self.model_prompt.horizontalHeaderItem(col).text() for col in range(num_cols)]
        df = pd.DataFrame(columns=column_names)
        for row in range(self.model_prompt.rowCount()):
            row_data = [self.model_prompt.item(row, col).text() for col in range(num_cols)]
            df.loc[len(df)] = row_data
        df.to_excel(self.ui.le_promptPath.text(), index=False)
        QMessageBox.information(self, '提示', '保存成功')
    def test(self):
        pass

    def save2path(self):
        options = QFileDialog.Options()
        try:
            folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", options=options)
            self.defultFolderPath = folder_path
        except:
            pass
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
        path_temp=os.path.join(self.defultFolderPath,formatted_datetime+"知识.xlsx")
        # QMessageBox.information(self,'提示','知识保存成功')
        self.ui.le_saveExcelPath.setText(str(self.defultFolderPath))

    def write(self, text):
        if self.redirect_output:
            self.ui.te_remind.append(text)
    def print_change(self, checked):
        self.redirect_output = checked
        if checked:
            sys.stdout = self
        else:
            sys.stdout = self.default_stdout

    def drawcixing_fun(self):
        self.ui.widget.figure.clear()
        ax1 = self.ui.widget.figure.add_subplot(111)

        name=self.ui.cobo_ciChoose.currentText()
        print(name)
        v = self.df_cixing[name].values.tolist()
        print(v)
        if len(v)<1:
            QMessageBox.warning(self, "警告", "数据太短，无法绘图", QMessageBox.Cancel)
            return
        words1=v
        words = [x for x in words1 if x == x]
        counts = {}
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        # 按照次数从大到小排序
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        # 提取键和值
        keys = [item[0] for item in sorted_counts]
        values = [item[1] for item in sorted_counts]
        X = [x for x in keys if x != None]
        Y = [values[i] for i, x in enumerate(keys) if x != None]
        # 绘制柱状图
        print(X,Y)
        print(len(X),len(Y))
        ax1.bar(X,Y)
        ax1.set_title('Word Counts')
        ax1.set_xlabel('Words')
        ax1.set_ylabel('Counts')
        # # 自动调整横坐标标签的位置
        # ax1.set_xticks(rotation=45, ha='right')
        # 显示图形
        self.ui.widget.redraw()


    def openfile_fun(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*);;Excel Files (*.xls)")
        # fileName1=r"knowledge/320neo-冷舱程序.txt"
        self.ui.le_showpath.setText(fileName1)
        type=os.path.splitext(fileName1)[-1].lower()
        print('文件名：',fileName1)
        if type==".txt":
            encodingtype = dialog.get_encoding()
            if not encodingtype:
                encodingtype='utf-8'
            with open(fileName1, encoding=encodingtype, errors='ignore') as f:
                data = f.read()
        if type==".pdf":
            data = ''
            loader = PyPDFLoader(fileName1)
            documents = loader.load()
            for docu in documents:
                data = data + docu.page_content
        print(data)
        print('文件读取完成')
        self.data=data
        self.ui.te_before.append(self.data)


    def washData_fun(self):
        self.lines = self.data.splitlines()
        for line in self.lines:
            text=self.clean_text(line)
            self.ui.te_after.append(text)

    def clean_text(self,sentence):  # 数据清洗
        # 过滤HTML标签
        sentence = re.sub(r'<.*?>', '', sentence)
        # 过滤数字
        # sentence = re.sub(r'\d+', '', sentence) #数字不过滤
        # 过滤特殊符号
        # sentence = re.sub(r'[^\w\s]', '', sentence)
        pattern = r"[。]+"
        sentence = re.sub(pattern, lambda match: match.group()[0], sentence)
        # 过滤空格
        # sentence = sentence.replace(' ', '')
        return sentence

    def tagStatistics_fun(self):
        stopword_path = r'knowledge/stopwords-master/baidu_stopwords.txt'
        with open(stopword_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        collect=[]
        if self.ui.cb_show.checkState():
            for line in self.lines:
                try:
                    result,temp=self.statistic_fun(line,stopwords)
                    collect=collect+result
                    self.myOutputDialog.print_text(str(temp))
                    self.myOutputDialog.show()
                except:
                    pass

        word_dict = self.count_cixing(collect)
        df = pd.DataFrame.from_dict(word_dict, orient='index').transpose()
        print(df)
        self.df_cixing=df
        self.show_cixing_fun(df)
    def show_cixing_fun(self,df):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(df.columns)
        for row in range(df.shape[0]):
            items = [QStandardItem(str(df.iloc[row, col])) for col in range(df.shape[1])]
            model.appendRow(items)
        self.ui.tb_cixing.setModel(model)
    def count_cixing(self,collect):
        word_dict = {}
        for word, pos in collect:
            if pos not in word_dict:
                word_dict[pos] = []  # 初始化列表
            word_dict[pos].append(word)
        return word_dict
    def is_english_string(self,text):
        return bool(re.match(r'^[a-zA-Z]+$', text))
    def statistic_fun(self,text1, stopwords):
        sentences = text1.split('，')
        result_all = []
        for text in sentences:
            temp1='{:<10}'.format('原始句子：') + str(text)
            seg_list = list(jieba.cut(text))
            temp2 = '{:<10}'.format('分词：') + "  ".join(seg_list)
            # 停用词去除
            filtered_list = [word for word in seg_list if word not in stopwords]
            temp3 = '{:<10}'.format('去停用词：') + str(filtered_list)
            # 简化
            result_list = []
            for word in filtered_list:
                if self.is_english_string(word):
                    result_list.append([word, 'nz'])
                else:

                    words = pseg.cut(word)
                    for word, flag in words:
                        result_list.append([word,flag])
            temp4 = '{:<10}'.format('赋予词性:') + str(result_list)
            temp=temp1+'\n'+temp2+'\n'+temp3+'\n'+temp4
            result_all = result_all + result_list
        return result_all,temp

# ******************************
# '''
# 知识提取阶段
# '''
    def EXTRACT_fun(self):
        path_modelSetting = self.ui.le_modelListPath.text()
        print(path_modelSetting)
        if len(path_modelSetting)<3:
            QMessageBox.information(self,'提示','请选择模型路径')
            return
        model_list = []
        models_setting = pd.read_excel(path_modelSetting).values
        try:
            model_gpt4 = OpenAI(openai_api_key=models_setting[0][1], base_url=models_setting[0][2],
                                temperature=models_setting[0][3], top_p=models_setting[0][4])
            self.model_gpt4=model_gpt4
            for i in range(1, len(models_setting)):
                model_temp = OpenAI(openai_api_key=models_setting[i][1], base_url=models_setting[i][2],
                                    temperature=models_setting[i][3], top_p=models_setting[i][4])
                model_list.append(model_temp)
        except:
            QMessageBox.information(self,'提示','llm模型初始化错误，请检查网络或llm模型列表')
            return
        ###
        columns = ['ActionReason', 'ActionObject', 'Action', 'ActionResult', 'score', 'text']
        df_action = pd.DataFrame(columns=columns)
        columns = ['condition', 'causedBy', 'score', 'text']
        df_condition = pd.DataFrame(columns=columns)
        columns = ['event', 'subtask', 'score', 'text']
        df_event = pd.DataFrame(columns=columns)
        columns = ['ActionResult', 'feedback', 'score', 'text']
        df_feedback = pd.DataFrame(columns=columns)
        ###
        dict_action, dict_condition, dict_event, dict_feedback = [], [], [], []
        #读取提示词

        excel_file =self.ui.le_promptPath.text()
        if len(excel_file)<3:
            QMessageBox.information(self,'提示','请选择提示词模板路径')
            return
        prompt_dict=extractGraph5.excel_to_dict(excel_file)
        self.prompt_dict=prompt_dict

    ##获得要分析的文本
        temp=self.ui.te_after.toPlainText()
        # replaced_string = temp.replace('\n', '.')
        replaced_string = temp.replace('\n', ' ') #换行视为空格
        split_strings = re.split(r'[.。]', replaced_string)
        TEXT1=split_strings
        print('TEXT=',TEXT1)
        # for i in TEXT:
        #     print(i)
        #     print(len(i))
        # return 0
        text_dict_scoreSave=[]
        ii=0
        length1=len(TEXT1)
        for data in TEXT1:
            ii=ii+1
            if len(data)<6:
                print(data+'文本太短，不进行知识提取')
                continue
            # 第一步，转换为词典
            a_action, a_condition, a_event, a_feedback, TEXT = extractGraph5.text2dict(model_gpt4, data, prompt_dict)
            print("-" * 10)
            print(str(ii)+'/'+str(length1+1)+'-知识提取：',data)
            print(a_action)
            print(a_condition)
            print(a_event)
            print(a_feedback)
            print("-" * 10)
            # 第二步，其他模型对转换过程打分，取平均值最高的那个
            dict_after_score, score_all, text_dict_score_all = extractGraph5.score_range(a_action, a_condition, a_event, a_feedback,
                                                                           data, model_list, prompt_dict)
            text_dict_scoreSave.extend(text_dict_score_all)
            print("打分结果")
            for i, j in score_all.items():
                print(i, j)
            print(dict_after_score)
            print("-" * 10)
            # 第三步，保存
            df_action, df_condition, df_event, df_feedback = extractGraph5.tranfer2df(df_action, df_condition, df_event, df_feedback,
                                                                        dict_after_score, data)
        print("*" * 10 + '显示全部' + "*" * 10)
        print(df_action)
        print(df_event)
        print(df_feedback)
        print(df_condition)
        temp = [{'crew': 'pilot', 'note': None}, {'crew': 'pilot2', 'note': None}, ]  # 列表对应的是第一维，即行，字典为同一行不同列元素
        df_pilot = pd.DataFrame(temp)  # 第 1 行 3 列没有元素，自动添加 NaN (Not a Number)

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
        filename = os.path.join(self.defultFolderPath, formatted_datetime + "-知识提取结果.xlsx")
        self.knowledge_path=filename
        if len(filename)<3:
            QMessageBox.information(self, '提示', '请选择知识保存路径')
            return
        try:
            with pd.ExcelWriter(filename) as writer:
                df_pilot.to_excel(writer, 'crewman')
                df_action.to_excel(writer, 'action')
                df_event.to_excel(writer, 'event')
                df_feedback.to_excel(writer, 'feedback')
                df_condition.to_excel(writer, 'condition')
            # QMessageBox.information(self,'提示-知识保存','知识保存成功')  #不能出界面
            print('知识保存成功'+filename)
        except:
            # QMessageBox.information(self,'提示-知识保存','知识保存失败')
            return
        self.ui.le_result_extractPath.setText(filename)
        df_scoreResult = pd.DataFrame(text_dict_scoreSave,columns=self.score_columnName)
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
        filename = os.path.join(self.defultFolderPath, formatted_datetime + "-过程打分结果.xlsx")
        df_scoreResult.to_excel(filename)
        print('知识打分结果保存成功'+filename)
        self.ui.le_result_scorePath.setText(filename)
        # QMessageBox.information(self, '提示-打分', '知识打分结果保存成功')
# ******************************
# '''
# 可视化阶段
# '''
    def ToNeoj(self):
        self.ui.le_getNeojName.setText('neo4j')
        self.ui.le_getNeojPw.setText('123456')
        try:
            excel_file=self.ui.le_neo4j_excelPath.text()
        # excel_file = r"E:\BaiduSyncdisk\AIAgentForPilots\AIcode\knowledge\demo-后推 - 副本.xlsx"
            data = pd.read_excel(excel_file, sheet_name=None)
        except:
            QMessageBox.critical(self, "错误", "文件读取错误")
            return
        toNeo4j.connectNeo4j_fun()
        try:
            usr=self.ui.le_getNeojName.text()
            key=self.ui.le_getNeojPw.text()
            graph, matcher = toNeo4j.connect_fun(usr,key)
        except:
            QMessageBox.critical(self, "错误", "neo4j连接错误")
            return
        toNeo4j.delAll_fun(graph)
        df_action = data['action']
        df_feedback = data['feedback']
        df_event = data['event']
        df_crewman = data['crewman']
        df_condition = data['condition']
        # 存储已定义的节点名称
        defined_nodes = set()
        # 定义crewman
        defined_nodes = toNeo4j.creat_crewman(df_crewman, defined_nodes, graph, matcher)
        # 定义feedback
        defined_nodes = toNeo4j.creat_feedback(df_feedback, defined_nodes, graph, matcher)
        # 定义action
        defined_nodes = toNeo4j.creat_action(df_action, defined_nodes, graph, matcher)
        # 定义event
        defined_nodes = toNeo4j.creat_event(df_event, defined_nodes, graph, matcher)
        # 定义condition
        defined_nodes = toNeo4j.creat_condition(df_condition, defined_nodes, graph, matcher)
        print('创建成功')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = EncodingDialog()
    form = QmyUI()
    form.show()
    sys.exit(app.exec_())


