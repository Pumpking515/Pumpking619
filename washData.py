import re
import jieba
import jieba.posseg as pseg
from LAC import LAC
import pandas as pd
import logging
logging.getLogger().setLevel(logging.WARNING)# 设置日志级别为 WARNING
# 设置jieba库的日志级别为 WARNING
jieba.setLogLevel(logging.WARNING)
import matplotlib.pyplot as plt
def clean_text(sentence):  #数据清洗
    # 过滤HTML标签
    sentence = re.sub(r'<.*?>', '', sentence)
    # 过滤数字
    sentence = re.sub(r'\d+', '', sentence)
    # 过滤特殊符号
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # 过滤空格
    sentence = sentence.replace(' ', '')
    return sentence
def is_english_string(text):
    return bool(re.match(r'^[a-zA-Z]+$', text))
def statistic_fun(text1,stopwords):
    sentences = text1.split('，')
    result_all=[]
    for text in sentences:
        print('{:<10}'.format('原始句子：'),text)
        # 清洗
        clean_text1 = clean_text(text)
        print('{:<10}'.format('清洗：'),clean_text1)
        # 分词
        seg_list = list(jieba.cut(clean_text1))
        print('{:<10}'.format('分词：'),"  ".join(seg_list))
        # 停用词去除
        filtered_list = [word for word in seg_list if word not in stopwords]
        print('{:<10}'.format('去停用词：'),filtered_list)
        #简化
        result_list=[]
        for word in filtered_list:
            if is_english_string(word):
                result_list.append([word,'nz'])
            else:
                lac = LAC(mode='lac')
                temp=lac.run(word)
                result_list.append([temp[0][0],temp[1][0]])
        # print('filtered_list:', filtered_list)
        print('{:<10}'.format('赋予词性:'), result_list)
        print('*'*10)
        result_all=result_all+result_list
    print(result_all)
    return result_all
def count_cixing(collect):
    word_dict = {}
    for word, pos in collect:
        if pos not in word_dict:
            word_dict[pos] = []  # 初始化列表
        word_dict[pos].append(word)
    return word_dict


if __name__=="__main__":
    # 加载停用词表
    stopword_path=r'knowledge/stopwords-master/baidu_stopwords.txt'
    with open(stopword_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    # text1='下面我们需要启动APU一辅助动力系统，他将为我们将来启动引擎提供动力，并在启动引擎之前提供电力和空调系统,依此点击黄框内的MASTER SWITCH和START'
    # statistic_fun(text1,stopwords)
    file_path = r'knowledge/320neo-冷舱程序.txt'  # 替换为你的文本文件路径
    lines = []  # 存储每行内容的列表
    # 逐行读取文本文件
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除行尾的换行符和空白字符
            lines.append(line)
    # 打印读取的每行内容
    collect=[]
    for line in lines:
        temp=statistic_fun(line,stopwords)
        collect=collect+temp

    word_dict =count_cixing(collect)
    print(word_dict)

    df = pd.DataFrame.from_dict(word_dict, orient='index').transpose()
    print(df)
    # df.to_excel('1.xlsx')















