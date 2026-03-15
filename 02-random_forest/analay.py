# 数据分析
import pandas as pd
import numpy as np
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from icecream import ic
# 读取数据
content = pd.read_csv('./data/data/train.txt', sep='\t', names=['sentence', 'label'])
print(content.head())

# 统计类别数量
count = Counter(content.label.values)
print(count)
print(len(count))

# 统计样本总数
total = 0
for i, v in count.items():
    total += v
print(total)

# 计算样本比重
for i, v in count.items():
    print(i,':',v/total*100,'%')

# 计算方差标准差 分析文本长度
content['sentence_len'] = content['sentence'].apply(len)
content_mean = np.mean(content['sentence_len'])
content_std = np.std(content['sentence_len'])
print(content_mean,content_std)

# 结巴分词
def cut_sentenc(s):
    return  jieba.lcut(s)
# 3.2 空格拼接文本,截断处理
content['words'] = content['sentence'].apply(lambda x:cut_sentenc(x))
print(f'分词后结果:{content.head()}')
content['words'] = content['words'].apply(lambda x:' '.join(x)[:30])
print(f'空格拼接后:{content.head()}')
# 3.3 保存到csv
content.to_csv('./data/data/train_new.csv')

# 随机森林模型
# 读取数据
TRAIN_CORPUS = './data/data/train_new.csv'
STOP_WORDS = './data/data/stopwords.txt'
WORDS_COLUMN = 'words'

# 读取数据集
content = pd.read_csv(TRAIN_CORPUS)
# 构建预料库
corpus = content[WORDS_COLUMN].values