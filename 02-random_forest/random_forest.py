import pandas as pd
import numpy as np
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from icecream import ic



# 随机森林模型
# 读取数据
TRAIN_CORPUS = './data/data/train_new.csv'
STOP_WORDS = './data/data/stopwords.txt'
WORDS_COLUMN = 'words'

# 读取数据集
content = pd.read_csv(TRAIN_CORPUS)
# 构建预料库
corpus = content[WORDS_COLUMN].values
# print(corpus)
# 读取停用词
stop_words = open(STOP_WORDS, encoding='utf-8').read().split()
# print(stop_words)

# 计算TF-IDF特征
tfidf = TfidfVectorizer(stop_words=stop_words)
text_vectors = tfidf.fit_transform(corpus)
print(text_vectors)

# 获取标签
label = content['label']
# 划分数据集
x_train, x_text, y_train, y_text = train_test_split(text_vectors, label, test_size=0.2)

model = RandomForestClassifier()
print('模型训练中------')
model.fit(x_train, y_train)
y_predict = model.predict(x_text)
acc_score = model.score(x_text,y_text)
ic(acc_score)

