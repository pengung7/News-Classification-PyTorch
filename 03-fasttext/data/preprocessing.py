#
import jieba

# 2.id2label
id2label = {}
id = 0
with open('./data/class.txt', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        class_label = line.strip('\n').strip()
        id2label[id] = class_label
        id+=1

    print(f'id2label{id2label}')

# 3.标签转换
train_data = []
with open('./data/train.txt', 'r', encoding='utf-8') as f2:
    for line in f2.readlines():
        line = line.strip('\n').strip()
        sentence, label = line.split('\t')
        new_label = id2label[int(label)]
        new_label = '__label__' + new_label
        sentence_new = ' '.join(jieba.lcut(sentence))
        new_sent = new_label + ' ' + sentence_new
        train_data.append(new_sent)

# print(train_data[:5])
with open('./data/train_data_jiebaex.txt', 'w', encoding='utf-8') as f3:
    for line in train_data:
        f3.write(line+'\n')







