import fasttext

train_data_path = './data/data/train_fast.txt'
text_data_path = './data/data/test_fast.txt'

# 开启训练模式
model = fasttext.train_supervised(input=train_data_path)
print('词的数量', len(model.words))
print('标签的数量', model.labels)

# 开启测试模式
result = model.test(text_data_path)
print(result)

# fasttext1优化 自动化参数搜索(模型)
import fasttext
train_data_path = './data/data/train_data.txt'
text_data_path = './data/data/test_fast.txt'

# 模型训练
model = fasttext.train_supervised(input=train_data_path, wordNgrams=2,
                                  autotuneValidationFile=text_data_path,
                                  autotuneDuration=6)

result = model.test(text_data_path)
print(result)
model.save_model('./data/data/model/fasttext_model2.bin')
print('模型保存完成!')

# fasttext2优化 jieba分词(模型)
import fasttext
train_data_path = './data/data/train_data_jiebaex.txt'
text_data_path = './data/data/test_fast1.txt'

# 模型训练
model = fasttext.train_supervised(input=train_data_path, wordNgrams=2,
                                  autotuneValidationFile=text_data_path,
                                  autotuneDuration=6)

result = model.test(text_data_path)
print(result)
model.save_model('./data/data/model/fasttext_model3jiebalcut.bin')
print('模型保存完成!')