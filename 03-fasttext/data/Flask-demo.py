import fasttext
from flask import Flask
from flask import request
import jieba

# 实例化Flask对象
app = Flask(__name__)

# 请求响应函数
@app.route('/v1/fasttext/', methods=['POST'])
def predict():
    # 加载自定义的停用词标
    jieba.load_userdict('./data/data/stopwords.txt')
    # 提供训练好的模型路径
    model_save_path = './data/data/model/fasttext_model2.bin'
    # 实例化model
    model = fasttext.load_model(model_save_path)
    print('Fasttext模型实例化完毕')


    # 1.接受输入
    text = request.form['text']
    text_new = ' '.join(list(text))

    # 2.模型预测
    predict = model.predict(text_new)
    print(predict)
    result = predict
    return '预测结果为:{}'.format(result)


if __name__ == '__main__':
    app.run()