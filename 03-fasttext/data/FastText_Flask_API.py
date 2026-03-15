import re

from flask import Flask, Response, json
from flask import request
import fasttext
import jieba
import time
app = Flask(__name__)
jieba.load_userdict('./data/stopwords.txt')
# 提供训练好的模型路径
model_save_path = './data/model/fasttext_model3jiebalcut.bin'
# 实例化model
model = fasttext.load_model(model_save_path)

@app.route('/NewsCls_submit', methods=['GET'])
def NewsCls_submit():
    with open('./data/NewsCls_submit.html', 'rb') as file:
        content = file.read()

    return content


@app.route('/NewsCls_handle', methods=['POST'])
def NewsCls_handle():
    # 获取json格式的输入
    request_json = request.get_json()
    email_data = request_json['content']


    # 调用模型预测
    t1 = time.time()
    text_new = ' '.join(jieba.lcut(email_data))
    # 模型预测
    pre = model.predict(text_new)
    result = pre[0][0]
    match = re.search('__label__(.+)', result)
    result = match.group(1)
    t2 = time.time()
    # 返回预测结果
    respose_data = {
        'status': 'success',
        "Result": result,
        'Time': '{:.4f}s'.format(t2-t1)
    }


    return Response(status=200, response=json.dumps(respose_data, sort_keys=False))

if __name__ == '__main__':
    app.run()
