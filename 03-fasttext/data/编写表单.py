from flask import Flask, Response, json
from flask import request
app = Flask(__name__)

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
    # 定义响应数据格式
    respose_data = {
        'status': 'success',
        "content": email_data,
    }


    return Response(status=200, response=json.dumps(respose_data, sort_keys=False))

if __name__ == '__main__':
    app.run()