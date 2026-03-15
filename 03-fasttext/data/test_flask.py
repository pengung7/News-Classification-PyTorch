import requests

url = "http://127.0.0.1:5000/v1/fasttext/"
data = {"text":"雷佳音获得飞天奖"}

res = requests.post(url, data=data)
print(f'分类结果{res}')