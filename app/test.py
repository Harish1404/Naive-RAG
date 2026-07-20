import requests

response = requests.post("https://hook.us2.make.com/fcd88iehc4l9rdm58mhr12zk199oixex", json={"name": "cfrsi", "message": "Hello"})

print(response.data)

