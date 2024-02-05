import json

import requests

url = "http://127.0.0.1:6677/get"
message = "what is kicode"
data = {'user_input': message}


with requests.post(url, data=json.dumps(data),  stream=True) as r:
    
    for chunk in r.iter_content(1024):
        print(chunk)
