import requests

resp = requests.post("http://127.0.0.1:5000/testpredict", files = {'file': open('five.png', 'rb')})

print(resp.text)


# resp = requests.get("http://127.0.0.1:5000/get_pred_result")

# print(resp.text)


# print('all tests sucessful')