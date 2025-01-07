import requests


External_IP_address = "None"


url = f'http://{External_IP_address}/predict'


data = {'url': 'http://bit.ly/mlbookcamp-pants'}


result = requests.post(url, json = data).json()

print(result)