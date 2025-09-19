import requests
url = "http://localhost:8000/detect/"
files = {'file': open('s4.jpg', 'rb')}
headers = {'x-api-key': 'gY5pR3L8aB1nS9eK4dH6cJ7mF2qV4oX'}
response = requests.post(url, files=files, headers=headers)
print(response.text)
