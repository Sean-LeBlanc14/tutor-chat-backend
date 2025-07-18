import requests

url = "http://localhost:8080/api/signup"
data = {
    "email": "test@example.com",
    "password": "supersecure123"
}

response = requests.post(url, json=data)

print(response.json())
