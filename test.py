import requests

url = "http://127.0.0.1:8000/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer myhackrxsecret123"
}

data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the title of this document?",
        "How many pages does it have?"
    ]
}

response = requests.post(url, headers=headers, json=data)
print("Status code:", response.status_code)
print("Response body:")
print(response.text)
