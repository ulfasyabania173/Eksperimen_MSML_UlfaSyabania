import requests
import json

data = {"columns": ["col1", "col2", "col3"], "data": [[1, 2, 3]]}  # Ganti dengan kolom dan data sesuai model Anda
response = requests.post("http://localhost:5001/invocations", json=data)
print(response.json())
