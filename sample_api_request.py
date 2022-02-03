import requests
import json

data = {
        "workclass": "Private",
        "education": "Masters",
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States"
    }

r = requests.post("http://127.0.0.1:8000/", data = json.dumps(data))

print(r.json())