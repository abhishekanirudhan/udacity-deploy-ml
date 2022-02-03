import requests
import json

data = {
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States"
    }
r = requests.post("http://127.0.0.1:8000/", data = json.dumps(data))

print(r.json())