import requests
import json

data = {
        'age': 31, 
        'workclass': 'Private',
        'education': 'Masters',
        'marital-status':'Never-married',
        'occupation':'Prof-speciality',
        'relationship':'Not-in-family',
        'race':'White',
        'sex':'Female',
        'capital-gain':14000,
        'capital-loss':0,
        'hours-per-week':55,
        'native-country':'United-States'
    }

r = requests.post("https://udacity-ml-app.herokuapp.com/predict", data = json.dumps(data))

print(r.status_code)
print(r.json())