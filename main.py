from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

from starter.ml.data import process_data


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


model_folder = os.path.abspath("./model/")

model_pkl = os.path.join(model_folder, "model.pkl")
encoder_pkl = os.path.join(model_folder, "encoder.pkl")
lb_pkl = os.path.join(model_folder, "lb.pkl")

model = joblib.load(model_pkl)
encoder = joblib.load(encoder_pkl)
lb = joblib.load(lb_pkl)


class Input(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str

app = FastAPI()

@app.get("/")
async def welcome():
    return {"Greetings": "Welcome to my ML Model!"}

@app.post("/")
async def predictions(item:Input):

    inputs = {
        "workclass": [item.workclass],
        "education": [item.education],
        "marital-status": [item.marital_status],
        "occupation": [item.occupation],
        "relationship": [item.relationship],
        "race": [item.race],
        "sex": [item.sex],
        "native-country": [item.native_country]
    }

    df = pd.DataFrame.from_dict(inputs)

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
    ]

    X,_,_,_ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    y_pred = model.predict(X)
    if y_pred[0] == 0:
        output = "Salary is <= 50k"
    else:
        output = "Salary > 50k"

    return output
