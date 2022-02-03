from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import uvicorn

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
    age: int = Field(..., example = 22)
    workclass: str = Field(..., example = 'Never-worked')
    education: str = Field(..., example = 'Doctorate')
    marital_status: str = Field(..., example = 'Separated', alias='marital-status')
    occupation: str = Field(..., example = 'Craft-repair')
    relationship: str = Field(..., example = 'Husband')
    race: str = Field(..., example = 'Other')
    sex: str = Field(..., example = 'Female')
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., example = 'India', alias='native-country')


app = FastAPI()

@app.get("/")
async def welcome():
    return {"Greetings": "Welcome to my ML Model!"}

@app.post("/predict")
async def predictions(item:Input):

    inputs = {
        "age": [item.age],
        "workclass": [item.workclass],
        "education": [item.education],
        "marital-status": [item.marital_status],
        "occupation": [item.occupation],
        "relationship": [item.relationship],
        "race": [item.race],
        "sex": [item.sex],
        "capital-gain": [item.capital_gain],
        "capital-loss": [item.capital_loss],
        "hours-per-week": [item.hours_per_week],
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

