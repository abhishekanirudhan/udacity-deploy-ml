import os
import pytest
import json

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get():
    """
    Test GET from main app, should return a welcome message
    """
    r = client.get("/")

    assert r.status_code == 200
    assert r.json() == {"Greetings": "Welcome to my ML Model!"}


def test_pred_post_high():
    """
    Enter a list of parameters knowing the answer will be > 50k
    """
    higher_parameters = {
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


    r = client.post("/predict", json=higher_parameters)

    assert r.status_code == 200
    assert r.json() == "Salary > 50k"

def test_pred_post_low():
    """
    Enter a list of parameters knowing the answer will be <= 50k
    """
    under_parameters = {
        'age': 22, 
        'workclass': 'Private',
        'education': '11th',
        'marital-status':'Never-married',
        'occupation':'Handlers-cleaners',
        'relationship':'Not-in-family',
        'race':'White',
        'sex':'Male',
        'capital-gain':10,
        'capital-loss':0,
        'hours-per-week':55,
        'native-country':'Honduras'
    }


    r = client.post("/predict", json=under_parameters)

    assert r.status_code == 200
    assert r.json() == "Salary is <= 50k"