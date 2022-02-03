import os
import unittest
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
        "workclass": "Private",
        "education": "Masters",
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States"
    }


    r = client.post("/", json=higher_parameters)
    assert r.status_code == 200
    assert r.json() == "Salary > 50k"

def test_pred_post_low():
    """
    Enter a list of parameters knowing the answer will be <= 50k
    """
    under_parameters = {
        "workclass": "Private",
        "education": "Some-college",
        "marital_status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "native_country": "Puerto-Rico"
    }


    r = client.post("/", json=under_parameters)
    assert r.status_code == 200
    assert r.json() == "Salary is <= 50k"