import pandas as pd
import numpy as np
import pytest
import sys
import logging
import os
from sklearn.model_selection import train_test_split

from starter.ml.data import import_data, process_data
from starter.ml.model import train_model, inference, model_metrics


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")

logger = logging.getLogger()

data_folder = os.path.abspath("./data/clean/")


@pytest.fixture
def load_data():
    
    path = os.path.join(data_folder, "clean_census.csv")
    data = pd.read_csv(path)
    return data

def test_import_data(load_data):
    
    assert load_data.shape[0] > 0
    assert load_data.shape[1] > 0

def test_columns_names(load_data):

    expected_cols = [
        "age",
        "workclass",
        "education",
        "marital-status",        
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary"
    ]

    curr_cols = load_data.columns.values
    
    assert expected_cols == list(curr_cols)
    logger.info("Column names do not match")


def test_age_range(load_data, min_age=0, max_age=100):

    ages = load_data['age'].between(min_age, max_age)

    assert np.sum(~ages) == 0
    logger.info("Age column has ages below or above min/max age")

def test_model_inference(load_data):
    
    train, test = train_test_split(load_data, test_size=0.2)

    
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)


    X_test, y_test, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)


    #train model from model.py
    model = train_model(X_train, y_train)

    #Making predictions with inference from model.py
    predictions = inference(model, X_test)

    #collecting metrics to see if the predictions are happening
    precision, recall, fbeta = model_metrics(y_test, predictions)

    assert precision > 0
    assert recall > 0
    assert fbeta > 0