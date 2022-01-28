# Script to train machine learning model.
from json import encoder
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model
from joblib import dump, load
from ml.model import inference
from ml.model import compute_model_metrics
from ml.model import compute_score_per_slice
import logging

# Add code to load in the data.
path = os.path.join(
    os.getcwd(), 
    "data/clean/clean_census.csv")

def train_test_data(path):
    """
    Get data and split into train & test sets.

    Returns
    ----------
    data, train_data, test_data, cat_features
    """
    data = pd.read_csv(path)
    
    train, test = train_test_split(data, test_size=0.20)
    
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
    
    return train, test, cat_features


def train_save_model(train, cat_features):
    """
    Train and dump model

    Returns
    -------

    """
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )

    trained_model = train_model(X_train, y_train)

    dump(trained_model, "model/trained_model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")


def test_model(test, cat_features):

    trained_model = load("model/trained_model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    X_test, y_test, _, _ =  process_data(
        test, 
        categorical_features=cat_features, 
        encoder = encoder,
        lb = lb, 
        training=False
    )

    pred = inference(trained_model, X_test)

    precision, recall, fbeta = compute_model_metrics(
        y_test, 
        pred
    )

    logging.info(f"Scores: precision = {precision}, recall = {recall}, fbeta = {fbeta}")

    compute_score_per_slice(
        trained_model,
        test,
        encoder,
        lb,
        cat_features
    )

if __name__ == '__main__':
    train, test, cat_features = train_test_data(path)
    train_save_model(train, cat_features)
    test_model(test, cat_features)
