"""
Helper functions to train, run inference, slice and score 
a logistic regression 

Author: Abhishek Anirudhan
Date: Feb 3, 2022
"""


import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    model = lr.fit(X_train, y_train)
    return model

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : joblib.dump
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pred = model.predict(X)
    return pred


def model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def slice_census(X_test, y_test, y_pred, features):
    """
    Calcule metrics on a slice of the data
    """
    
    df = X_test.drop(["salary"], axis=1)
    df["salary"] = y_test
    df["salary_pred"] = y_pred

    slices = []
    for feature in features:
        for val in df[feature].unique():
            precision, recall, fbeta = model_metrics(
                df[df[feature]==val]["salary"],
                df[df[feature]==val]["salary_pred"]
            )
            slices.append([val, precision, recall, fbeta])

    slice_df = pd.DataFrame(
        slices,
        columns=["Category",
                "Precision",
                "Recall",
                "Fbeta"])
    
    return slice_df