import pandas as pd
import numpy as np
import pytest
import sys
import logging
import os

from ml.data import import_data


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")

logger = logging.getLogger()

data_folder = os.path.abspath("./data/clean/")


@pytest.fixture
def load_data():
    
    path = os.path.join(data_folder, "clean_census.csv")
    data = import_data(path)
    return data

def test_import_data(load_data):
    
    assert load_data.shape[0] > 0
    assert load_data.shape[1] > 0

def test_columns_names(load_data):
    expected_cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income"
    ]

    curr_cols = load_data.columns.values
    
    assert expected_cols == list(curr_cols)
    logger.info("Column names do not match")


def test_age_range(load_data, min_age=0, max_age=100):

    ages = load_data['age'].between(min_age, max_age)

    assert np.sum(~ages) == 0
    logger.info("Age column has ages below or above min/max age")