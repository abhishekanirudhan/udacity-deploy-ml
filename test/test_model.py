import numpy as np
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")

logger = logging.getLogger()

data_folder = os.path.abspath("./data/")

@pytest.fixture
def load_data():
    
    path = os.path.join(data_folder, "clean_census.csv")
    data = pd.read_csv(path)
    return data

def test_import_data(data_read):
    
    assert data_read.shape[0] > 0
    assert data_read.shape[1] > 0

def test_columns_names(data_read):
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

    curr_cols = data_read.columns.values
    
    assert expected_cols == list(curr_cols)
    logger.info("Column names do not match")


def test_age_range(data_read, min_age=0, max_age=100):

    ages = data_read['age'].between(min_age, max_age)

    assert np.sum(~ages) == 0
    logger.info("Age column has ages below or above min/max age")