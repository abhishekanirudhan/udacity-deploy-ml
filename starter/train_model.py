import pandas as pd
from sklearn.model_selection import train_test_split

import pickle

from ml.data import import_data, process_data
from ml.model import train_model, model_metrics, inference, slice_census

# Add code to load in the data.
path = "./data/clean/clean_census.csv"
data = import_data(path)

def go():

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",]

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

    model = train_model(X_train, y_train)

    X_test, y_test,_ ,_ = process_data(test,
                            categorical_features=cat_features,
                            label="salary",
                            training=False,
                            encoder=encoder,
                            lb=lb)

    y_pred = model.predict(X_test)

    slice_census_data = slice_census(test,
                                y_test,
                                y_pred,
                                cat_features)
    
    slice_census_data.to_csv("./data/slice_output.txt", 
                    header=None, 
                    index=None, 
                    sep=" ", 
                    mode="a")

    filename = "./model/model.pkl"
    pickle.dump(model, open(filename, "wb"))

    enc_filename = "./model/encoder.pkl"
    pickle.dump(encoder, open(enc_filename, "wb"))

    lb_filename = "./model/lb.pkl"
    pickle.dump(lb, open(lb_filename, "wb"))

if __name__ == '__main__':
    go()