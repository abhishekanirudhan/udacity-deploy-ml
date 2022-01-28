from re import M
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from joblib import load
from ml.data import process_data

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
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
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

def compute_score_per_slice(trained_model, test, encoder,
                            lb, cat_features):
    """
    Compute score per category class slice
    Parameters
    ----------
    trained_model
    test
    encoder
    lb
    Returns
    -------
    """
    with open('model/slice_output.txt', 'w') as file:
        for category in cat_features:
            for cls in test[category].unique():
                temp_df = test[test[category] == cls]

                x_test, y_test, _, _ = process_data(
                    temp_df,
                    categorical_features=cat_features, training=False,
                    label="salary", encoder=encoder, lb=lb)

                pred = trained_model.predict(x_test)

                prc, rcl, fb = compute_model_metrics(y_test, pred)

                metric_info = "[%s]-[%s] Precision: %s " \
                              "Recall: %s FBeta: %s" % (category, cls,
                                                        prc, rcl, fb)
                logging.info(metric_info)
                file.write(metric_info + '\n')