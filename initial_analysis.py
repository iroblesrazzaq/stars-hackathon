from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

ignore = ["Mass(1)", "Mass(2)", "Eccentricity@DCO", "SemiMajorAxis@DCO", "Coalescence_Time"]
DATADIR = Path("/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/stellar-paleontology-data/")

def load_classification_data(datadir=DATADIR):
    double_compact_objects = pd.read_pickle(datadir / "compas-data.pkl")
    for key in ignore:
        if key in double_compact_objects:
            del double_compact_objects[key]
    return double_compact_objects

def df_stats(data_df):
    pd.set_option('display.max_columns', None)
    print(data_df.describe())
    print(data_df.head(5))
    print(data_df.columns)
    col_names = list(data_df.columns)
    print(col_names)
    print("column value counts:")
    for col in col_names:
        print(data_df[col].value_counts())
    print(data_df.isna().sum())

def statistics(confusion: np.ndarray, verbose=True) -> float:
    """
    Compute summary statistics based on the confusion matrix.
    ...
    """
    confusion = np.nan_to_num(confusion / confusion.sum(axis=0))
    true_positive_rate = confusion[1, 1]
    true_negative_rate = confusion[0, 0]
    false_positive_rate = confusion[1, 0]
    false_negative_rate = confusion[0, 1]
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
    if verbose:
        print(f"True positive rate: {true_positive_rate:.4f}")
        print(f"False positive rate: {false_positive_rate:.4f}")
        print(f"True negative rate: {true_negative_rate:.4f}")
        print(f"False negative rate: {false_negative_rate:.4f}")
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    return balanced_accuracy

def build_model(data_df):
    y = data_df['Merges_Hubble_Time']
    data_df.drop(['Merges_Hubble_Time'], axis=1, inplace=True)
    X = data_df
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    xgb_model_1 = XGBClassifier(random_state=0, n_estimators=500, early_stopping_rounds=5, learning_rate=0.05)
    xgb_model_1.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    predictions_1 = xgb_model_1.predict(X_valid)
    return xgb_model_1, predictions_1, X_valid, y_valid

def evaluate_model(model, predictions, y_valid):
    confusion = confusion_matrix(y_valid, predictions)
    balanced_accuracy = statistics(confusion, verbose=True)
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    return confusion, balanced_accuracy

if __name__ == "__main__":
    data_df = load_classification_data()
    df_stats(data_df)
    model, predictions, X_valid, y_valid = build_model(data_df)
    evaluate_model(model, predictions, y_valid)
