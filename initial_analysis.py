from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sys

if len(sys.argv) > 1:
    DATADIR = Path(sys.argv[1])
else:
    DATADIR = Path("/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/stellar-paleontology-data/")

def load_classification_data(datadir=DATADIR, fraction=1.0):
    ignore = ["Mass(1)", "Mass(2)", "Eccentricity@DCO", "SemiMajorAxis@DCO", "Coalescence_Time"]
    double_compact_objects = pd.read_pickle(datadir / "compas-data.pkl")
    for key in ignore:
        if key in double_compact_objects:
            del double_compact_objects[key]
    if fraction < 1.0:
        double_compact_objects = double_compact_objects.sample(frac=fraction, random_state=42)
    return double_compact_objects

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
    xgb_params = {
                            'objective': 'binary:logistic',
                                                'eval_metric': 'logloss',
                                                        'use_label_encoder': False,
                                                                'tree_method': 'gpu_hist',
                                                                        'predictor': 'gpu_predictor',
                                                                                'n_estimators': 500,
                                                                                        'early_stopping_rounds': 5,
                                                                                                'learning_rate': 0.05
                                                                                                    }
    xgb_model_1 = XGBClassifier(**xgb_params)
    xgb_model_1.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=True)
    predictions_1 = xgb_model_1.predict(X_valid)
    return xgb_model_1, predictions_1, X_valid, y_valid

def evaluate_model(model, predictions, y_valid):
    confusion = confusion_matrix(y_valid, predictions)
    balanced_accuracy = statistics(confusion, verbose=True)
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    return confusion, balanced_accuracy

if __name__ == "__main__":
    fraction = 0.01  # Specify the fraction of the dataset to use (e.g., 0.1 for 10%)
    data_df = load_classification_data(fraction=fraction)
    model, predictions, X_valid, y_valid = build_model(data_df)
    confusion, balanced_acc = evaluate_model(model, predictions, y_valid)
