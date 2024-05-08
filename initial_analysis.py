
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

ignore = ["Mass(1)", "Mass(2)", "Eccentricity@DCO", "SemiMajorAxis@DCO", "Coalescence_Time"]
DATADIR = Path("/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/stellar-paleontology-data/")

def load_classification_data(datadir=DATADIR, metallicity="all"):
    double_compact_objects = pd.read_pickle(datadir / "compas-data.pkl")
    for key in ignore:
        if key in double_compact_objects:
            del double_compact_objects[key]
    return double_compact_objects



def df_stats(data_df):
    pd.set_option('display.width', None) 
    data_df = load_classification_data()
    print(data_df.describe())
    print(data_df.head(5))
    print(data_df.columns)   
    col_names = list(data_df.columns)
    print(col_names)
    print("column value counts:")
    for col in col_names:
        print(data_df[col].value_counts())   
    print(data_df.isna().sum())



if __name__ == "__main__":
    data_df =     load_classification_data()
    df_stats(data_df)
