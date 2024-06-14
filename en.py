import os
import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse


#getting the data

def data():
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

    try:
        df = pd.read_csv(csv_url, sep = ";")

    except Exception as e:
        raise e
    
def evaluate(actual, pred):
    rmse  = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)


def main(alpha, l1_ration):
    df = data():

    
