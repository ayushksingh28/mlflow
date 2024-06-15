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


def main(alpha, l1_ratio):
    df = data()

    train, test = train_test_split(df)

    train_x = train.drop(["quality"], axis = 1)
    test_x = test.drop(["quality"], axis = 1)

    train_y = train[["quality"]]
    test_y = test[["quality"]]

    lr = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = 42)
    lr.fit(train_x, train_y)

    pred = lr.predict(test_x)
    rmse, mae, r2 = evaluate(test_y, pred)

    print(f"Elastic Net parameters : {alpha}, l1_ratio: {l1_ratio}")
    print(f"Elasti Net Metric : rmse: {rmse}, mae: {mae}, r2_score: {r2}")


if __name__== "__main__":
    args = argparse.ArgumentParser()
    

