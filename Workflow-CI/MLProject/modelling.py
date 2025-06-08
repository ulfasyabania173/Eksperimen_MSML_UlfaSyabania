import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='CaliforniaHousing_preprocessing.csv')
args = parser.parse_args()

mlflow.sklearn.autolog()

# Load data hasil preprocessing
df = pd.read_csv(args.data_path)

# Pisahkan fitur dan target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    mlflow.sklearn.log_model(model, "model")

print("Training selesai.")
