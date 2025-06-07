import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Load data hasil preprocessing
DATA_PATH = "CaliforniaHousing_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur dan target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data (gunakan 80:20, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai MLflow run
with mlflow.start_run():
    # Inisialisasi dan latih model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    # Model dan metrik otomatis dilog oleh autolog
    
print("Training selesai. Silakan jalankan 'mlflow ui' untuk melihat hasilnya.")
