import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import os
from sklearn.metrics import max_error, median_absolute_error

# Konfigurasi MLflow ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/<USERNAME>/<REPO_NAME>.mlflow")
mlflow.set_experiment("CaliforniaHousing_Ridge_DagsHub")

# Load data hasil preprocessing
DATA_PATH = "CaliforniaHousing_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur dan target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data (gunakan 80:20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning dengan GridSearchCV
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'fit_intercept': [True, False]
}

ridge = Ridge()
gs = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)

best_model = gs.best_estimator_

# Mulai MLflow run manual logging
with mlflow.start_run(run_name="Ridge_GridSearchCV_DagsHub"):
    # Logging parameter terbaik
    mlflow.log_param("alpha", getattr(best_model, "alpha", None))
    mlflow.log_param("fit_intercept", getattr(best_model, "fit_intercept", None))
    mlflow.log_param("cv_folds", 5)
    # Logging tambahan: jumlah fitur dan jumlah data
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])
    
    # Training metrics (train set)
    y_train_pred = best_model.predict(X_train)
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(train_mse))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))
    mlflow.log_metric("training_mean_squared_error", train_mse)
    mlflow.log_metric("training_root_mean_squared_error", train_rmse)
    mlflow.log_metric("training_mean_absolute_error", train_mae)
    mlflow.log_metric("training_r2_score", train_r2)
    mlflow.log_metric("training_score", float(best_model.score(X_train, y_train)))
    # Tambahan: training max error & median absolute error
    train_max_error = float(max_error(y_train, y_train_pred))
    train_median_ae = float(median_absolute_error(y_train, y_train_pred))
    mlflow.log_metric("training_max_error", train_max_error)
    mlflow.log_metric("training_median_absolute_error", train_median_ae)
    
    # Test metrics (test set)
    y_test_pred = best_model.predict(X_test)
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))
    mlflow.log_metric("test_mean_squared_error", test_mse)
    mlflow.log_metric("test_root_mean_squared_error", test_rmse)
    mlflow.log_metric("test_mean_absolute_error", test_mae)
    mlflow.log_metric("test_r2_score", test_r2)
    mlflow.log_metric("test_score", float(best_model.score(X_test, y_test)))
    # Tambahan: test max error & median absolute error
    test_max_error = float(max_error(y_test, y_test_pred))
    test_median_ae = float(median_absolute_error(y_test, y_test_pred))
    mlflow.log_metric("test_max_error", test_max_error)
    mlflow.log_metric("test_median_absolute_error", test_median_ae)
    
    # Simpan model
    mlflow.sklearn.log_model(best_model, "model")
    print(f"Best Params: {gs.best_params_}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Test R2: {test_r2}")

print("Training dan tuning selesai. Jalankan 'mlflow ui' untuk melihat hasilnya.")
