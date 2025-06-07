import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_california_housing(test_size=0.2, random_state=42):
    """
    Memuat dan melakukan preprocessing dataset California Housing.
    Tahapan:
    1. Memuat dataset
    2. Mengisi missing values dengan median (jika ada)
    3. Menghapus duplikasi
    4. Memisahkan fitur dan target
    5. Membagi data menjadi training dan testing set
    6. Melakukan standardisasi fitur numerik
    
    Returns:
        X_train_scaled (pd.DataFrame): Data training yang sudah di-scale
        X_test_scaled (pd.DataFrame): Data testing yang sudah di-scale
        y_train (pd.Series): Target training
        y_test (pd.Series): Target testing
    """
    # 1. Memuat dataset
    data = fetch_california_housing(as_frame=True)
    # Handle if data is a tuple (older sklearn versions)
    if isinstance(data, tuple):
        data = data[0]
    if hasattr(data, 'frame') or ('frame' in data):
        # Support both attribute and dict access
        df = data.frame.copy() if hasattr(data, 'frame') else data['frame'].copy()
    else:
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['MedHouseVal'] = data.target

    # 2. Mengisi missing values dengan median (jaga-jaga jika ada)
    df.fillna(df.median(), inplace=True)

    # 3. Menghapus duplikasi
    df = df.drop_duplicates()

    # 4. Memisahkan fitur dan target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    # 5. Membagi data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 6. Standardisasi fitur numerik
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Kembalikan dalam bentuk DataFrame agar mudah digunakan
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_california_housing()
    processed = pd.concat([X_train, y_train], axis=1)
    processed.to_csv("Preprocessing/CaliforniaHousing_preprocessing.csv", index=False)
    print("Preprocessing/CaliforniaHousing_preprocessing.csv berhasil dibuat.")
