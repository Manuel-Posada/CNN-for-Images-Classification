# utils/data.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split(csv_path, target_column="target", test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_column]).values.astype("float32")
    y = df[target_column].values.astype("float32")
    print(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Split into {X_train.shape[0]} training and {X_test.shape[0]} test samples.")
    return X_train, X_test, y_train, y_test 


def prepare_tabular_data(csv_path, target_column="target"):
    X_train, X_test, y_train, y_test = load_and_split(csv_path, target_column)
    print(f"Before scaling: X_train mean {X_train.mean(axis=0)}, std {X_train.std(axis=0)}")
    print(f"Before scaling: X_test mean {X_test.mean(axis=0)}, std {X_test.std(axis=0)}")
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Guardar el scaler para reutilizarlo en la predicción de un registro
    scaler_path = Path("models/saved/tabular_scaler.pkl")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print(f"prepare_tabular_data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"prepare_tabular_data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    return X_train, y_train, X_test, y_test