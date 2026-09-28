"""
Funciones de preprocesamiento para datos tabulares, de imagen y de audio.
Se usan tanto en entrenamiento como en predicción.
"""
import io
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

# Opcionales: solo si planeas usar imágenes/audio
from PIL import Image
import librosa


# -----------------------------
# Tabular
# -----------------------------
def preprocess_tabular(df: pd.DataFrame, target_column: str = None) -> tuple[np.ndarray, np.ndarray | None]:
    df = df.copy()

    # Separar y antes de limpiar X
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column].values
        df = df.drop(columns=[target_column])

    # Limpiar columnas con strings numéricos mal formateados
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .str.replace('.', '', regex=False)  # quitar puntos de miles
                .str.replace(',', '.', regex=False) # coma decimal a punto
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.astype(np.float32))

    return X, y


# -----------------------------
# Tabular - registro único (interfaz de predicción manual)
# -----------------------------
# Columnas de entrada del dataset Iris.
TABULAR_FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

SCALER_PATH = Path("models/saved/tabular_scaler.pkl")


def _load_or_fit_scaler() -> StandardScaler:
    """Carga el scaler guardado durante el entrenamiento; si aún no existe,
    ajusta uno de referencia con el dataset Iris para poder predecir de inmediato."""
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)

    ref_csv = Path("datasets/tabular/iris.csv")
    if ref_csv.exists():
        df = pd.read_csv(ref_csv)
        if "target" in df.columns:
            df = df.drop(columns=["target"])
        scaler = StandardScaler().fit(df.astype(np.float32))
    else:
        scaler = StandardScaler()

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def preprocess_tabular_record(values: list[float]) -> np.ndarray:
    """Convierte un único registro (4 medidas de la flor) en la matriz que
    espera el modelo, aplicando el mismo escalado usado en el entrenamiento."""
    scaler = _load_or_fit_scaler()
    df = pd.DataFrame([values], columns=TABULAR_FEATURE_COLUMNS)
    return scaler.transform(df.astype(np.float32))


# -----------------------------
def preprocess_image(file_bytes: bytes, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Abre una imagen, la convierte a RGB, la redimensiona y normaliza a [0,1].
    Retorna un array (H, W, 3).
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    return arr


# -----------------------------
# Audio
# -----------------------------
def preprocess_audio(file_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """
    Carga un archivo de audio en memoria, lo resamplea y extrae un mel-spectrogram.
    Retorna un array 2D (frecuencias x frames).
    """
    # librosa.load acepta un file-like object
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalizamos a [0,1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype("float32")
