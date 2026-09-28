from fastapi import FastAPI, UploadFile, File, Form
from utils.inference import run_inference
from utils.preprocess import preprocess_tabular, preprocess_tabular_record, preprocess_image, preprocess_audio
from pathlib import Path
from utils.data import load_and_split, prepare_tabular_data
from models import trainer_tf, trainer_pt
import pandas as pd
from io import StringIO
app = FastAPI()

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# En lugar de print():

@app.post("/predict/")
async def predict(
    data_type: str = Form(...),          # 'tabular' | 'image' | 'audio'
    framework: str = Form(...),          # 'tensorflow' | 'pytorch'
    csv_file: UploadFile = File(None),  # CSV para datos tabulares (opcional)
    file: UploadFile = File(None),      # Imagen o audio
    # Parámetros de un único registro tabular (Iris)
    sepal_length: float = Form(None),
    sepal_width: float = Form(None),
    petal_length: float = Form(None),
    petal_width: float = Form(None),
):
    # Preprocesar según tipo
    if data_type == "tabular":
        values = [sepal_length, sepal_width, petal_length, petal_width]
        if all(v is not None for v in values):
            # Predicción con un único registro introducido a mano
            x = preprocess_tabular_record(values)
        elif csv_file is not None:
            temp_path = Path("uploads") / csv_file.filename
            with open(temp_path, "wb") as f:
                f.write(await csv_file.read())
            df = pd.read_csv(temp_path)
            x, _ = preprocess_tabular(df)  # No hay target en predicción
        else:
            return {"error": "Proporciona los 4 parámetros de la flor o un CSV"}
        
    elif data_type == "image":
        x = preprocess_image(await file.read())
    elif data_type == "audio":
        x = preprocess_audio(await file.read())
    else:
        return {"error": "Tipo de dato no soportado"}

    result = run_inference(x, framework, data_type)
    return {"prediction": result}

@app.post("/train")
async def train_model(
    csv_file: UploadFile,
    framework: str = Form(...),
    epochs: int = Form(20),
    target_column: str = Form(...)  
):
    temp_path = Path("uploads") / csv_file.filename
    with open(temp_path, "wb") as f:
        f.write(await csv_file.read())

    logger.info(f"Training target_column: {target_column}")
    X_train, y_train, X_test, y_test = prepare_tabular_data(temp_path, target_column=target_column)
    logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}, {y_test.shape}")
    logger.info(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")

    if framework.lower() == "pytorch":
        model_path = trainer_pt.train_tabular(X_train, y_train, X_test, y_test, epochs)
    else:
        model_path = trainer_tf.train_tabular(X_train, y_train, X_test, y_test, epochs)

    return {"status": "ok", "saved_model": str(model_path)}
