"""API FastAPI del clasificador de imagenes de animales (PyTorch).

Endpoints:
    GET  /models                 -> resumen de los modelos disponibles
    GET  /models/{arch}/metrics  -> metricas completas de un modelo
    POST /predict/               -> inferencia sobre una imagen subida

Las metricas se LEEN de `models/saved/metrics.json`; nunca se recalculan.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from models.pytorch_models import (
    available_archs,
    get_img_size,
    is_included,
    load_config,
    load_metrics,
)
from utils.inference import run_inference
from utils.preprocess import preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clasificador de animales (PyTorch)")


@app.get("/models")
def list_models() -> list[dict]:
    """Resumen de cada arquitectura y sus metricas de test."""
    config = load_config()
    return [
        {
            "arch": arch,
            "label": entry["label"],
            "included": is_included(arch),
            "img_size": entry["img_size"],
            "test_accuracy": entry.get("test_accuracy"),
            "test_balanced_accuracy": entry.get("test_balanced_accuracy"),
            "test_macro_f1": entry.get("test_macro_f1"),
        }
        for arch, entry in config["models"].items()
    ]


@app.get("/models/{arch}/metrics")
def model_metrics(arch: str) -> dict:
    """Metricas completas (metrics.json) del modelo `arch`, con classes y split_sizes."""
    metrics = load_metrics()
    section = metrics["models"].get(arch)
    if section is None:
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado: '{arch}'")

    return {
        "arch": arch,
        "classes": metrics["classes"],
        "split_sizes": metrics["split_sizes"],
        **section,
    }


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    arch: str = Form(...),
    topk: int = Form(3),
):
    """Clasifica una imagen con la arquitectura `arch` y devuelve el top-k."""
    if arch not in available_archs():
        raise HTTPException(
            status_code=422,
            detail=f"Arquitectura no valida: '{arch}'. Opciones: {available_archs()}",
        )
    if not is_included(arch):
        raise HTTPException(
            status_code=422,
            detail=(
                f"El modelo '{arch}' no esta incluido en esta version de la app "
                "(es mas pesado). Sus metricas siguen disponibles en la comparacion de modelos."
            ),
        )
    if topk < 1:
        raise HTTPException(status_code=400, detail="'topk' debe ser >= 1")

    content = await file.read()
    try:
        x = preprocess_image(content, arch)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen valida") from exc

    try:
        result = run_inference(x, arch, topk=topk)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Los pesos del modelo '{arch}' no estan disponibles en el servidor",
        ) from exc
    return {
        "arch": arch,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "top_k": result["top_k"],
        "probabilities": result["probabilities"],
        "inference_ms": result["inference_ms"],
        "img_size": get_img_size(arch),
    }
