"""Carga perezosa y cacheada de los modelos PyTorch de animales.

Toda la metadata (clases, normalizacion, tamano de entrada, fichero de pesos)
se lee de `models/saved/model_config.json`: no hay valores hardcodeados aqui.

La arquitectura se reconstruye con `build_model(arch, num_classes, pretrained=False)`
y encima se carga el `state_dict` local, por lo que no se descarga nada de internet.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from .pytorch_arch import build_model

# Directorio de pesos/metadata: <repo>/models/saved (relativo a este paquete)
SAVED_DIR = Path(__file__).resolve().parent / "saved"
CONFIG_PATH = SAVED_DIR / "model_config.json"
METRICS_PATH = SAVED_DIR / "metrics.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """`model_config.json`: classes, mean, std y por modelo label/weights/img_size/metrics."""
    return _read_json(CONFIG_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    """`metrics.json`: classes, split_sizes y el desglose completo por modelo."""
    return _read_json(METRICS_PATH)


def get_classes() -> list[str]:
    return list(load_config()["classes"])


def available_archs() -> list[str]:
    return list(load_config()["models"].keys())


def get_model_entry(arch: str) -> dict[str, Any]:
    """Entrada de `model_config.json` para una arquitectura. KeyError si no existe."""
    return load_config()["models"][arch]


def get_img_size(arch: str) -> int:
    return int(get_model_entry(arch)["img_size"])


def get_mean_std() -> tuple[list[float], list[float]]:
    config = load_config()
    return list(config["mean"]), list(config["std"])


@lru_cache(maxsize=None)
def get_model(arch: str) -> torch.nn.Module:
    """Construye y carga (cacheado) el modelo `arch` en modo evaluacion.

    Lanza `KeyError` si la arquitectura no esta en `model_config.json` y
    `FileNotFoundError` si falta el checkpoint.
    """
    entry = get_model_entry(arch)  # KeyError -> el llamador lo traduce a 404

    num_classes = len(get_classes())
    module = build_model(arch, num_classes, pretrained=False)

    weights = SAVED_DIR / entry["weights"]
    if not weights.exists():
        raise FileNotFoundError(f"No se encontro el checkpoint: {weights}")

    state = torch.load(weights, map_location=DEVICE, weights_only=True)
    module.load_state_dict(state)
    module.to(DEVICE)
    module.eval()
    return module
