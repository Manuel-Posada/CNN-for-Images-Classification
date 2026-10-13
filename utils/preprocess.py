"""Preprocesamiento de imagenes para inferencia.

Reproduce EXACTAMENTE el pipeline de evaluacion del notebook de entrenamiento:

    PIL -> convert('RGB') -> Resize((img_size, img_size)) -> ToTensor -> Normalize(mean, std)

usando `torchvision.transforms`, de modo que las predicciones coincidan con las
del notebook. `img_size`, `mean` y `std` se leen de `models/saved/model_config.json`
(a traves de `models.pytorch_models`); no hay valores hardcodeados.
"""
from __future__ import annotations

import io
from functools import lru_cache

import torch
from PIL import Image
from torchvision import transforms

from models.pytorch_models import get_img_size, get_mean_std


@lru_cache(maxsize=None)
def _build_transform(arch: str) -> transforms.Compose:
    img_size = get_img_size(arch)
    mean, std = get_mean_std()
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def preprocess_image(file_bytes: bytes, arch: str) -> torch.Tensor:
    """Convierte los bytes de una imagen en el tensor (C, H, W) que espera `arch`.

    Lanza `PIL.UnidentifiedImageError`/`OSError` si los bytes no son una imagen valida.
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return _build_transform(arch)(image)
