"""Inferencia sobre imagenes con los modelos PyTorch de animales.

Aplica softmax sobre los logits y devuelve la clase predicha, la confianza, el
top-k, el vector completo de probabilidades y el tiempo de inferencia en ms.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from models.pytorch_models import get_classes, get_model


def run_inference(x, arch: str, topk: int = 3) -> dict[str, Any]:
    """Ejecuta el modelo `arch` sobre la entrada `x`.

    `x` puede ser un tensor (C, H, W), un array/lote numpy o una lista. Devuelve un
    diccionario con predicted_class, confidence, top_k, probabilities e inference_ms.
    """
    model = get_model(arch)
    classes = get_classes()
    num_classes = len(classes)

    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)

    # (C, H, W) -> (1, C, H, W)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    device = next(model.parameters()).device
    x = x.to(device=device, dtype=torch.float32)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    inference_ms = (time.perf_counter() - start) * 1000.0

    k = max(1, min(int(topk), num_classes))
    top_values, top_indices = probs.topk(k)

    top_k = [
        {"class": classes[int(idx)], "prob": float(val)}
        for val, idx in zip(top_values, top_indices)
    ]
    probabilities = {cls: float(p) for cls, p in zip(classes, probs)}

    predicted_index = int(top_indices[0])
    return {
        "predicted_class": classes[predicted_index],
        "confidence": float(top_values[0]),
        "top_k": top_k,
        "probabilities": probabilities,
        "inference_ms": inference_ms,
    }
