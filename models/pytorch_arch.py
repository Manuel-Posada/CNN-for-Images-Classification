# app/models/pytorch_arch.py
"""Arquitecturas PyTorch.

Contrato de salida (equivalente al de `tensorflow_arch.py`, que usa
`activation="softmax"` en la última capa + `sparse_categorical_crossentropy`):

    - TabularNet -> logits de forma (batch, n_classes). NO se aplica ninguna
      activación final a propósito: `nn.CrossEntropyLoss` espera logits, y
      `utils.inference.run_inference` ya decide entre sigmoide (1 salida) o
      argmax (varias salidas) leyendo `preds.shape[1]`.
    - ImageCNN   -> logits (batch, n_classes), admite cualquier tamaño de
      entrada (por defecto el 128x128 que produce `utils.preprocess.preprocess_image`).
    - AudioCNN   -> logits (batch, n_classes), admite espectrogramas de
      cualquier tamaño (mel-spectrogramas de `utils.preprocess.preprocess_audio`).
"""
import torch
import torch.nn as nn


class TabularNet(nn.Module):
    """MLP para datos tabulares. Valores por defecto acordes al dataset Iris
    (4 medidas de flor, 3 especies), que es el caso tabular del proyecto."""

    def __init__(self, input_dim: int = 4, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),  # logits: sin activación final
        )

    def forward(self, x):
        return self.net(x)


class ImageCNN(nn.Module):
    """CNN para imágenes. `AdaptiveAvgPool2d` hace que la cabeza densa no
    dependa del tamaño de entrada, así la misma arquitectura sirve para 32x32
    o para el 128x128 que genera el preprocesado."""

    def __init__(self, n_channels: int = 3, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, n_classes),  # logits: sin activación final
        )

    def forward(self, x):
        return self.net(x)


class AudioCNN(nn.Module):
    """CNN para mel-espectrogramas (1 canal)."""

    def __init__(self, n_classes: int = 35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, n_classes),  # logits: sin activación final
        )

    def forward(self, x):
        return self.net(x)
