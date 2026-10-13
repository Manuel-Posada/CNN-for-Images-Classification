"""Arquitecturas PyTorch del clasificador de imagenes de animales.

Contenido migrado desde `notebooks/model_defs.py` (misma logica, sin cambios)
para poder reconstruir los checkpoints guardados en `models/saved/*.pth`.

    - CNNBaseline  : CNN desde cero (entrada 32x32).
    - build_resnet18 / build_alexnet : transfer learning con backbone congelado.
    - build_model  : factoria por nombre de arquitectura.

Con `pretrained=False` no se descarga nada de internet: solo se instancia la
topologia para cargar encima el `state_dict` local.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNNBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1);   self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1);  self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)   # sin softmax: lo incluye CrossEntropyLoss

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(self.gap(x), 1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


def build_resnet18(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    for p in m.parameters():
        p.requires_grad = False                       # congelar backbone
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    return m


def build_alexnet(num_classes, pretrained=True):
    weights = models.AlexNet_Weights.DEFAULT if pretrained else None
    m = models.alexnet(weights=weights)
    for p in m.parameters():
        p.requires_grad = False                       # congelar todo
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)   # nueva capa de salida
    return m


def build_model(arch, num_classes, pretrained=True):
    if arch == 'cnn':
        return CNNBaseline(num_classes)
    if arch == 'resnet18':
        return build_resnet18(num_classes, pretrained)
    if arch == 'alexnet':
        return build_alexnet(num_classes, pretrained)
    raise ValueError(f'Arquitectura desconocida: {arch}')
