# Clasificador de imágenes de animales — CNN / ResNet18 / AlexNet

Sistema completo (backend **FastAPI** + frontend **Streamlit**) para clasificar imágenes
de animales con tres modelos **PyTorch** entrenados sobre el dataset `animal_data`
(15 clases: Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse,
Kangaroo, Lion, Panda, Tiger, Zebra):

| Arquitectura | Tamaño de entrada | Tipo |
|---|---|---|
| `cnn` | 32 × 32 | CNN desde cero |
| `resnet18` | 128 × 128 | Transfer learning (pretrained=False) |
| `alexnet` | 128 × 128 | Transfer learning (pretrained=False) |

## Qué hace la app

- **Backend (FastAPI, `app/main.py`)**: expone los endpoints de listado de modelos,
  métricas y predicción. Lee la configuración y las métricas de `models/saved/` —
  **nunca** recalcula ni descarga nada.
- **UI (Streamlit, `ui/app.py`)** con tres pestañas:
  1. **Clasificar imagen**: sube una imagen (png/jpg/jpeg/webp), predicción con top-3,
     confianza, tiempo de inferencia y rendimiento histórico de la clase predicha.
  2. **Métricas del modelo**: curvas de entrenamiento, matriz de confusión normalizada,
     curvas ROC y tabla de métricas por clase.
  3. **Comparar modelos**: los tres modelos lado a lado con sus métricas y ROC superpuesta.

Toda la información mostrada por la UI se consume de la API; el frontend no recalcula
métricas.

## Estructura del repositorio

```
CNN-for-Images-Classification/
├── app/
│   └── main.py                  # API FastAPI: endpoints y lógica HTTP
├── models/
│   ├── pytorch_arch.py          # Arquitecturas: build_model(arch, num_classes, pretrained=False)
│   ├── pytorch_models.py        # Carga lazy + cacheada de config, métricas y pesos
│   └── saved/                   # Salidas del notebook (ver "Origen de los modelos")
│       ├── model_config.json    #  Versionado
│       ├── metrics.json         #  Versionado
│       ├── cnn.pth              #  En disco, NO versionado
│       ├── resnet18.pth         #  En disco, NO versionado
│       └── alexnet.pth          #  En disco, NO versionado (217 MB)
├── ui/
│   └── app.py                   # Frontend Streamlit
├── utils/
│   ├── preprocess.py            # Preprocesado: Resize → ToTensor → Normalize (desde el JSON)
│   └── inference.py             # Inferencia: softmax, top-k, tiempos
├── notebooks/
│   └── Animal_CNN_TransferLearning_PyTorch.ipynb   # Entrenamiento de los 3 modelos
├── doc/                         # Material del curso (presentación)
├── requirements.txt
└── .gitignore
```

(Los paquetes `app/`, `models/`, `ui/` y `utils/` contienen su `__init__.py`.)

## Entorno virtual e instalación

Requisito: Python 3.10+ (se recomienda 3.11/3.12).

### Windows (PowerShell / cmd)

```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Abre **dos terminales** (con el venv activado). Primero el backend:

```
uvicorn app.main:app --reload --port 8000
```

Y en la otra, la interfaz:

```
streamlit run ui/app.py
```

Abre `http://localhost:8501` en el navegador. La UI llama a `http://localhost:8000`
por defecto; si el backend corre en otro puerto, usa la variable de entorno
`API_URL` (por ejemplo `API_URL=http://localhost:8001 streamlit run ui/app.py`).

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/models` | Arquitecturas disponibles: `arch`, `label`, `img_size`, `params_total`, `params_trainable`, `epochs`, `train_time_s`, `ms_per_image`, métricas de test. |
| `GET` | `/models/{arch}/metrics` | Métricas completas de un modelo: `test`, `per_class`, `confusion_matrix`, `roc`, `roc_per_class`, `history`, etc. **404** si la arquitectura no existe. |
| `POST` | `/predict/` | Multipart con `file` (imagen), `arch` y `topk` (default 3). Devuelve `predicted_class`, `confidence`, `top_k`, `probabilities`, `inference_ms` e `img_size`. Errores: **400** imagen inválida, **404** arquitectura desconocida, **422** validación. |

Ejemplo de predicción con `curl` (Windows PowerShell usa `curl.exe`):

```bash
curl -X POST http://localhost:8000/predict/ \
  -F "file=@panda.jpg" \
  -F "arch=alexnet" \
  -F "topk=3"
```

## Origen de los modelos y las métricas

Los tres modelos se entrenan (desde cero la CNN y por *transfer learning* ResNet18 y
AlexNet) en **`notebooks/Animal_CNN_TransferLearning_PyTorch.ipynb`** con el dataset
`animal_data`. Al final del notebook se exportan los pesos y las métricas a:

```
models/saved/
├── model_config.json   # clases (15), mean/std de normalización (ImageNet),
│                       # img_size por arquitectura y nombre de su archivo .pth
├── metrics.json        # métricas completas de evaluación: test, per_class,
│                       # confusion_matrix, roc, roc_per_class, history
├── cnn.pth             # pesos de la CNN (32 px)
├── resnet18.pth        # pesos de ResNet18 (128 px)
└── alexnet.pth         # pesos de AlexNet (128 px)
```

- **`model_config.json`** es la fuente de verdad del preprocesado: la API lee de ahí
  las clases, la media/desviación de normalización y el tamaño de entrada por
  arquitectura — nada está *hardcodeado* en el código.
- **`metrics.json`** contiene exactamente las métricas que sirve la API
  (`GET /models/{arch}/metrics`); la UI las muestra sin recalcular nada.
- **Los `.pth` NO están versionados** (`.gitignore`): `alexnet.pth` pesa ~217 MB,
  por encima del límite de GitHub (100 MB por archivo). Si clonas el repositorio,
  genera los pesos ejecutando el notebook y deja los archivos en `models/saved/` con
  los nombres exactos de `model_config.json` (`cnn.pth`, `resnet18.pth`, `alexnet.pth`).
  Solo los dos JSON viajan en el repositorio.
- **`pretrained=False`**: las arquitecturas se reconstruyen en blanco (`weights=None`)
  y se cargan los pesos guardados desde disco. Al arrancar el backend **no se descarga
  nada de internet**.