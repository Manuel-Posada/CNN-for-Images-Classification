# Clasificador de imágenes de animales — CNN / ResNet18 / AlexNet

Sistema completo (backend **FastAPI** + frontend **Streamlit**) para clasificar imágenes
de animales con cuatro modelos **PyTorch** entrenados sobre el dataset `animal_data`
(15 clases: Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse,
Kangaroo, Lion, Panda, Tiger, Zebra):

| Arquitectura | Tamaño de entrada | Tipo | ¿Incluida en la app? |
|---|---|---|---|
| `cnn` | 32 × 32 | CNN desde cero | Sí |
| `resnet18` | 128 × 128 | Transfer learning (pretrained=False) | Sí |
| `resnet18_32` | 32 × 32 | Transfer learning (pretrained=False) | No (solo se comparan sus métricas) |
| `alexnet` | 128 × 128 | Transfer learning (pretrained=False) | No (solo se comparan sus métricas) |

## Qué hace la app

- **Backend (FastAPI, `app/main.py`)**: expone los endpoints de listado de modelos,
  métricas y predicción. Lee la configuración y las métricas de `models/saved/` —
  **nunca** recalcula ni descarga nada.
- **UI (Streamlit, `ui/app.py`)** con tres pestañas:
  1. **Clasificar imagen**: sube una imagen (png/jpg/jpeg/webp), predicción con top-3,
     confianza, tiempo de inferencia y rendimiento histórico de la clase predicha.
  2. **Métricas del modelo**: curvas de entrenamiento, matriz de confusión normalizada,
     curvas ROC y tabla de métricas por clase.
  3. **Comparar modelos**: los cuatro modelos lado a lado con sus métricas y ROC superpuesta.

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
│       ├── model_config.json    #  Versionado (campo `included` por modelo)
│       ├── metrics.json         #  Versionado (métricas de los 4 modelos → comparaciones)
│       ├── cnn.pth              #  Incluido en la app, NO versionado
│       ├── resnet18.pth         #  Incluido en la app, NO versionado
│       # Los pesos pesados (alexnet.pth, resnet18_32.pth) NO se incluyen en esta
│       # versión; sus métricas sí (metrics.json) para la pestaña "Comparar modelos".
├── ui/
│   └── app.py                   # Frontend Streamlit
├── utils/
│   ├── preprocess.py            # Preprocesado: Resize → ToTensor → Normalize (desde el JSON)
│   └── inference.py             # Inferencia: softmax, top-k, tiempos
├── notebooks/
│   └── Animal_CNN_TransferLearning_PyTorch.ipynb   # Entrenamiento de los 4 modelos
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
| `GET` | `/models` | Arquitecturas y sus métricas de test: `arch`, `label`, `included` (si se ofrece en esta versión de la app), `img_size`, `params_total`, `params_trainable`, `epochs`, `train_time_s`, `ms_per_image`. |
| `GET` | `/models/{arch}/metrics` | Métricas completas de un modelo: `test`, `per_class`, `confusion_matrix`, `roc`, `roc_per_class`, `history`, etc. **404** si la arquitectura no existe. |
| `POST` | `/predict/` | Multipart con `file` (imagen), `arch` y `topk` (default 3). Devuelve `predicted_class`, `confidence`, `top_k`, `probabilities`, `inference_ms` e `img_size`. Errores: **400** imagen inválida, **422** validación / arquitectura desconocida o no incluida, **503** pesos no disponibles en el servidor. |

Ejemplo de predicción con `curl` (Windows PowerShell usa `curl.exe`):

```bash
curl -X POST http://localhost:8000/predict/ \
  -F "file=@panda.jpg" \
  -F "arch=resnet18" \
  -F "topk=3"
```

## Origen de los modelos y las métricas

Los cuatro modelos se entrenan (desde cero la CNN y por *transfer learning* ResNet18 y
AlexNet, con `resnet18_32` igual a `resnet18` pero con imágenes de 32×32) en
**`notebooks/Animal_CNN_TransferLearning_PyTorch.ipynb`** con el dataset
`animal_data`. Al final del notebook se exportan los pesos y las métricas a:

```
models/saved/
├── model_config.json   # clases (15), mean/std de normalización (ImageNet),
│                       # img_size por arquitectura, archivo .pth y `included`
├── metrics.json        # métricas completas de evaluación: test, per_class,
│                       # confusion_matrix, roc, roc_per_class, history
├── cnn.pth             # pesos de la CNN (32 px)       -> incluido en la app
└── resnet18.pth        # pesos de ResNet18 (128 px)    -> incluido en la app
                        # resnet18_32.pth y alexnet.pth (más pesados) NO se incluyen
                        # en esta versión; sus métricas sí (metrics.json)
```

- **`model_config.json`** es la fuente de verdad: la API lee de ahí las clases, la
  media/desviación de normalización, el tamaño de entrada por arquitectura y el campo
  `included` — nada está *hardcodeado* en el código.
- **`metrics.json`** contiene exactamente las métricas que sirve la API
  (`GET /models/{arch}/metrics`), incluidas las de `alexnet` y `resnet18_32`; la UI las
  muestra sin recalcular nada (pestaña «Comparar modelos»).
- **`included`** marca qué modelos se ofrecen en esta versión de la app. `cnn` y
  `resnet18` son ligeros y se incluyen; `alexnet` (~217 MB) y `resnet18_32` (~43 MB) son
  más pesados, así que quedan con `included: false`: no aparecen en el selector y
  `/predict/` los rechaza, pero **sus métricas y comparaciones siguen disponibles**.
- **Los `.pth` NO están versionados** (`.gitignore`): `alexnet.pth` pesa ~217 MB, por
  encima del límite de GitHub (100 MB por archivo). Si clonas el repositorio, genera los
  pesos ejecutando el notebook y deja los archivos en `models/saved/` con los nombres
  exactos de `model_config.json`. Solo los dos JSON viajan en el repositorio.
- **`pretrained=False`**: las arquitecturas se reconstruyen en blanco (`weights=None`)
  y se cargan los pesos guardados desde disco. Al arrancar el backend **no se descarga
  nada de internet**.