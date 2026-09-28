import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def run_inference(x, framework, data_type):
    if framework == "tensorflow":
        from models.tensorflow_models import get_model
        model = get_model(data_type)
        preds = model.predict(x)
    else:
        from models.pytorch_models import get_model
        model = get_model(data_type)
        import torch
        with torch.no_grad():
            preds = model(x).cpu().numpy()

    # Convertir probabilidades a clases
    if data_type == "tabular":
        if preds.shape[1] == 1:
            # Clasificación binaria (sigmoid)
            classes = (preds > 0.5).astype(int).flatten()
        else:
            # Multiclase (softmax) - ej: Iris con 3 clases
            classes = np.argmax(preds, axis=1)
        return classes.tolist()
    return preds.tolist()
