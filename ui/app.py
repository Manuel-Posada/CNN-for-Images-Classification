"""Interfaz Streamlit del clasificador de imagenes de animales (PyTorch).

Estilo conservado: tema por defecto de Streamlit, sin CSS/HTML inyectado, textos
en espanol y misma estructura de widgets (st.title, st.tabs, st.selectbox,
st.columns, st.file_uploader, st.image, st.spinner, st.success/st.error).

Toda la informacion se consume de la API FastAPI; la UI no recalcula metricas.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Acceso a la API (cacheado). Las funciones lanzan excepcion si fallan para que
# los errores no se cacheen.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_models() -> list[dict]:
    """GET /models -> resumen de las arquitecturas disponibles."""
    r = requests.get(f"{API_URL}/models", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def get_model_metrics(arch: str) -> dict:
    """GET /models/{arch}/metrics -> metricas completas del modelo."""
    r = requests.get(f"{API_URL}/models/{arch}/metrics", timeout=10)
    if r.status_code == 404:
        raise ValueError(f"Modelo no encontrado: '{arch}'")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Arranque: comprobar que el backend esta levantado
# ---------------------------------------------------------------------------
try:
    MODELS = get_models()
except Exception:
    st.error(
        "No se pudo conectar con el backend. Levantalo primero:\n\n"
        "    uvicorn app.main:app --reload"
    )
    st.stop()

INCLUDED = [m for m in MODELS if m.get("included", True)]
EXCLUDED = [m for m in MODELS if not m.get("included", True)]
ARCH_OPTIONS = [m["arch"] for m in INCLUDED]


def arch_label(a: str) -> str:
    return next(m["label"] for m in MODELS if m["arch"] == a)


st.title("Clasificador de animales — CNN / ResNet18 / AlexNet")

# ---------------------------------------------------------------------------
# Selector de modelo (comun a toda la app) + resumen
# ---------------------------------------------------------------------------
arch = st.selectbox("Modelo", options=ARCH_OPTIONS, format_func=arch_label)
label = arch_label(arch)
metrics = get_model_metrics(arch)
test = metrics["test"]

st.subheader(f"Resumen: {label}")
row1 = st.columns(5)
row1[0].metric("Accuracy", f'{test["accuracy"]:.3f}')
row1[1].metric("Balanced accuracy", f'{test["balanced_accuracy"]:.3f}')
row1[2].metric("Macro-F1", f'{test["macro_f1"]:.3f}')
row1[3].metric("ROC AUC macro", f'{test["roc_auc_macro"]:.3f}')
row1[4].metric("Tamaño de entrada", f'{metrics["img_size"]}×{metrics["img_size"]}')
row2 = st.columns(5)
row2[0].metric("Parámetros totales", f'{metrics["params_total"]:,}')
row2[1].metric("Parámetros entrenables", f'{metrics["params_trainable"]:,}')
row2[2].metric("Épocas", int(metrics["epochs"]))
row2[3].metric("Tiempo de entrenamiento", f'{metrics["train_time_s"]:.1f} s')
row2[4].metric("ms/imagen (evaluación)", f'{metrics["ms_per_image"]:.2f}')

if EXCLUDED:
    nombres = ", ".join(m["label"] for m in EXCLUDED)
    st.info(
        f"Los modelos **{nombres}** son más pesados, por lo que no se incluyen en esta "
        "versión de la app; sus métricas sí están disponibles y puedes revisar sus "
        "comparaciones en la pestaña «Comparar modelos»."
    )

tab_clasif, tab_metricas, tab_compare = st.tabs(
    ["Clasificar imagen", "Métricas del modelo", "Comparar modelos"]
)


# ---------------------------------------------------------------------------
# PESTAÑA: Clasificar imagen
# ---------------------------------------------------------------------------
def mostrar_resultado_predict(result: dict, model_arch: str, model_label: str) -> None:
    """Renderiza el resultado real de /predict/ (se reutiliza en los tests)."""
    predicted = result["predicted_class"]
    conf = result["confidence"]

    st.success(f"Predicción: **{predicted}** — confianza {conf * 100:.1f} %")
    if conf < 0.5:
        st.info("Confianza baja (< 50 %): la predicción puede no ser fiable.")

    st.write("**Top 3 categorías**")
    top_k = result["top_k"]
    df_top = pd.DataFrame(
        {"Probabilidad": [p["prob"] for p in top_k]},
        index=[p["class"] for p in top_k],
    )
    st.bar_chart(df_top)

    col_ms, col_size = st.columns(2)
    col_ms.metric("Tiempo de inferencia", f'{result["inference_ms"]:.1f} ms')
    col_size.metric(
        "Redimensionado (img_size)",
        f'{result["img_size"]}×{result["img_size"]} px',
    )

    pm = get_model_metrics(model_arch)
    per = pm["per_class"].get(predicted)
    if per:
        st.write(f"**Rendimiento histórico de la clase {predicted} con {model_label}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precisión", f'{per["precision"]:.3f}')
        c2.metric("Recall", f'{per["recall"]:.3f}')
        c3.metric("F1", f'{per["f1"]:.3f}')
        c4.metric("AUC", f'{per["auc"]:.3f}')
        c5.metric("Soporte", int(per["support"]))

    with st.expander("Respuesta cruda de la API"):
        st.json(result)


with tab_clasif:
    st.subheader("Predicción de imagen")
    file = st.file_uploader("Sube imagen", type=["png", "jpg", "jpeg", "webp"])
    if file is not None:
        st.image(file, caption="Imagen subida", width="stretch")

    if file and st.button("Predecir", key="predecir_imagen"):
        data = {"arch": arch, "topk": 3}
        files = {"file": (file.name, file.getvalue())}
        with st.spinner("Prediciendo..."):
            try:
                r = requests.post(f"{API_URL}/predict/", data=data, files=files, timeout=60)
            except requests.RequestException:
                st.error(f"No se pudo contactar con el backend en {API_URL}")
                r = None
        if r is not None and r.ok:
            mostrar_resultado_predict(r.json(), arch, label)
        elif r is not None:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.error(f"Error del backend: {detail}")


# ---------------------------------------------------------------------------
# PESTAÑA: Métricas del modelo
# ---------------------------------------------------------------------------
with tab_metricas:
    classes = metrics["classes"]

    st.write("**Curvas de entrenamiento** (eje x: época)")
    his = metrics["history"]
    n = len(his["train_loss"])
    df_loss = pd.DataFrame(
        {"Entrenamiento": his["train_loss"], "Validación": his["val_loss"]},
        index=range(1, n + 1),
    )
    df_acc = pd.DataFrame(
        {"Entrenamiento": his["train_acc"], "Validación": his["val_acc"]},
        index=range(1, n + 1),
    )
    col_loss, col_acc = st.columns(2)
    with col_loss:
        st.write("Pérdida")
        st.line_chart(df_loss)
    with col_acc:
        st.write("Exactitud")
        st.line_chart(df_acc)

    st.write("**Matriz de confusión** (normalizada por fila)")
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Ver conteos"):
        st.dataframe(pd.DataFrame(cm.astype(int), index=classes, columns=classes))

    st.write("**Curvas ROC**")
    roc = metrics["roc"]
    sel_clases = st.multiselect("Clases a mostrar", options=classes, default=[])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        roc["micro"]["fpr"], roc["micro"]["tpr"], lw=2,
        label=f'micro (AUC={roc["micro"]["auc"]:.3f})',
    )
    ax.plot(
        roc["macro"]["fpr"], roc["macro"]["tpr"], lw=2,
        label=f'macro (AUC={roc["macro"]["auc"]:.3f})',
    )
    for cls in sel_clases:
        curva = metrics["roc_per_class"].get(cls)
        auc = metrics["per_class"].get(cls, {}).get("auc")
        if curva:
            lbl = f'{cls} (AUC={auc:.3f})' if auc is not None else cls
            ax.plot(curva["fpr"], curva["tpr"], lw=1, alpha=0.7, label=lbl)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8, loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

    st.write("**Métricas por clase**")
    filas = [
        {
            "Clase": cls,
            "Precision": p["precision"],
            "Recall": p["recall"],
            "F1": p["f1"],
            "AUC": p["auc"],
            "Soporte": p["support"],
        }
        for cls, p in metrics["per_class"].items()
    ]
    st.dataframe(pd.DataFrame(filas))


# ---------------------------------------------------------------------------
# PESTAÑA: Comparar modelos (no depende del selector)
# ---------------------------------------------------------------------------
with tab_compare:
    all_metrics = {m["arch"]: get_model_metrics(m["arch"]) for m in MODELS}

    st.write("**Comparativa lado a lado**")
    filas_cmp = []
    for m in MODELS:
        mm = all_metrics[m["arch"]]
        t = mm["test"]
        filas_cmp.append(
            {
                "Modelo": mm["label"],
                "Accuracy": t["accuracy"],
                "Balanced accuracy": t["balanced_accuracy"],
                "Macro-F1": t["macro_f1"],
                "ROC AUC macro": t["roc_auc_macro"],
                "ROC AUC micro": t["roc_auc_micro"],
                "Loss test": t["loss"],
                "Tamaño de entrada": f'{mm["img_size"]}×{mm["img_size"]}',
                "Parámetros totales": mm["params_total"],
                "Parámetros entrenables": mm["params_trainable"],
                "Épocas": mm["epochs"],
                "Entrenamiento (s)": round(mm["train_time_s"], 1),
                "ms/imagen": round(mm["ms_per_image"], 2),
            }
        )
    df_cmp = pd.DataFrame(filas_cmp).set_index("Modelo")
    st.dataframe(df_cmp)

    st.write("**Métricas principales por modelo**")
    df_plot = df_cmp[["Accuracy", "Balanced accuracy", "Macro-F1", "ROC AUC macro"]]
    st.bar_chart(df_plot)

    st.write("**ROC macro por modelo**")
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in MODELS:
        roc_macro = all_metrics[m["arch"]]["roc"]["macro"]
        ax.plot(
            roc_macro["fpr"], roc_macro["tpr"], lw=2,
            label=f'{m["label"]} (AUC={roc_macro["auc"]:.3f})',
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8, loc="lower right")
    st.pyplot(fig)
    plt.close(fig)