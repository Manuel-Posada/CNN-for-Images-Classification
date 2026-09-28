import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"

st.title("Demo Microservicio mostrado a JU / PyTorch")

# Pestañas principales (solo predicción; el entrenamiento se hace en Jupyter)
tab_tabular, tab_imagen = st.tabs(["Tabular", "Imágenes"])

# -------------------------------
# Pestaña: Tabular
# -------------------------------
with tab_tabular:
    st.subheader("Predicción de un registro")
    st.write("Introduce las medidas de la flor (cm):")

    framework = st.selectbox(
        "Framework", ["tensorflow", "pytorch"], key="framework_tabular"
    )

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input(
            "Largo del sépalo (cm)", min_value=0.0, value=5.1, step=0.1
        )
        sepal_width = st.number_input(
            "Ancho del sépalo (cm)", min_value=0.0, value=3.5, step=0.1
        )
    with col2:
        petal_length = st.number_input(
            "Largo del pétalo (cm)", min_value=0.0, value=1.4, step=0.1
        )
        petal_width = st.number_input(
            "Ancho del pétalo (cm)", min_value=0.0, value=0.2, step=0.1
        )

    if st.button("Predecir", key="predecir_tabular"):
        data = {
            "data_type": "tabular",
            "framework": framework,
            "sepal_length": str(sepal_length),
            "sepal_width": str(sepal_width),
            "petal_length": str(petal_length),
            "petal_width": str(petal_width),
        }

        res = requests.post(f"{FASTAPI_URL}/predict/", data=data)

        st.write("Status:", res.status_code)
        st.write("Raw response:", res.text)

        if res.ok:
            payload = res.json()
            prediction = payload.get("prediction", [None])[0]
            species = ["Iris setosa", "Iris versicolor", "Iris virginica"]
            label = (
                species[prediction]
                if isinstance(prediction, int) and 0 <= prediction < len(species)
                else prediction
            )
            st.success(f"Predicción: {label} (clase {prediction})")
            st.json(payload)
        else:
            st.error("Error en la predicción")

# -------------------------------
# Pestaña: Imágenes
# -------------------------------
with tab_imagen:
    st.subheader("Predicción de imágenes")

    framework = st.selectbox(
        "Framework", ["tensorflow", "pytorch"], key="framework_imagen"
    )

    file = st.file_uploader("Sube imagen", type=["png", "jpg", "jpeg"])
    if file is not None:
        st.image(file, caption="Imagen subida", use_container_width=True)

    if file and st.button("Predecir", key="predecir_imagen"):
        data = {"data_type": "image", "framework": framework}
        files = {"file": (file.name, file.getvalue())}
        with st.spinner("Prediciendo..."):
            r = requests.post(f"{FASTAPI_URL}/predict/", data=data, files=files)
        st.write("Status:", r.status_code)
        st.write("Raw response:", r.text)
        if r.ok:
            st.json(r.json())
        else:
            st.error("Error en la predicción")
