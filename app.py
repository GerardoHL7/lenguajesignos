import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# Enlace de Google Drive con el modelo (cambia el ID por el tuyo)
ID_MODELO = "1-HYnvXZQFycx9rYGogBN1CMffHHoXmuX"
URL_MODELO = f"https://drive.google.com/uc?id={ID_MODELO}"
RUTA_MODELO = "mimodelo.h5"

# Descargar el modelo si no existe
if not os.path.exists(RUTA_MODELO):
    with st.spinner("Descargando modelo... Esto puede tardar un momento ⏳"):
        gdown.download(URL_MODELO, RUTA_MODELO, quiet=False)

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return load_model(RUTA_MODELO)

modelo = cargar_modelo()

# Diccionario de clases (ajusta según tu dataset)
clases = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}

# Interfaz de Streamlit
st.title("Reconocimiento de Lenguaje de Signos 🤟")
st.write("Sube una imagen de tu mano mostrando un número en lenguaje de signos.")

imagen_subida = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    # Preprocesar la imagen
    imagen = imagen.resize((64, 64))
    imagen_array = np.array(imagen) / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)

    # Hacer la predicción
    prediccion = modelo.predict(imagen_array)
    clase_predicha = np.argmax(prediccion)

    # Mostrar resultado
    st.write(f"El modelo predice que estás mostrando el número: **{clases[clase_predicha]}** ✋")
