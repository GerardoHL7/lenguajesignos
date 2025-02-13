import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# Enlace de Google Drive con el modelo (reemplaza con tu ID de archivo)
ID_MODELO = "1-HYnvXZQFycx9rYGogBN1CMffHHoXmuX"
URL_MODELO = f"https://drive.google.com/uc?id={ID_MODELO}"
RUTA_MODELO = "mimodelo.h5"

# Descargar el modelo si no existe
if not os.path.exists(RUTA_MODELO):
    with st.spinner("Descargando modelo... Esto puede tardar un momento ⏳"):
        gdown.download(URL_MODELO, RUTA_MODELO, quiet=False)

# Cargar el modelo solo cuando sea necesario
@st.cache_resource
def cargar_modelo():
    modelo = load_model(RUTA_MODELO)
    return modelo

# Diccionario de clases (ajusta según tu dataset)
clases = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}

# Interfaz de Streamlit
st.title("Reconocimiento de Lenguaje de Signos 🤟")
st.write("Sube una imagen de tu mano mostrando un número en lenguaje de signos.")

imagen_subida = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    # Cargar y mostrar la imagen
    imagen = Image.open(imagen_subida)
    imagen = imagen.convert("RGB")  # Asegurarse de que sea RGB
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    # Redimensionar la imagen a 64x64 (ajustar según el tamaño de entrada de tu modelo)
    imagen = imagen.resize((64, 64))  # Cambia (64, 64) según lo que necesite tu modelo

    # Convertir la imagen a un array numpy y normalizar (valores entre 0 y 1)
    imagen_array = np.array(imagen) / 255.0  # Normalización a [0, 1]

    # Verificar la forma de la imagen antes de pasarla al modelo
    print(f"Forma de la imagen: {imagen_array.shape}")

    # Asegurarse de que la imagen tenga la forma correcta (1, 64, 64, 3)
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir la dimensión de batch: (1, 64, 64, 3)

    # Cargar el modelo solo cuando se sube una imagen
    if 'modelo' not in st.session_state:
        st.session_state.modelo = cargar_modelo()

    modelo = st.session_state.modelo

    # Hacer la predicción
    prediccion = modelo.predict(imagen_array)
    clase_predicha = np.argmax(prediccion)  # Obtener la clase con la probabilidad más alta

    # Mostrar el resultado
    st.write(f"El modelo predice que estás mostrando el número: **{clases[clase_predicha]}** ✋")

