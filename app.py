import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# Enlace de Google Drive con el modelo (reemplaza con tu ID de archivo)
ID_MODELO = "1121-HYnvXZQFycx9rYGogBN1CMffHHoXmuX"
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

    # Redimensionar la imagen a las dimensiones que el modelo espera (obtenemos el tamaño de entrada del modelo)
    modelo = cargar_modelo()
    input_shape = modelo.input_shape  # Verificar la forma de la entrada esperada por el modelo
    print(f"Forma de entrada esperada por el modelo: {input_shape}")

    # Asumiendo que el modelo espera imágenes de entrada de 224x224, redimensionamos
    imagen = imagen.resize((input_shape[1], input_shape[2]))  # Ajustamos las dimensiones a las del modelo

    # Convertir la imagen a un array numpy y normalizar (valores entre 0 y 1)
    imagen_array = np.array(imagen, dtype=np.float32) / 255.0  # Normalización a [0, 1]

    # Verificar la forma de la imagen antes de pasarla al modelo
    print(f"Forma de la imagen (sin expandir dimensiones): {imagen_array.shape}")

    # Asegurarse de que la imagen tenga la forma correcta (1, 224, 224, 3) por ejemplo
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir la dimensión de batch: (1, 224, 224, 3)

    # Verificar la forma de la imagen antes de pasársela al modelo
    print(f"Forma de la imagen (con dimensión de batch): {imagen_array.shape}")

    # Hacer la predicción
    try:
        prediccion = modelo.predict(imagen_array)
        clase_predicha = np.argmax(prediccion)  # Obtener la clase con la probabilidad más alta

        # Mostrar el resultado
        st.write(f"El modelo predice que estás mostrando el número: **{clases[clase_predicha]}** ✋")
    except Exception as e:
        st.error(f"Ha ocurrido un error durante la predicción: {e}")

