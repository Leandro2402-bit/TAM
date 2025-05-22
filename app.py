
import streamlit as st
import streamlit as st

# ===================== TÍTULO Y DESCRIPCIÓN =====================

st.set_page_config(page_title="Predicción de Precio de Viviendas - Ames", layout="centered")

st.title("🏡 Predicción de Precio de Viviendas - AmesHousing")

st.markdown("""
Este dashboard presenta un análisis del conjunto de datos **AmesHousing**, utilizado para desarrollar modelos de regresión que predicen el precio de venta de una vivienda.

**Objetivos:**
- Explorar y procesar los datos.
- Comparar el rendimiento de diferentes modelos de regresión.
- Usar un modelo entrenado para realizar predicciones interactivas.

Los tres modelos con mejor rendimiento fueron:
- 🌲 **Random Forest Regressor**
- 🧮 **Kernel Ridge Regressor**
- 🌐 **Gaussian Process Regressor**
""")

# Título
st.title("🏠 Dashboard de Predicción de Precios de Vivienda")

# Subtítulo
st.subheader("Proyecto de Machine Learning con el dataset Ames Housing")

# Texto de bienvenida
st.write("""
Bienvenido al dashboard interactivo del proyecto.
Aquí podrás:
- Explorar los datos del conjunto Ames Housing.
- Visualizar estadísticas y gráficos.
- Consultar los mejores modelos entrenados.
- Realizar predicciones personalizadas.
""")
