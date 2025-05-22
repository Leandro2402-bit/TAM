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
