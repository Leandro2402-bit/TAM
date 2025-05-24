import streamlit as st

# ===================== TÍTULO Y DESCRIPCIÓN =====================

st.set_page_config(page_title="Predicción de Precio de Viviendas - Ames", layout="centered")

st.title("🏡 Predicción de Precio de Viviendas - AmesHousing")

st.markdown("""
Este dashboard presenta un análisis del conjunto de datos *AmesHousing*, utilizado para desarrollar modelos de regresión que predicen el precio de venta de una vivienda.

*Objetivos:*
- Explorar y procesar los datos.
- Comparar el rendimiento de diferentes modelos de regresión.
- Usar un modelo entrenado para realizar predicciones interactivas.

Los tres modelos con mejor rendimiento fueron:
- 🌲 *Random Forest Regressor*
- 🧮 *Kernel Ridge Regressor*
- 🌐 *Gaussian Process Regressor*
""")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== CARGA DE DATOS =====================
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/Leandro2402-bit/TAM/main/AmesHousing.csv"
    return pd.read_csv(url)

df = cargar_datos()

# ===================== VISUALIZACIONES =====================
st.subheader("📊 Análisis Exploratorio de Datos")

# Histograma de SalePrice
st.markdown("### Distribución del Precio de Venta")
fig1, ax1 = plt.subplots()
sns.histplot(df['SalePrice'], bins=40, kde=True, ax=ax1)
st.pyplot(fig1)

# Mapa de calor de correlación (solo numéricas)
st.markdown("### Mapa de Correlaciones")
num_df = df.select_dtypes(include=['int64', 'float64'])
corr = num_df.corr(numeric_only=True)
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5, ax=ax2)
st.pyplot(fig2)

# Top 10 variables más correlacionadas con SalePrice
st.markdown("### Variables Más Correlacionadas con el Precio")
top_corr = corr['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False).head(10)
st.bar_chart(top_corr)

# ===================== COMPARACIÓN DE MODELOS =====================

# Importar pandas para crear tablas
import pandas as pd

# Mostrar título de la sección
st.header("📊 Comparación de Modelos de Regresión")

# Crear un diccionario con las métricas de cada modelo (extraídas de tus tablas comparativas)
metricas_modelos = {
    "Modelo": ["Random Forest", "Kernel Ridge", "Gaussian Process"],
    "MAE": [15112.41, 16985.84, 16312.68],          # Error absoluto medio
    "RMSE": [23479.05, 24523.56, 24015.44],         # Raíz del error cuadrático medio
    "R²": [0.9292, 0.9221, 0.9256]                  # Coeficiente de determinación
}

# Convertir el diccionario en un DataFrame de pandas
df_metricas = pd.DataFrame(metricas_modelos)

# Mostrar la tabla de métricas en el dashboard
st.dataframe(df_metricas.style.format({
    "MAE": "{:,.2f}",
    "RMSE": "{:,.2f}",
    "R²": "{:.4f}"
}))

# ===================== MÉTRICAS DE EVALUACIÓN =====================

# Subtítulo para la nueva sección
st.subheader("📏 ¿Qué significan las métricas de evaluación?")

# Texto explicativo con Markdown
st.markdown("""
Para evaluar la calidad de los modelos de regresión, se usan estas tres métricas principales:

### 🔢 MAE – Error Absoluto Medio (Mean Absolute Error)
- Mide el *promedio de los errores absolutos* entre los precios reales y los predichos.
- *Fácil de interpretar*: un MAE de 15,000 indica un error promedio de $15,000.
- *No penaliza demasiado los errores grandes*.

### 🔢 RMSE – Raíz del Error Cuadrático Medio (Root Mean Squared Error)
- Calcula la *raíz cuadrada del promedio de los errores al cuadrado*.
- Penaliza más los *errores grandes* que el MAE.
- Si el RMSE es 23,000, en promedio el error es de unos $23,000, con énfasis en errores grandes.

### 📈 R² – Coeficiente de Determinación
- Mide cuánta *proporción de la variación del precio* puede explicar el modelo.
- R² = 1.0 → Predicción perfecta.
- R² = 0.0 → No es mejor que predecir el promedio.
- R² < 0 → El modelo es peor que adivinar el valor medio.

---
Estas métricas permiten entender si el modelo predice bien y en qué magnitud se equivoca.
""")

# ===================== Prediccion interactiva =====================
import streamlit as st
st.set_page_config(page_title="Predicción AmesHousing", layout="wide")

import pandas as pd
import numpy as np
import joblib
import os
import gdown

st.title("🏠 Predicción de precios de vivienda - AmesHousing")

# --- Enlaces de Drive ---
url_rf = "https://drive.google.com/uc?id=1tDd35bq8W_MoL5UabRR29esliSANYw35"
url_kr = "https://drive.google.com/uc?id=1CVDu6oJxWS112a1MCn9vDcWwBVwLL8Nm"

# --- Descargar modelos si no existen ---
if not os.path.exists("Random_Forest.pkl"):
    with st.spinner("Descargando modelo Random Forest..."):
        gdown.download(url_rf, "Random_Forest.pkl", quiet=False)

if not os.path.exists("Kernel_Ridge.pkl"):
    with st.spinner("Descargando modelo Kernel Ridge..."):
        gdown.download(url_kr, "Kernel_Ridge.pkl", quiet=False)

# --- Cargar modelos ---
@st.cache_resource
def cargar_modelos():
    rf = joblib.load("Random_Forest.pkl")
    kr = joblib.load("Kernel_Ridge.pkl")
    return {"Random Forest": rf, "Kernel Ridge": kr}

modelos = cargar_modelos()

# --- Selección de modelo ---
modelo_nombre = st.selectbox("Selecciona el modelo para predecir:", list(modelos.keys()))
modelo = modelos[modelo_nombre]

# --- Columnas necesarias ---
columnas_esperadas = modelo.named_steps["preprocessor"].transformers_[0][2] + \
                     modelo.named_steps["preprocessor"].transformers_[1][2]

st.markdown("### 📝 Ingrese los datos de la vivienda")

input_data = {}

for columna in columnas_esperadas:
    if columna.lower().startswith("year") or columna.lower().endswith("yr") or "flr" in columna.lower():
        valor = st.number_input(f"{columna}", step=1)
    elif "area" in columna.lower() or "sf" in columna.lower() or "frontage" in columna.lower() or "porch" in columna.lower():
        valor = st.number_input(f"{columna}", step=1.0)
    elif "qual" in columna.lower() or "cond" in columna.lower() or "bath" in columna.lower() or "room" in columna.lower():
        valor = st.number_input(f"{columna}", step=1)
    elif "type" in columna.lower() or "style" in columna.lower() or "zone" in columna.lower():
        valor = st.text_input(f"{columna}")
    elif columna.lower() == "central air":
        valor = st.selectbox(f"{columna}", ["Y", "N"])
    else:
        valor = st.text_input(f"{columna}")
    input_data[columna] = valor

if st.button("Predecir precio"):
    try:
        df_input = pd.DataFrame([input_data])
        pred = modelo.predict(df_input)[0]
        st.success(f"💰 El precio estimado de la vivienda es: ${pred:,.2f}")
    except Exception as e:
        st.error(f"❌ Error al predecir: {e}")
