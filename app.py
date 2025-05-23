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
- Mide el **promedio de los errores absolutos** entre los precios reales y los predichos.
- **Fácil de interpretar**: un MAE de `15,000` indica un error promedio de $15,000.
- **No penaliza demasiado los errores grandes**.

### 🔢 RMSE – Raíz del Error Cuadrático Medio (Root Mean Squared Error)
- Calcula la **raíz cuadrada del promedio de los errores al cuadrado**.
- Penaliza más los **errores grandes** que el MAE.
- Si el RMSE es `23,000`, en promedio el error es de unos $23,000, con énfasis en errores grandes.

### 📈 R² – Coeficiente de Determinación
- Mide cuánta **proporción de la variación del precio** puede explicar el modelo.
- R² = `1.0` → Predicción perfecta.
- R² = `0.0` → No es mejor que predecir el promedio.
- R² < 0 → El modelo es peor que adivinar el valor medio.

---
Estas métricas permiten entender si el modelo predice bien y en qué magnitud se equivoca.
""")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

# Título
st.title("🏡 Predicción de Precios de Vivienda - Dashboard")

# URLs públicas de Google Drive
model_urls = {
    "Random Forest": "https://drive.google.com/uc?id=1y992YhEfjkipa8tI0A-MMxegvPaEBHZR",
    "Kernel Ridge": "https://drive.google.com/uc?id=1txUpSg0meC7-o6ABkQAYCaSOCMXo2Sm0"
}

# Cargar modelos
loaded_models = {}

for name, url in model_urls.items():
    filename = f"{name.replace(' ', '_')}.pkl"
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)
    try:
        loaded_models[name] = joblib.load(filename)
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo {name}.")
        st.exception(e)

# Mostrar selectbox para elegir el modelo
model_name = st.selectbox("Selecciona el modelo para predecir:", list(loaded_models.keys()))

# Mostrar las columnas esperadas por el modelo
if st.checkbox("Mostrar columnas requeridas por el modelo"):
    try:
        st.write("📦 Columnas requeridas por el modelo:")
        st.write(list(loaded_models[model_name].feature_names_in_))
    except Exception as e:
        st.error("⚠️ No se pudieron obtener las columnas del modelo.")
        st.exception(e)

# Subir CSV
st.subheader("📤 Sube un archivo CSV con los datos:")
uploaded_file = st.file_uploader("Carga tu archivo .csv", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Vista previa de tus datos:")
        st.dataframe(df.head())

        # Filtrar las columnas necesarias
        model = loaded_models[model_name]
        try:
            X_input = df[model.feature_names_in_]
        except Exception as e:
            st.error("⚠️ Asegúrate de que el archivo contenga las columnas requeridas por el modelo.")
            st.stop()

        # Realizar predicción
        preds = model.predict(X_input)
        df["Precio Predicho"] = np.round(preds, 2)

        st.subheader("📈 Resultados de la predicción:")
        st.dataframe(df[["Precio Predicho"]])

        # Descargar resultados
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar resultados", data=csv, file_name="predicciones.csv", mime='text/csv')

    except Exception as e:
        st.error("❌ Error al procesar el archivo.")
        st.exception(e)



    
