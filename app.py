import streamlit as st
st.set_page_config(page_title="Predicción de Precio de Viviendas - Ames", layout="centered")

# ===================== TÍTULO Y DESCRIPCIÓN =====================

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

import pandas as pd
import numpy as np
import joblib
import gdown
import os

# --- URLS de los modelos en Drive ---
urls = {
    "Random Forest": "https://drive.google.com/uc?id=1tDd35bq8W_MoL5UabRR29esliSANYw35",
    "Kernel Ridge": "https://drive.google.com/uc?id=1CVDu6oJxWS112a1MCn9vDcWwBVwLL8Nm"
}

# --- Cargar modelo si no está en disco ---
def load_model(model_name, url):
    filename = f"{model_name.replace(' ', '_')}.pkl"
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)
    return joblib.load(filename)

# --- Selección de modelo ---
st.title("🏠 Predicción de Precios de Vivienda - AmesHousing")
modelo_seleccionado = st.selectbox("🔍 Selecciona un modelo para predecir:", list(urls.keys()))
modelo = load_model(modelo_seleccionado, urls[modelo_seleccionado])

# --- Lista de campos importantes a mostrar (puedes ajustarlos tú mismo) ---
campos_clave = {
    "Overall Qual": {"tipo": "slider", "min": 1, "max": 10, "recomendacion": "Calidad general del material y acabado"},
    "Gr Liv Area": {"tipo": "number", "recomendacion": "Área habitable sobre el nivel del suelo (en pies²)"},
    "Garage Cars": {"tipo": "slider", "min": 0, "max": 4, "recomendacion": "Número de carros que caben en el garaje"},
    "Total Bsmt SF": {"tipo": "number", "recomendacion": "Área total del sótano (en pies²)"},
    "Year Built": {"tipo": "number", "recomendacion": "Año de construcción de la casa"},
    "Full Bath": {"tipo": "slider", "min": 0, "max": 4, "recomendacion": "Número de baños completos"},
    "1st Flr SF": {"tipo": "number", "recomendacion": "Área del primer piso (en pies²)"},
    "Garage Area": {"tipo": "number", "recomendacion": "Área del garaje (en pies²)"},
    "Kitchen Qual": {
        "tipo": "select", 
        "opciones": ["Ex", "Gd", "TA", "Fa", "Po"], 
        "recomendacion": "Calidad de la cocina (Ex: Excelente, Gd: Buena, TA: Típica, Fa: Regular, Po: Pobre)"
    },
    "Neighborhood": {
        "tipo": "select",
        "opciones": ["NridgHt", "CollgCr", "Crawfor", "Somerst", "OldTown", "Mitchel", "NWAmes", "Sawyer"],
        "recomendacion": "Vecindario donde está ubicada la casa"
    }
}

# --- Recolección de inputs amigables ---
st.markdown("### 📝 Ingresa los datos principales de la vivienda")
input_data = {}
for campo, config in campos_clave.items():
    st.markdown(f"**{campo}** — _{config['recomendacion']}_")
    
    if config["tipo"] == "slider":
        input_data[campo] = st.slider("", min_value=config["min"], max_value=config["max"], value=config["min"])
    elif config["tipo"] == "number":
        input_data[campo] = st.number_input("", min_value=0, step=10)
    elif config["tipo"] == "select":
        input_data[campo] = st.selectbox("", config["opciones"])
    st.markdown("---")

# --- Botón de predicción ---
if st.button("🔮 Predecir Precio"):
    try:
        df = pd.DataFrame([input_data])
        # Completamos el resto de columnas que espera el modelo con ceros o valores nulos
        for col in modelo.feature_names_in_:
            if col not in df.columns:
                df[col] = 0  # O puedes usar np.nan dependiendo del modelo
        df = df[modelo.feature_names_in_]  # Asegurar el orden correcto

        pred = modelo.predict(df)[0]
        st.success(f"💵 Precio estimado: ${pred:,.2f}")
    except Exception as e:
        st.error(f"❌ Error al predecir: {e}")

