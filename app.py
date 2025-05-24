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

# ===================== PREDICCIÓN INTERACTIVA =====================

import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# Subtítulo de la sección
st.header("🎯 Predicción Interactiva de Precio de Vivienda")

# Diccionario con las URLs públicas de Google Drive para descargar los modelos
model_urls = {
    "Random Forest": 'https://drive.google.com/uc?id=1cB_Oc6Y9pz7kjb7Kd2VH79kBP7jYVASj',
    "Kernel Ridge": 'https://drive.google.com/uc?id=1NwXf6o1WXHr8JDVDlLfxh_MFxpPCjUW9'
}

# Diccionario donde se almacenarán los modelos cargados
loaded_models = {}

# Descargar y cargar los modelos
for name, url in model_urls.items():
    filename = f"{name.replace(' ', '_')}.pkl"
    try:
        if not os.path.exists(filename):
            gdown.download(url, filename, quiet=True)
        loaded_models[name] = joblib.load(filename)
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo {name}: {str(e)}")
        continue

# Verificar si se cargaron modelos correctamente
if not loaded_models:
    st.error("No se pudo cargar ningún modelo. Por favor verifica los archivos.")
    st.stop()

# Selección del modelo
model_name = st.selectbox("📌 Selecciona el modelo para predecir:", list(loaded_models.keys()))
model = loaded_models[model_name]

# Mostrar las variables necesarias para la predicción
st.markdown("### ✍️ Ingresa los datos de la vivienda")

# Inputs del usuario (asegúrate que estos nombres coincidan con los que usó el modelo)
GrLivArea = st.number_input("Área habitable sobre el suelo (GrLivArea)", min_value=300, max_value=6000, value=1500)
OverallQual = st.slider("Calidad general (OverallQual)", 1, 10, 5)
GarageCars = st.slider("Espacios en garaje (GarageCars)", 0, 5, 2)
TotalBsmtSF = st.number_input("Área total del sótano (TotalBsmtSF)", min_value=0, max_value=3000, value=800)
YearBuilt = st.number_input("Año de construcción (YearBuilt)", min_value=1870, max_value=2023, value=2000)

# Crear DataFrame con los nombres EXACTOS que espera el modelo
input_data = pd.DataFrame({
    "GrLivArea": [GrLivArea],
    "OverallQual": [OverallQual],
    "GarageCars": [GarageCars],
    "TotalBsmtSF": [TotalBsmtSF],
    "YearBuilt": [YearBuilt]
})

# Realizar predicción
if st.button("🔮 Predecir Precio"):
    try:
        # Verificar que el modelo esté cargado correctamente
        if not hasattr(model, 'predict'):
            st.error("El modelo no tiene método predict. Verifica el archivo del modelo.")
            st.stop()
            
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 Precio estimado de la vivienda: ${predicted_price:,.2f}")
        
        # Mostrar advertencia sobre la precisión
        st.info("""
        ℹ️ Esta es una estimación aproximada. 
        El precio real puede variar según otros factores no considerados en este modelo.
        """)
    except Exception as e:
        st.error(f"❌ Error al realizar la predicción: {str(e)}")
        st.write("Detalles técnicos del error:")
        st.code(str(e), language='python')

    
