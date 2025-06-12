import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import gdown
import os

# ===================== CONFIG =====================
st.set_page_config(page_title="Predicción de Precio de Viviendas - Ames", layout="centered")

# ===================== CARGA DE DATOS =====================
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/Leandro2402-bit/TAM/main/AmesHousing.csv"
    return pd.read_csv(url)

df_ames = cargar_datos()

# ===================== VISUALIZACIONES =====================
st.title("🏡 Predicción de Precio de Viviendas - AmesHousing")
st.subheader("📊 Análisis Exploratorio")

# Histograma
fig1, ax1 = plt.subplots()
sns.histplot(df_ames['SalePrice'], bins=40, kde=True, ax=ax1)
st.pyplot(fig1)

# Heatmap de correlación
num_df = df_ames.select_dtypes(include=['int64', 'float64'])
corr = num_df.corr(numeric_only=True)
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, ax=ax2)
st.pyplot(fig2)

st.markdown("**Top 10 variables más correlacionadas con `SalePrice`**")
top_corr = corr['SalePrice'].drop('SalePrice').abs().nlargest(10)
st.bar_chart(top_corr)

# ===================== COMPARACIÓN DE MODELOS =====================
st.header("📊 Comparación de Modelos")
metricas_modelos = {
    "Modelo": ["Random Forest", "Kernel Ridge", "Gaussian Process"],
    "MAE": [15112.41, 16985.84, 16312.68],
    "RMSE": [23479.05, 24523.56, 24015.44],
    "R²": [0.9292, 0.9221, 0.9256]
}
df_metricas = pd.DataFrame(metricas_modelos)
st.dataframe(df_metricas.style.format({"MAE":"{:,.2f}","RMSE":"{:,.2f}","R²":"{:.4f}"}))

st.subheader("📏 Interpretación de Métricas")
st.markdown("""
- **MAE**: error promedio absoluto.  
- **RMSE**: penaliza errores grandes.  
- **R²**: proporción de la variabilidad explicada (“1” = perfecta, “0” = predecir la media).  
""")

# ===================== PREDICCIÓN INTERACTIVA =====================
st.title("🏠 Predicción Interactiva")

# URLs de modelos en Drive
urls = {
    "Random Forest": "https://drive.google.com/uc?id=1tDd35bq8W_MoL5UabRR29esliSANYw35",
    "Kernel Ridge": "https://drive.google.com/uc?id=1rJqTDNebuv6fOnECRSI_jF4XrdvSr2Nj"
}

def load_model(model_name, url):
    fn = model_name.replace(" ", "_") + ".pkl"
    if not os.path.exists(fn):
        gdown.download(url, fn, quiet=False)
    return joblib.load(fn)

modelo_seleccionado = st.selectbox("🔍 Elige modelo:", list(urls.keys()))
modelo = load_model(modelo_seleccionado, urls[modelo_seleccionado])

# Definición de campos con rango válido
campos = {
    "Overall Qual": {"tipo":"slider","min":1,"max":10,"rec":"Calidad general del material"},
    "Gr Liv Area": {"tipo":"number","min":300,"max":5000,"rec":"Área habitable (pies²)"},
    "Garage Cars": {"tipo":"slider","min":0,"max":4,"rec":"Coches en garaje"},
    "Total Bsmt SF": {"tipo":"number","min":0,"max":3000,"rec":"Área sótano (pies²)"},
    "Year Built": {"tipo":"number","min":1870,"max":2025,"rec":"Año de construcción"},
    "Full Bath": {"tipo":"slider","min":0,"max":4,"rec":"Baños completos"},
    "1st Flr SF": {"tipo":"number","min":300,"max":3000,"rec":"Área primer piso (pies²)"},
    "Garage Area": {"tipo":"number","min":0,"max":1500,"rec":"Área garaje (pies²)"},
    "Kitchen Qual": {"tipo":"select","opciones":["Ex","Gd","TA","Fa","Po"],"rec":"Calidad cocina"},
    "Neighborhood": {"tipo":"select","opciones":["NridgHt","CollgCr","Crawfor","Somerst","OldTown","Mitchel","NWAmes","Sawyer"],"rec":"Vecindario"}
}

st.markdown("### 📝 Ingresa datos de la vivienda")
input_data = {}
for campo, cfg in campos.items():
    st.markdown(f"**{campo}** — _{cfg['rec']}_")
    if cfg["tipo"] == "slider":
        input_data[campo] = st.slider(campo, cfg["min"], cfg["max"], cfg["min"])
    elif cfg["tipo"] == "number":
        input_data[campo] = st.number_input(campo, cfg["min"], cfg["max"], cfg["min"], step=10)
    elif cfg["tipo"] == "select":
        input_data[campo] = st.selectbox(campo, cfg["opciones"])
    st.markdown("---")

if st.button("🔮 Predecir Precio"):
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)
    for feat in modelo.feature_names_in_:
        if feat not in df_input.columns:
            df_input[feat] = 0
    df_input = df_input[modelo.feature_names_in_]
    try:
        pred = modelo.predict(df_input)[0]
        st.success(f"💵 Precio estimado: ${pred:,.2f}")
        st.info("Este precio se basa solo en los parámetros ingresados y puede variar por otros factores.")
    except Exception as e:
        st.error(f"❌ No se pudo predecir: {e}")




