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

# ================== CARGA DEL MODELO RANDOM FOREST DESDE GOOGLE DRIVE ==================
import os                 # Para verificar si el archivo ya existe
import gdown              # Para descargar desde Google Drive
import joblib             # Para cargar el modelo .pkl
import streamlit as st    # Para mensajes visuales en la app

# ID de tu archivo en Google Drive
file_id = '1y992YhEfjkipa8tI0A-MMxegvPaEBHZR'

# Construimos la URL de descarga directa desde Google Drive
url = f'https://drive.google.com/uc?id={file_id}'

# Nombre del archivo que tendrá localmente en la app
output_path = 'random_forest_model.pkl'

# Si el archivo aún no existe, lo descargamos
if not os.path.exists(output_path):
    with st.spinner('🔽 Descargando modelo Random Forest desde Google Drive...'):
        gdown.download(url, output_path, quiet=False)  # Descarga el archivo

# Cargamos el modelo una vez descargado
model_rf = joblib.load(output_path)

# Mostramos mensaje de éxito en la interfaz de Streamlit
st.success("✅ Modelo Random Forest cargado exitosamente.")

# ================== CARGA DEL MODELO RANDOM FOREST DESDE GOOGLE DRIVE ==================
import os
import gdown
import joblib

# ID de tu archivo en Google Drive (ajusta esto con tu ID real)
file_id = '1ABCdEFghIJklmnOPqrSTUvWxyz'  # 👈 Reemplaza esto con tu ID real

# Construimos la URL directa de descarga
url = f'https://drive.google.com/uc?id={file_id}'
output_path = 'random_forest_model.pkl'

# Si el archivo aún no está en el entorno, lo descargamos
if not os.path.exists(output_path):
    with st.spinner('Descargando modelo Random Forest desde Google Drive...'):
        gdown.download(url, output_path, quiet=False)

# Cargamos el modelo ya descargado
model_rf = joblib.load(output_path)

st.success("✅ Modelo Random Forest cargado exitosamente.")

# ======================= PREDICCIÓN PERSONALIZADA =======================

import pandas as pd

st.header("🔮 Predicción personalizada del precio de una vivienda")

# Mostramos solo si el modelo se cargó bien
if model is not None:
    st.markdown("Ingresa las características de la vivienda:")

    # ====== Entradas interactivas ======
    overall_qual = st.slider("Calidad general de la casa (1 - 10)", 1, 10, 5)
    gr_liv_area = st.slider("Área habitable sobre nivel (pies²)", 500, 4000, 1500)
    garage_cars = st.slider("Capacidad del garaje (número de autos)", 0, 4, 2)
    total_bsmt_sf = st.slider("Área total del sótano (pies²)", 0, 3000, 800)
    first_flr_sf = st.slider("Área del primer piso (pies²)", 300, 3000, 1000)
    year_built = st.slider("Año de construcción", 1870, 2010, 1980)
    full_bath = st.slider("Número de baños completos", 0, 4, 2)
    tot_rms_abv_grd = st.slider("Total de habitaciones sobre nivel", 2, 14, 6)

    # ====== Botón para predecir ======
    if st.button("Predecir precio"):
        # Creamos un DataFrame con las entradas del usuario
        user_input = pd.DataFrame([{
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'GarageCars': garage_cars,
            'TotalBsmtSF': total_bsmt_sf,
            '1stFlrSF': first_flr_sf,
            'YearBuilt': year_built,
            'FullBath': full_bath,
            'TotRmsAbvGrd': tot_rms_abv_grd
        }])

        # Usamos el modelo para predecir
        pred_price = model.predict(user_input)[0]

        # Mostramos resultado
        st.success(f"🏷️ Precio estimado de la vivienda: **${int(pred_price):,} USD**")
else:
    st.warning("El modelo no está disponible. Asegúrate de que se haya cargado correctamente.")

