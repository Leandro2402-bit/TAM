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

# ===============================================
# DESCARGA Y CARGA DEL MODELO RANDOM FOREST =====

import gdown  # Librería para descargar archivos de Google Drive
import joblib  # Librería para cargar el modelo .pkl
import os      # Librería para verificar existencia del archivo

# ID del archivo compartido en Google Drive (modelo entrenado)
file_id = "1y992YhEfjkipa8tI0A-MMxegvPaEBHZR"

# Nombre con el que se guardará el archivo descargado localmente
output_file = "random_forest_model.pkl"

# Definimos la variable del modelo como None por defecto
model = None

try:
    # Solo descarga si el archivo no existe
    if not os.path.exists(output_file):
        with st.spinner("🔽 Descargando modelo Random Forest desde Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)

    # Verificamos que el archivo se haya descargado correctamente
    if os.path.exists(output_file):
        # Cargamos el modelo desde el archivo .pkl
        model = joblib.load(output_file)
        st.success("✅ Modelo Random Forest cargado exitosamente.")
    else:
        st.error("❌ No se encontró el archivo del modelo.")
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo: {e}")

# Verificación final del modelo
if model is not None:
    st.info("📊 El modelo está listo para realizar predicciones.")
        # ==============================
    # 📝 FORMULARIO DE ENTRADA
    # ==============================

    st.subheader("🔍 Ingresa los datos de la vivienda para predecir el precio")

    # Creamos columnas para una mejor disposición en pantalla
    col1, col2 = st.columns(2)

    # Variables numéricas típicas
    with col1:
        OverallQual = st.slider("Calidad general (OverallQual)", 1, 10, 5)
        GrLivArea = st.number_input("Área habitable (GrLivArea)", min_value=100, max_value=6000, value=1500)
        GarageCars = st.slider("Número de autos en garaje (GarageCars)", 0, 4, 2)
        GarageArea = st.number_input("Área del garaje (GarageArea)", min_value=0, max_value=1500, value=500)

    with col2:
        TotalBsmtSF = st.number_input("Área total del sótano (TotalBsmtSF)", min_value=0, max_value=3000, value=800)
        FullBath = st.slider("Baños completos (FullBath)", 0, 4, 2)
        YearBuilt = st.slider("Año de construcción (YearBuilt)", 1870, 2020, 1990)
        YearRemodAdd = st.slider("Año de remodelación (YearRemodAdd)", 1950, 2020, 2005)

    # ==============================
    # 🔮 PREDICCIÓN (versión simple)
    # ==============================

    if st.button("Predecir Precio"):
        # Lista completa de columnas que espera el modelo
        columnas_modelo = [
            "Order", "PID", "MS SubClass", "MS Zoning", "Lot Frontage", "Lot Area", "Street", "Alley", "Lot Shape",
            "Land Contour", "Utilities", "Lot Config", "Land Slope", "Neighborhood", "Condition 1", "Condition 2",
            "Bldg Type", "House Style", "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add", "Roof Style",
            "Roof Matl", "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", "Mas Vnr Area", "Exter Qual", "Exter Cond",
            "Foundation", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin SF 1", "BsmtFin Type 2",
            "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "Heating", "Heating QC", "Central Air", "Electrical",
            "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath",
            "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "Kitchen Qual", "TotRms AbvGrd", "Functional", "Fireplaces",
            "Fireplace Qu", "Garage Type", "Garage Yr Blt", "Garage Finish", "Garage Cars", "Garage Area", "Garage Qual",
            "Garage Cond", "Paved Drive", "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch",
            "Screen Porch", "Pool Area", "Pool QC", "Fence", "Misc Feature", "Misc Val", "Mo Sold", "Yr Sold",
            "Sale Type", "Sale Condition"
        ]

        # Creamos un diccionario base con ceros
        datos_defecto = {col: 0 for col in columnas_modelo}

        # Reemplazamos con los valores que el usuario realmente ingresó
        datos_defecto.update({
            "Overall Qual": OverallQual,
            "Gr Liv Area": GrLivArea,
            "Garage Cars": GarageCars,
            "Garage Area": GarageArea,
            "Total Bsmt SF": TotalBsmtSF,
            "Full Bath": FullBath,
            "Year Built": YearBuilt,
            "Year Remod/Add": YearRemodAdd
        })

        # Creamos el DataFrame con los datos actualizados
        input_data = pd.DataFrame([datos_defecto])

        # Realizamos la predicción
        predicted_price = model.predict(input_data)[0]

        # Mostramos el resultado
        st.success(f"💰 Precio estimado de la vivienda: **${predicted_price:,.0f}**")

    
