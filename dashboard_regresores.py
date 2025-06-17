
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Dashboard - Regresores Ames Housing",
    page_icon="🏠",
    layout="wide"
)

# Título principal
st.title("🏠 Dashboard de Comparación de Regresores")
# Reemplazar st.subtitle() con st.write() o st.markdown()
st.write("Análisis del Dataset Ames Housing")
st.markdown("---")

# Datos de los resultados (sustituir con los datos reales)
results_data = {
    'SGDRegressor': {'MAE': 14599.92, 'MSE': 523980268.87, 'R2': 0.918, 'MAPE': 8.90},
    'RandomForestRegressor': {'MAE': 15673.01, 'MSE': 658723700.00, 'R2': 0.896, 'MAPE': 9.57},
    'ElasticNet': {'MAE': 18845.58, 'MSE': 996140900.00, 'R2': 0.841, 'MAPE': 11.34},
    'BayesianRidge': {'MAE': 18899.55, 'MSE': 1002326000.00, 'R2': 0.840, 'MAPE': 11.40},
    'LinearRegression': {'MAE': 18970.79, 'MSE': 1010674000.00, 'R2': 0.839, 'MAPE': 11.46},
    'Lasso': {'MAE': 18970.18, 'MSE': 1010560000.00, 'R2': 0.839, 'MAPE': 11.46},
    'KernelRidge': {'MAE': 22419.10, 'MSE': 2065620000.00, 'R2': 0.665, 'MAPE': 12.04},
    'SupportVectorMachines': {'MAE': 55991.67, 'MSE': 6790110000.00, 'R2': -0.065, 'MAPE': 32.08},
    'GaussianProcessRegressor': {'MAE': 178353.46, 'MSE': 38317940000.00, 'R2': -5.090, 'MAPE': 98.64}
}

# Convertir a DataFrame
df_results = pd.DataFrame(results_data).T
df_results = df_results.round(3)

# Sidebar para navegación
st.sidebar.title("📊 Navegación")
page = st.sidebar.selectbox("Seleccionar página:",
                           ["Resumen General", "Comparación Detallada", "Top 3 Modelos", "Análisis por Métrica"])

if page == "Resumen General":
    st.header("📈 Resumen General del Análisis")

    # Métricas principales en columnas
    col1, col2, col3, col4 = st.columns(4)

    best_model = df_results.loc[df_results['R2'].idxmax()]

    with col1:
        st.metric("🏆 Mejor Modelo", "SGDRegressor", f"R² = {best_model['R2']:.3f}")
    with col2:
        st.metric("📊 MAE Mínimo", f"${best_model['MAE']:,.0f}", "SGDRegressor")
    with col3:
        st.metric("📈 R² Máximo", f"{best_model['R2']:.3f}", "SGDRegressor")
    with col4:
        st.metric("📉 MAPE Mínimo", f"{best_model['MAPE']:.1f}%", "SGDRegressor")

    st.markdown("---")

    # Gráfico de barras comparativo
    st.subheader("Comparación de R² por Modelo")

    # Crear gráfico de barras con Plotly
    fig_bar = px.bar(
        x=df_results.index,
        y=df_results['R2'],
        title="R² Score por Modelo de Regresión",
        labels={'x': 'Modelos', 'y': 'R² Score'},
        color=df_results['R2'],
        color_continuous_scale='Viridis'
    )
    fig_bar.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Comparación Detallada":
    st.header("🔍 Comparación Detallada de Todos los Modelos")

    # Tabla completa de resultados
    st.subheader("Tabla de Resultados Completa")
    st.dataframe(df_results.sort_values('R2', ascending=False), use_container_width=True)

    # Gráficos comparativos
    st.subheader("Visualizaciones Comparativas")

    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('MAE por Modelo', 'MSE por Modelo', 'R² por Modelo', 'MAPE por Modelo'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # MAE
    fig.add_trace(
        go.Bar(x=df_results.index, y=df_results['MAE'], name='MAE', marker_color='lightblue'),
        row=1, col=1
    )

    # MSE (en escala logarítmica para mejor visualización)
    fig.add_trace(
        go.Bar(x=df_results.index, y=np.log10(df_results['MSE']), name='log10(MSE)', marker_color='lightgreen'),
        row=1, col=2
    )

    # R²
    fig.add_trace(
        go.Bar(x=df_results.index, y=df_results['R2'], name='R²', marker_color='salmon'),
        row=2, col=1
    )

    # MAPE
    fig.add_trace(
        go.Bar(x=df_results.index, y=df_results['MAPE'], name='MAPE', marker_color='gold'),
        row=2, col=2
    )

    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Top 3 Modelos":
    st.header("🥇 Top 3 Mejores Modelos")

    # Seleccionar top 3
    top_3 = df_results.sort_values('R2', ascending=False).head(3)

    # Mostrar en columnas
    col1, col2, col3 = st.columns(3)

    models_list = top_3.index.tolist()

    with col1:
        st.subheader("🥇 1er Lugar")
        st.metric("Modelo", models_list[0])
        st.metric("R²", f"{top_3.iloc[0]['R2']:.3f}")
        st.metric("MAE", f"${top_3.iloc[0]['MAE']:,.0f}")
        st.metric("MAPE", f"{top_3.iloc[0]['MAPE']:.1f}%")

    with col2:
        st.subheader("🥈 2do Lugar")
        st.metric("Modelo", models_list[1])
        st.metric("R²", f"{top_3.iloc[1]['R2']:.3f}")
        st.metric("MAE", f"${top_3.iloc[1]['MAE']:,.0f}")
        st.metric("MAPE", f"{top_3.iloc[1]['MAPE']:.1f}%")

    with col3:
        st.subheader("🥉 3er Lugar")
        st.metric("Modelo", models_list[2])
        st.metric("R²", f"{top_3.iloc[2]['R2']:.3f}")
        st.metric("MAE", f"${top_3.iloc[2]['MAE']:,.0f}")
        st.metric("MAPE", f"{top_3.iloc[2]['MAPE']:.1f}%")

    st.markdown("---")

    # Gráfico radar para top 3
    st.subheader("Comparación Radar - Top 3 Modelos")

    # Normalizar métricas para el gráfico radar (invertir MAE y MAPE para que más alto sea mejor)
    normalized_data = top_3.copy()
    normalized_data['MAE_norm'] = 1 - (normalized_data['MAE'] / normalized_data['MAE'].max())
    normalized_data['MAPE_norm'] = 1 - (normalized_data['MAPE'] / normalized_data['MAPE'].max())
    normalized_data['R2_norm'] = normalized_data['R2'] / normalized_data['R2'].max()

    fig_radar = go.Figure()

    for i, model in enumerate(models_list):
        fig_radar.add_trace(go.Scatterpolar(
            r=[normalized_data.loc[model, 'R2_norm'],
               normalized_data.loc[model, 'MAE_norm'],
               normalized_data.loc[model, 'MAPE_norm']],
            theta=['R² Score', 'MAE (invertido)', 'MAPE (invertido)'],
            fill='toself',
            name=model
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=500
    )

    st.plotly_chart(fig_radar, use_container_width=True)

elif page == "Análisis por Métrica":
    st.header("📊 Análisis Detallado por Métrica")

    # Selector de métrica
    metric = st.selectbox("Seleccionar métrica:", ['R2', 'MAE', 'MSE', 'MAPE'])

    # Ordenar según la métrica seleccionada
    if metric in ['MAE', 'MSE', 'MAPE']:
        sorted_df = df_results.sort_values(metric, ascending=True)
        better_text = "menor"
    else:
        sorted_df = df_results.sort_values(metric, ascending=False)
        better_text = "mayor"

    st.subheader(f"Ranking por {metric} ({better_text} es mejor)")

    # Mostrar ranking
    for i, (model, row) in enumerate(sorted_df.iterrows()):
        if i < 3:  # Top 3
            emoji = ["🥇", "🥈", "🥉"][i]
            st.success(f"{emoji} **{model}**: {row[metric]:.3f}")
        else:
            st.info(f"{i+1}. **{model}**: {row[metric]:.3f}")

    # Gráfico específico de la métrica
    fig_metric = px.bar(
        x=sorted_df.index,
        y=sorted_df[metric],
        title=f"Distribución de {metric} por Modelo",
        labels={'x': 'Modelos', 'y': metric},
        color=sorted_df[metric],
        color_continuous_scale='RdYlBu_r' if metric in ['MAE', 'MSE', 'MAPE'] else 'RdYlBu'
    )
    fig_metric.update_layout(height=500, showlegend=False)
    fig_metric.update_xaxes(tickangle=45)
    st.plotly_chart(fig_metric, use_container_width=True)

    # Estadísticas de la métrica
    st.subheader(f"Estadísticas de {metric}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mejor", f"{sorted_df[metric].iloc[0]:.3f}", sorted_df.index[0])
    with col2:
        st.metric("Peor", f"{sorted_df[metric].iloc[-1]:.3f}", sorted_df.index[-1])
    with col3:
        st.metric("Promedio", f"{sorted_df[metric].mean():.3f}")
    with col4:
        st.metric("Desv. Estándar", f"{sorted_df[metric].std():.3f}")

# Footer
st.markdown("---")
st.markdown("**Dashboard creado para el análisis de regresores en el dataset Ames Housing**")
st.markdown("*Utiliza el menú lateral para navegar entre las diferentes secciones*")
