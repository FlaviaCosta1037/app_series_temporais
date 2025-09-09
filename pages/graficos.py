import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# -------------------------------
# FUNÇÃO DE GRÁFICOS
# -------------------------------
def exploratory_plots(df_ready: pd.DataFrame, y_cols, cat_col=None, chart_types=None, agg="sum"):
    if not chart_types or not y_cols:
        st.info("Selecione pelo menos uma coluna numérica e um tipo de gráfico.")
        return

    for y_col in y_cols:
        for chart_type in chart_types:
            st.markdown(f"### {chart_type}: {y_col}")
            try:
                if chart_type == "Linha":
                    fig = px.line(df_ready, x=df_ready.index, y=y_col, markers=True,
                                  title=f"Linha: {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Histograma":
                    fig = px.histogram(df_ready, x=y_col,
                                       color=cat_col if cat_col else None,
                                       nbins=30,
                                       title=f"Histograma: {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Box Plot":
                    if cat_col:
                        fig = px.box(df_ready, x=cat_col, y=y_col, points="all")
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Violin Plot":
                    if cat_col:
                        fig = px.violin(df_ready, x=cat_col, y=y_col, box=True, points="all")
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Scatter":
                    if cat_col:
                        fig = px.scatter(df_ready, x=cat_col, y=y_col)
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Bar":
                    if cat_col:
                        df_bar = df_ready.groupby(cat_col)[y_col].agg(agg).reset_index()
                        fig = px.bar(df_bar, x=cat_col, y=y_col,
                                     title=f"Bar Plot ({agg}) - {y_col}")
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Strip":
                    if cat_col:
                        fig = px.strip(df_ready, x=cat_col, y=y_col)
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao gerar gráfico {chart_type}: {e}")


# -------------------------------
# MAIN APP
# -------------------------------
st.title("📊 Análise Gráfica de Séries Temporais")

# -------------------------------
# UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Carregue CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Leitura
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("Selecione a aba", xls.sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet)

    st.write("### Prévia dos dados")
    st.dataframe(df.head())

    # -------------------------------
    # Seleção de colunas numéricas
    # -------------------------------
    st.subheader("Selecione as colunas numéricas")
    numeric_cols_selected = st.multiselect(
        "Colunas numéricas",
        options=df.columns.tolist(),
        default=[],
        key="numeric_cols_select"
    )

    # Filtro positivo/negativo
    num_filter = st.radio(
        "Filtrar valores",
        ["Todos", "Apenas positivos", "Apenas negativos"],
        horizontal=True,
        key="num_filter"
    )

    # -------------------------------
    # Coluna categórica opcional
    # -------------------------------
    st.subheader("Configuração opcional de categóricas")
    cat_col_opt = st.selectbox(
        "Selecione uma coluna categórica (opcional)",
        [""] + df.columns.tolist(),
        index=0,
        key="cat_col_opt"
    )
    cat_col = cat_col_opt if cat_col_opt else None
    if cat_col and is_numeric_dtype(df[cat_col]):
        df[cat_col] = df[cat_col].astype(str)

    # -------------------------------
    # Data
    # -------------------------------
    st.subheader("Configuração de Data")
    has_date_col = st.radio(
        "A base possui uma coluna de data completa?",
        ["Sim", "Não"],
        key="has_date_col"
    )

    if has_date_col == "Sim":
        date_col = st.selectbox("Selecione a coluna de data", df.columns)
        df["__date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        year_col = st.selectbox("Coluna de ano", df.columns)
        month_col = st.selectbox("Coluna de mês", df.columns)
        df["__date"] = pd.to_datetime(
            df[year_col].astype(str) + "-" + df[month_col].astype(str) + "-01",
            errors="coerce"
        )

    df = df.sort_values("__date").reset_index(drop=True)
    df.index = df["__date"]

    # -------------------------------
    # Aplicar filtros numéricos
    # -------------------------------
    df_ready = df.copy()
    for col in numeric_cols_selected:
        if num_filter == "Apenas positivos":
            df_ready = df_ready[df_ready[col] > 0]
        elif num_filter == "Apenas negativos":
            df_ready = df_ready[df_ready[col] < 0]

    # -------------------------------
    # Gráficos
    # -------------------------------
    st.subheader("Configuração dos Gráficos")
    chart_types = st.multiselect(
        "Escolha os tipos de gráfico",
        ["Linha", "Histograma", "Box Plot", "Violin Plot", "Scatter", "Bar", "Strip"]
    )
    agg_choice = st.radio("Agregação", ["sum", "mean"], horizontal=True)

    if st.button("Gerar gráficos"):
        exploratory_plots(
            df_ready,
            y_cols=numeric_cols_selected,
            cat_col=cat_col,
            chart_types=chart_types,
            agg=agg_choice
        )
