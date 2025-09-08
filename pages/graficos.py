import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from utils.preprocessing import validate_and_prepare

# -------------------------------
# FUNÇÃO PARA FILTROS DINÂMICOS
# -------------------------------
def build_dynamic_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("#### Filtros avançados (opcional)")
    st.caption("Aplique filtros em qualquer coluna antes da modelagem.")

    filtered = df.copy()
    with st.expander("Abrir filtros", expanded=False):
        cols_to_filter = st.multiselect(
            "Escolha as colunas que deseja filtrar",
            options=list(filtered.columns),
            default=[],
            key="cols_to_filter"
        )

        for col in cols_to_filter:
            col_data = filtered[col]

            # Filtro para datas
            if is_datetime64_any_dtype(col_data):
                min_d = pd.to_datetime(col_data.min())
                max_d = pd.to_datetime(col_data.max())
                start, end = st.date_input(
                    f"Intervalo de datas — {col}",
                    value=(min_d.date(), max_d.date()),
                    key=f"filter_date_{col}"
                )
                start = pd.to_datetime(start)
                end = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                filtered = filtered[(pd.to_datetime(filtered[col]) >= start) & (pd.to_datetime(filtered[col]) <= end)]

            # Filtro para valores categóricos
            elif not is_numeric_dtype(col_data):
                uniques = sorted([str(u) for u in pd.Series(col_data).dropna().unique().tolist()])
                if len(uniques) <= 50:
                    selected = st.multiselect(
                        f"Filtrar valores — {col}",
                        options=uniques,
                        default=[],
                        key=f"filter_cat_{col}"
                    )
                    if selected:
                        filtered = filtered[filtered[col].astype(str).isin(set(selected))]
                else:
                    txt = st.text_input(
                        f"Contém (texto) — {col}",
                        value="",
                        key=f"filter_txt_{col}"
                    )
                    if txt:
                        filtered = filtered[filtered[col].astype(str).str.contains(txt, case=False, na=False)]

    return filtered

# -------------------------------
# FUNÇÃO DE GRÁFICOS EXPLORATÓRIOS AGREGADOS
# -------------------------------
def exploratory_plots_aggregated(df_ready: pd.DataFrame, y_col, cat_col=None, chart_types=None):
    if not chart_types or y_col == "":
        st.info("Selecione a coluna numérica e o(s) tipo(s) de gráfico.")
        return

    for chart_type in chart_types:
        st.markdown(f"### Gráfico: {chart_type}")
        try:
            if chart_type == "Linha":
                fig = px.line(df_ready, x=df_ready.index if df_ready.index.dtype.kind in "M" else df_ready.index,
                              y=y_col, markers=True, title=f"Linha: {y_col}")
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Histograma":
                fig = px.histogram(df_ready, x=y_col,
                                   color=cat_col if cat_col in df_ready.columns else None,
                                   nbins=30,
                                   title=f"Histograma: {y_col}" + (f" por {cat_col}" if cat_col else ""))
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box Plot":
                if not cat_col or cat_col not in df_ready.columns:
                    st.warning("Box plot requer uma coluna categórica válida para comparar.")
                else:
                    fig = px.box(df_ready, x=cat_col, y=y_col, points="all",
                                 title=f"Box Plot: {y_col} por {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Violin Plot":
                if not cat_col or cat_col not in df_ready.columns:
                    st.warning("Violin plot requer uma coluna categórica válida para comparar.")
                else:
                    fig = px.violin(df_ready, x=cat_col, y=y_col, box=True, points="all",
                                    title=f"Violin Plot: {y_col} por {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Scatter":
                if not cat_col or cat_col not in df_ready.columns:
                    st.warning("Scatter plot requer uma coluna categórica válida para comparar.")
                else:
                    fig = px.scatter(df_ready, x=cat_col, y=y_col, title=f"Scatter: {y_col} por {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Bar":
                if not cat_col or cat_col not in df_ready.columns:
                    st.warning("Bar plot requer uma coluna categórica válida para agrupar.")
                else:
                    agg_choice = st.radio("Agregação", ["sum", "mean"], horizontal=True, key=f"agg_{chart_type}")
                    df_bar = df_ready.groupby(cat_col)[y_col].agg(agg_choice).reset_index()
                    fig = px.bar(df_bar, x=cat_col, y=y_col,
                                 title=f"Bar Plot ({agg_choice}) de {y_col} por {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Strip":
                if not cat_col or cat_col not in df_ready.columns:
                    st.warning("Strip plot requer uma coluna categórica válida para comparar.")
                else:
                    fig = px.strip(df_ready, x=cat_col, y=y_col,
                                   title=f"Strip Plot: {y_col} por {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao gerar gráfico {chart_type}: {e}")

# -------------------------------
# MAIN APP
# -------------------------------
st.title("📈 Análise Gráfica de Séries Temporais")

# -------------------------------
# UPLOAD DE ARQUIVO
# -------------------------------
uploaded_file = st.file_uploader(
    "Carregue seu arquivo CSV ou Excel",
    type=["csv", "xlsx"],
    key="file_uploader_main"
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox(
            "Selecione a aba da planilha",
            xls.sheet_names,
            key="sheet_select_main"
        )
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    st.write("### Prévia dos dados")
    st.dataframe(df.head())

    # -------------------------------
    # Configuração das colunas
    # -------------------------------
    st.markdown("#### Configure suas colunas")
    numeric_cols = st.multiselect(
        "Selecione as colunas numéricas",
        options=list(df.columns),
        default=[],
        key="numeric_cols"
    )

    col_types = {}
    for col in df.columns:
        if col in numeric_cols:
            col_types[col] = "Numérica"
        else:
            col_types[col] = "Categórica"

    # -------------------------------
    # Seleção de coluna de data
    # -------------------------------
    has_date_col = st.radio(
        "Você possui uma coluna de data completa?",
        ["Sim", "Não"],
        key="has_date_col"
    )
    if has_date_col == "Sim":
        date_col = st.selectbox(
            "Selecione a coluna de data",
            options=df.columns,
            index=0,
            key="date_col"
        )
        # Verifica o tipo da coluna
        if np.issubdtype(df[date_col].dtype, np.datetime64):
            df["data"] = df[date_col]
        elif np.issubdtype(df[date_col].dtype, np.object_):
            # Tenta converter string para datetime
            df["data"] = pd.to_datetime(df[date_col], errors="coerce")
        elif np.issubdtype(df[date_col].dtype, np.number):
            # Caso seja número de Excel, converte para datetime
            df["data"] = pd.to_datetime(df[date_col], origin='1899-12-30', unit='D', errors="coerce")
        elif np.issubdtype(df[date_col].dtype, np.timedelta64):
            # Se for timedelta, converte para datetime baseado em 1900-01-01
            df["data"] = pd.to_datetime("1900-01-01") + df[date_col]
        else:
            # Se for hora pura, adiciona uma data padrão
            df["data"] = pd.to_datetime("2000-01-01 " + df[date_col].astype(str), errors="coerce")

        # Remove valores inválidos
        df = df.dropna(subset=["data"])

    else:
        year_col = st.selectbox(
            "Selecione a coluna de ano",
            options=df.columns,
            index=0,
            key="year_col"
        )
        month_col = st.selectbox(
            "Selecione a coluna de mês",
            options=df.columns,
            index=0,
            key="month_col"
        )
        df["data"] = pd.to_datetime(df[year_col].astype(str) + "-" + df[month_col].astype(str) + "-01")

    # -------------------------------
    # Filtros dinâmicos
    # -------------------------------
    df_filtered = build_dynamic_filters(df)

    # -------------------------------
    # Inicializando session_state
    # -------------------------------
    for key in ["y_col", "cat_col", "chart_types", "apply"]:
        if key not in st.session_state:
            st.session_state[key] = "" if key in ["y_col", "cat_col"] else [] if key == "chart_types" else False

    # Botão para atualizar gráficos
    if st.button("Atualizar gráficos com filtros"):
        st.session_state.apply = True

    # -------------------------------
    # Apenas processa se o botão for clicado
    # -------------------------------
    if st.session_state.apply:
        filtered_df_ready = df_filtered.copy()

        numeric_cols_filtered = [c for c in filtered_df_ready.columns if is_numeric_dtype(filtered_df_ready[c])]
        cat_cols_filtered = [c for c in filtered_df_ready.columns if filtered_df_ready[c].dtype == "object"
                             or filtered_df_ready[c].dtype.name == "category"]

        # Coluna numérica
        y_col = st.selectbox(
            "Selecione a coluna Numérica para análise",
            options=[""] + numeric_cols_filtered,
            index=0,
            key="y_col"
        )

        # Coluna categórica
        valid_cat_cols = [c for c in cat_cols_filtered if c in filtered_df_ready.columns]
        cat_col_val = st.selectbox(
            "Selecione a coluna categórica (opcional)",
            options=[""] + valid_cat_cols,
            index=0,
            key="cat_col"
        )
        cat_col = cat_col_val if cat_col_val != "" else None

        # Tipos de gráfico
        chart_types = st.multiselect(
            "Escolha os tipos de gráfico",
            ["Linha", "Histograma", "Box Plot", "Violin Plot", "Scatter", "Bar", "Strip"],
            default=[],
            key="chart_types"
        )

        # -------------------------------
        # Séries temporais
        # -------------------------------
        if "data" in filtered_df_ready.columns:
            filtered_df_ready = filtered_df_ready.sort_values("data")
            filtered_df_ready = filtered_df_ready.set_index("data")

        # Geração de gráficos
        exploratory_plots_aggregated(
            filtered_df_ready,
            y_col=y_col,
            cat_col=cat_col,
            chart_types=chart_types
        )
