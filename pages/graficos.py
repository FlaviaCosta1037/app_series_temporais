# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from services.auth import logout
from utils.preprocessing import validate_and_prepare
from components.sidebar import sidebar
from utils.ui import center_title

st.set_page_config(page_title="Forecast App", layout="wide")

# Esconde sidebar antes do login
if st.session_state.get("user") is None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        .css-1d391kg {margin-left: 0px;}
        </style>
        """,
        unsafe_allow_html=True
    )

if st.session_state.get("user") is not None:
    sidebar()
else:
    if "sidebar_hidden" not in st.session_state or not st.session_state["sidebar_hidden"]:
        st.session_state["sidebar_hidden"] = True

# -------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------        
def build_dynamic_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Cria UI para filtros dinâmicos por tipo de coluna e aplica sobre o DF."""
    st.markdown("#### Filtros avançados (opcional)")
    st.caption("Aplique filtros em qualquer coluna antes da modelagem.")

    filtered = df.copy()
    with st.expander("Abrir filtros", expanded=False):
        # O usuário escolhe quais colunas quer filtrar (para não poluir a UI).
        cols_to_filter = st.multiselect(
            "Escolha as colunas que deseja filtrar",
            options=list(filtered.columns),
            default=[]
        )
        for col in cols_to_filter:
            col_data = filtered[col]
            if is_datetime64_any_dtype(col_data):
                # Intervalo de datas
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
            elif is_numeric_dtype(col_data):
                min_v = float(np.nanmin(col_data))
                max_v = float(np.nanmax(col_data))
                sel_min, sel_max = st.slider(
                    f"Intervalo numérico — {col}",
                    min_value=min_v, max_value=max_v,
                    value=(min_v, max_v),
                    key=f"filter_num_{col}"
                )
                filtered = filtered[(filtered[col] >= sel_min) & (filtered[col] <= sel_max)]
            else:
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
# Título
# -------------------------------
center_title("📊 Análise Exploratória")
st.subheader("***Upload da Base de Dados***")
df = pd.DataFrame()

uploaded_file = st.file_uploader(
    "Carregue sua planilha (.csv ou .xlsx)", 
    type=["csv", "xlsx"], 
    key="upload_file"
)

def read_uploaded_file(file, sheet_name=None):
    """Lê arquivo CSV ou Excel (com suporte a múltiplas abas)."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file, sep=None, engine="python")  # auto-detecta separador
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file, sheet_name=sheet_name)
    else:
        st.error("Formato de arquivo não suportado.")
        return None

# -------------------------------
# Processamento do arquivo
# -------------------------------
if uploaded_file is not None:
    # Excel -> escolher aba
    if uploaded_file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Selecione a aba da planilha", xls.sheet_names)
        df = read_uploaded_file(uploaded_file, sheet_name=sheet_name)
    else:
        df = read_uploaded_file(uploaded_file)

    if df is not None:
        # Converte colunas de texto para maiúsculo
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.upper()


        st.write("### Prévia dos dados")
        st.dataframe(df.head())

        # -------------------------------
        # Configuração das colunas
        # -------------------------------
        st.subheader("⚙️ Configuração das Colunas")

        cols_to_config = st.multiselect(
            "Selecione as colunas que deseja configurar",
            options=df.columns.tolist(),
            default=[]
        )

        col_types = {}
        for col in cols_to_config:
            col_type = st.selectbox(
                f"Selecione o tipo da coluna '{col}'",
                ["Numérica", "Categórica", "Data", "Ignorar"],
                key=f"type_{col}"
            )
            col_types[col] = col_type

        # -------------------------------
        # Filtros dinâmicos
        # -------------------------------
        st.subheader("🔍 Filtrar Dados")
        df_filtered = build_dynamic_filters(df)

        st.write("### Dados Filtrados")
        st.dataframe(df_filtered.head())

    # -------------------------------
    # Preparação dos Dados
    # -------------------------------
    st.subheader("Preparação dos Dados")

    # (A) Definição de data
    tem_coluna_data = st.radio("Sua base já possui uma coluna de data completa (dia/mês/ano)?", ["Sim", "Não"], index=0)
    if tem_coluna_data == "Sim":
        col_data = st.selectbox("Selecione a coluna de Data", df_filtered.columns, key="col_data")
        # Permitir informar formato, opcional
        date_fmt = st.text_input("Formato da data (opcional, ex: %d/%m/%Y)", value="", key="date_fmt")
        if date_fmt.strip():
            df_filtered[col_data] = pd.to_datetime(df_filtered[col_data], format=date_fmt.strip(), errors="coerce")
        else:
            df_filtered[col_data] = pd.to_datetime(df_filtered[col_data], errors="coerce")
    else:
        # Criar data a partir de ano/mês
        col_ano = st.selectbox("Selecione a coluna de Ano", df_filtered.columns, key="col_ano")
        col_mes = st.selectbox("Selecione a coluna de Mês", df_filtered.columns, key="col_mes")
        df_filtered["__data"] = pd.to_datetime(
            df_filtered[col_ano].astype(str) + "-" + df_filtered[col_mes].astype(str) + "-01",
            errors="coerce"
        )
        col_data = "__data"

    # (B) Seleção da coluna alvo e tipo
    col_target = st.selectbox("Selecione a coluna Numérica (variável alvo)", df_filtered.columns, key="col_target")
    tipo_target = st.radio("Tipo da coluna alvo", ["Decimal", "Inteiro"], index=0, horizontal=True)

    # Conversões gerais
    for col in df_filtered.columns:
        if is_numeric_dtype(df_filtered[col]):
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        elif is_datetime64_any_dtype(df_filtered[col]):
            df_filtered[col] = pd.to_datetime(df_filtered[col], errors='coerce')
        else:
            df_filtered[col] = df_filtered[col].astype(str).str.upper()

    # Remove linhas inválidas
    df_filtered = df_filtered.dropna(subset=[col_data, col_target])

    if tipo_target == "Inteiro":
        df_filtered[col_target] = np.round(df_filtered[col_target]).astype(int)

    # (C) Frequência/Agrupamento
    st.markdown("##### Agregação temporal")
    freq = st.selectbox("Escolha a frequência para agregar a série", ["D", "W", "M"], index=2, help="D=Diária, W=Semanal, M=Mensal")
    agg_fn = st.selectbox("Como agregar a variável alvo nesse período?", ["sum", "mean", "median"], index=0)

    df_ready = df_filtered[[col_data, col_target]].copy()
    df_ready = df_ready.sort_values(col_data)
    if agg_fn == "sum":
        df_ready = df_ready.groupby(pd.Grouper(key=col_data, freq=freq))[col_target].sum().reset_index()
    elif agg_fn == "mean":
        df_ready = df_ready.groupby(pd.Grouper(key=col_data, freq=freq))[col_target].mean().reset_index()
    else:
        df_ready = df_ready.groupby(pd.Grouper(key=col_data, freq=freq))[col_target].median().reset_index()

    # Validação final
    df_valid, error = validate_and_prepare(df_ready.copy(), col_data, col_target)
    if error:
        st.error(error)
        st.stop()

    ts = df_valid.set_index(col_data)
    st.success("✅ Dados preparados com sucesso!")
    st.dataframe(ts.head())
    st.line_chart(ts)

# -------------------------------
# Gráficos Dinâmicos
# -------------------------------
st.subheader("📈 Visualizações")

if uploaded_file is not None and not df_filtered.empty:
    numeric_cols = [col for col in df_filtered.columns if is_numeric_dtype(df_filtered[col])]
    date_cols = [col for col in df_filtered.columns if is_datetime64_any_dtype(df_filtered[col])]
    cat_cols = [col for col in df_filtered.columns if df_filtered[col].dtype == "object" or df_filtered[col].dtype.name == "category"]

    if numeric_cols:
        y_col = st.selectbox("Selecione a coluna Numérica para Análise", numeric_cols)

        if date_cols:
            x_date_col = st.selectbox("Selecione a coluna de Data", date_cols)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_filtered[x_date_col], df_filtered[y_col], color="#007acc", linewidth=2.5, marker="o")
            ax.set_title(f"Evolução Temporal de {y_col}", fontsize=16, fontweight="bold")
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_xlabel(x_date_col, fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.xticks(rotation=45)
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(df_filtered[y_col].dropna(), bins=30, color="#009688", edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribuição de {y_col}", fontsize=16, fontweight="bold")
        ax.set_xlabel(y_col, fontsize=12)
        ax.set_ylabel("Frequência", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)

        if cat_cols:
            cat_col = st.selectbox("Selecione a coluna categórica para comparar", cat_cols)
            fig, ax = plt.subplots(figsize=(12, 6))
            df_filtered.boxplot(column=y_col, by=cat_col, ax=ax, grid=False)
            ax.set_title(f"{y_col} por {cat_col}")
            ax.set_xlabel(cat_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            plt.suptitle("")
            st.pyplot(fig)
    else:
        st.info("Nenhuma coluna numérica disponível para gerar gráficos.")
