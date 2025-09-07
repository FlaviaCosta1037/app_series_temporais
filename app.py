# app.py
import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from services.auth import login, signup, login_google, logout, supabase
from utils.preprocessing import validate_and_prepare
from components.sidebar import sidebar
from components.navbar import navbar
from utils.ui import center_title

st.set_page_config(page_title="Forecast App", layout="wide")

if st.session_state.get("user") is None:
    st.markdown(
        """
        <style>
        /* Esconde o sidebar completamente */
        [data-testid="stSidebar"] {display: none;}
        /* Ajusta o conteúdo para ocupar toda a largura */
        .css-1d391kg {margin-left: 0px;}
        </style>
        """,
        unsafe_allow_html=True
    )
# -------------------------------
# CONTROLE DE LOGIN
# -------------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
if "show_signup" not in st.session_state:
    st.session_state["show_signup"] = False

# -------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------
def perform_login(email, password):
    user = login(email, password)
    if user:
        st.session_state["user"] = user
    else:
        st.error("Credenciais inválidas")

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def read_uploaded_file(uploaded_file):
    """Leitura robusta para csv (vírgula/; ) e excel."""
    if uploaded_file.name.lower().endswith(".csv"):
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=";")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";")
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

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
                # aplica filtro
                start = pd.to_datetime(start)
                end = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                filtered = filtered[(pd.to_datetime(filtered[col]) >= start) & (pd.to_datetime(filtered[col]) <= end)]
            elif is_numeric_dtype(col_data):
                # Range numérico
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
                # Categórico / texto: se baixo cardinalidade -> multiselect; senão -> texto (contains)
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

def infer_default_seasonality(freq: str) -> int:
    """Sugere sazonalidade padrão baseada na frequência escolhida."""
    if freq == "M":
        return 12
    if freq == "W":
        return 52
    if freq == "D":
        return 7  # semanal
    return 12

def make_future_index(last_index, n_periods, freq):
    """Cria índice futuro respeitando a frequência."""
    if is_datetime64_any_dtype(pd.Series([last_index])):
        return pd.date_range(start=last_index + pd.tseries.frequencies.to_offset(freq), periods=n_periods, freq=freq)
    # fallback
    return pd.date_range(start=pd.Timestamp.utcnow().normalize(), periods=n_periods, freq=freq)

# -------------------------------
# TELA DE LOGIN
# -------------------------------
if st.session_state["user"] is None:
    st.title("Entrar no App")

    # Login com Google
    if st.button("Continuar com Google", key="google_login"):
        auth_url = login_google()
        if auth_url:
            st.markdown(f"[Clique aqui para entrar com Google]({auth_url})")
        st.stop()

    # Login com Email/Senha
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Senha", type="password", key="login_password")
    if st.button("Entrar", key="login_button"):
        perform_login(email, password)

    # Criar conta
    if st.button("Criar Conta"):
        st.session_state["show_signup"] = True

    if st.session_state["show_signup"]:
        st.subheader("Preencha seus dados para criar a conta")

        # Inicializa session_state para endereço
        for campo in ["rua", "bairro", "cidade", "uf"]:
            if campo not in st.session_state:
                st.session_state[campo] = ""

        first_name = st.text_input("Nome", key="first_name")
        last_name  = st.text_input("Sobrenome", key="last_name")
        email_new  = st.text_input("Email", key="email_new")
        password_new = st.text_input("Senha", type="password", key="password_new")
        phone = st.text_input("Telefone (Ex: 11999999999)", key="phone")
        cep = st.text_input("CEP", key="cep")

        if st.button("Buscar endereço pelo CEP", key="buscar_cep") and cep:
            try:
                res = requests.get(f"https://viacep.com.br/ws/{cep}/json/")
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["rua"] = data.get("logradouro", "")
                    st.session_state["bairro"] = data.get("bairro", "")
                    st.session_state["cidade"] = data.get("localidade", "")
                    st.session_state["uf"] = data.get("uf", "")
                    st.success("Endereço encontrado!")
                else:
                    st.error("CEP não encontrado")
            except Exception:
                st.error("Erro ao buscar CEP")

        rua         = st.text_input("Rua",     value=st.session_state["rua"], key="rua")
        numero      = st.text_input("Número",  key="numero")
        bairro      = st.text_input("Bairro",  value=st.session_state["bairro"], key="bairro")
        cidade      = st.text_input("Cidade",  value=st.session_state["cidade"], key="cidade")
        uf          = st.text_input("UF",      value=st.session_state["uf"], key="uf")
        complemento = st.text_input("Complemento", key="complemento")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Salvar", key="save_user"):
                if not all([first_name, last_name, email_new, password_new, phone, cep, rua, numero, bairro, cidade, uf]):
                    st.error("Preencha todos os campos obrigatórios")
                else:
                    new_user = signup(email_new, password_new)
                    if new_user:
                        st.success("Conta criada! Agora faça login.")
                        user_id = new_user.user.id
                        supabase.table("users").insert({
                            "id": user_id,
                            "first_name": first_name,
                            "last_name": last_name,
                            "email": email_new,
                            "phone": phone,
                            "cep": cep,
                            "rua": rua,
                            "numero": numero,
                            "bairro": bairro,
                            "cidade": cidade,
                            "uf": uf,
                            "complemento": complemento
                        }).execute()
                        st.session_state["show_signup"] = False
                    else:
                        st.error("Erro ao criar conta")
        with col2:
            if st.button("Limpar Dados", key="clear_form"):
                for key in ["first_name","last_name","email_new","password_new","phone","cep","rua","numero","bairro","cidade","uf","complemento"]:
                    st.session_state[key] = ""
                st.session_state["show_signup"] = False
                st.rerun()

    st.stop()

# -------------------------------
# HOME
# -------------------------------
if st.session_state.get("user") is not None:
    sidebar()
else:
    if "sidebar_hidden" not in st.session_state or not st.session_state["sidebar_hidden"]:
        st.session_state["sidebar_hidden"] = True

center_title("Análise de vendas")
st.subheader("***Upload da Base de Dados***")
uploaded_file = st.file_uploader("Carregue sua planilha (.csv ou .xlsx)", type=["csv", "xlsx"], key="upload_file")

def read_uploaded_file(file, sheet_name=None):
    """Lê arquivo CSV ou Excel (com suporte a múltiplas abas)."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file, sep=None, engine="python")  # auto-detecta separador
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file, sheet_name=sheet_name)
    else:
        st.error("Formato de arquivo não suportado.")
        return None

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Selecione a aba da planilha", xls.sheet_names)
        df = read_uploaded_file(uploaded_file, sheet_name=sheet_name)
    else:
        df = read_uploaded_file(uploaded_file)

    if df is not None:
        st.write("Prévia dos dados:")
        st.dataframe(df.head())

        # -------------------------------
        # Configuração das colunas
        # -------------------------------
        st.subheader("Configuração das colunas")

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
        # Filtro dinâmico
        # -------------------------------
        st.subheader("Filtrar dados")

        cols_to_filter = st.multiselect(
            "Selecione as colunas que deseja aplicar filtros",
            options=df.columns.tolist(),
            default=[]
        )

        filters = {}
        for col in cols_to_filter:
            col_data = df[col]
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
                df = df[(pd.to_datetime(df[col]) >= start) & (pd.to_datetime(df[col]) <= end)]
            elif is_numeric_dtype(col_data):
                # Range numérico
                min_v = float(np.nanmin(col_data))
                max_v = float(np.nanmax(col_data))
                # Escolha de intervalo
                filter_type = st.selectbox(
                    f"Filtrar valores de {col}:", 
                    ["Todos", "Somente positivos", "Somente negativos"], 
                    index=0,
                    key=f"filter_num_type_{col}"
                )
                if filter_type == "Somente positivos":
                    df = df[df[col] > 0]
                elif filter_type == "Somente negativos":
                    df = df[df[col] < 0]
                else:
                    sel_min, sel_max = st.slider(
                        f"Intervalo numérico — {col}",
                        min_value=float(np.nanmin(col_data)),
                        max_value=float(np.nanmax(col_data)),
                        value=(float(np.nanmin(col_data)), float(np.nanmax(col_data))),
                        key=f"filter_num_{col}"
                    )
                    df = df[(df[col] >= sel_min) & (df[col] <= sel_max)]

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
                        df = df[df[col].astype(str).isin(set(selected))]
                else:
                    txt = st.text_input(
                        f"Contém (texto) — {col}",
                        value="",
                        key=f"filter_txt_{col}"
                    )
                    if txt:
                        df = df[df[col].astype(str).str.contains(txt, case=False, na=False)]

        st.write("Dados filtrados:")
        st.dataframe(df.head())


    # 2) Preparação dos Dados
    st.subheader("Preparação dos Dados")

    # (A) Filtros gerais (opcional) ANTES de construir a série
    df_filtered = build_dynamic_filters(df)

    # (B) Definição de data
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
        # cria coluna de data (dia = 1)
        df_filtered["__data"] = pd.to_datetime(
            df_filtered[col_ano].astype(str) + "-" + df_filtered[col_mes].astype(str) + "-01",
            errors="coerce"
        )
        col_data = "__data"

    # (C) Seleção da coluna alvo e tipo
    col_target = st.selectbox("Selecione a coluna Numérica (variável alvo)", df_filtered.columns, key="col_target")
    tipo_target = st.radio("Tipo da coluna alvo", ["Decimal", "Inteiro"], index=0, horizontal=True)

    # Conversões e limpeza
    df_filtered[col_target] = pd.to_numeric(df_filtered[col_target], errors="coerce")
    df_filtered[col_data] = pd.to_datetime(df_filtered[col_data], errors="coerce")
    # remove linhas inválidas
    df_filtered = df_filtered.dropna(subset=[col_data, col_target])

    if tipo_target == "Inteiro":
        df_filtered[col_target] = np.round(df_filtered[col_target]).astype(int)

    # (D) Frequência/Agrupamento
    st.markdown("##### Agregação temporal")
    freq = st.selectbox("Escolha a frequência para agregar a série", ["D", "W", "M"], index=2, help="D=Diária, W=Semanal, M=Mensal")
    agg_fn = st.selectbox("Como agregar a variável alvo nesse período?", ["sum", "mean", "median"], index=0)

    # Aplica agregação
    df_ready = df_filtered[[col_data, col_target]].copy()
    df_ready = df_ready.sort_values(col_data)
    if agg_fn == "sum":
        df_ready = df_ready.groupby(pd.Grouper(key=col_data, freq=freq))[col_target].sum().reset_index()
    elif agg_fn == "mean":
        df_ready = df_ready.groupby(pd.Grouper(key=col_data, freq=freq))[col_target].mean().reset_index()
    else:
        df_ready = df_ready.groupby(pd.Grouper(key=col_data, freq=freq))[col_target].median().reset_index()

    # Valida (usa sua função existente)
    df_valid, error = validate_and_prepare(df_ready.copy(), col_data, col_target)
    if error:
        st.error(error)
        st.stop()

    # Série temporal final
    ts = df_valid.set_index(col_data)
    st.success("✅ Dados preparados com sucesso!")
    st.dataframe(ts.head())
    st.line_chart(ts)

    # -------------------------------
    # MODELAGEM
    # -------------------------------
    st.subheader("Modelagem & Previsão")
    n_forecast = st.number_input("Horizonte de previsão (períodos)", min_value=1, max_value=60, value=12 if freq == "M" else 30, step=1)
    saz_default = infer_default_seasonality(freq)
    saz = st.number_input("Sazonalidade (período)", min_value=0, max_value=365, value=saz_default, step=1, help="Ex.: 12 para mensal, 7 para diária/semana")

    chart_placeholder = st.empty()

    if st.button("Fazer Previsão"):
        series = ts[col_target].astype(float)
        n_total = len(series)
        n_train = int(n_total * 0.6)
        n_val   = int(n_total * 0.2)
        n_test  = n_total - n_train - n_val

        train = series.iloc[:n_train]
        val   = series.iloc[n_train:n_train+n_val]
        test  = series.iloc[n_train+n_val:]

        st.write(f"Divisão: Treino={len(train)}, Validação={len(val)}, Teste={len(test)}")

        # -----------------------------
        # ARIMA
        # -----------------------------
        try:
            arima_order = (4, 1, 4)
            arima_model = ARIMA(train, order=arima_order)
            arima_fit = arima_model.fit()
            arima_val_pred = arima_fit.forecast(steps=len(val))
            arima_test_pred = arima_fit.forecast(steps=len(val)+len(test))[-len(test):]
            # Próximo período
            arima_next = arima_fit.forecast(steps=1)[-1]
        except Exception as e:
            st.warning(f"ARIMA falhou: {e}")
            arima_val_pred = arima_test_pred = np.zeros(len(val))
            arima_next = 0

        # -----------------------------
        # SARIMA
        # -----------------------------
        try:
            if saz > 0:
                sarima_model = SARIMAX(train, order=(2,1,2), seasonal_order=(2,1,2,int(saz)),
                                    enforce_stationarity=False, enforce_invertibility=False)
            else:
                sarima_model = SARIMAX(train, order=(2,1,2),
                                    enforce_stationarity=False, enforce_invertibility=False)
            sarima_fit = sarima_model.fit(disp=False)
            sarima_val_pred = sarima_fit.forecast(steps=len(val))
            sarima_test_pred = sarima_fit.forecast(steps=len(val)+len(test))[-len(test):]
            sarima_next = sarima_fit.forecast(steps=1)[-1]
        except Exception as e:
            st.warning(f"SARIMA falhou: {e}")
            sarima_val_pred = sarima_test_pred = np.zeros(len(val))
            sarima_next = 0

        # -----------------------------
        # SVR
        # -----------------------------
        try:
            ts_reset = ts.reset_index()
            ts_reset["date_ordinal"] = ts_reset[col_data].map(pd.Timestamp.toordinal)
            X = ts_reset["date_ordinal"].values.reshape(-1,1)
            y = ts_reset[col_target].astype(float).values.reshape(-1,1)
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            Xs = x_scaler.fit_transform(X)
            ys = y_scaler.fit_transform(y).ravel()

            svr_model = SVR(kernel="rbf", C=100, gamma=0.001, epsilon=0.1)
            svr_model.fit(Xs[:n_train], ys[:n_train])

            # Validação e teste
            X_val_test = Xs[n_train:]
            svr_pred_scaled = svr_model.predict(X_val_test)
            svr_val_pred = y_scaler.inverse_transform(svr_pred_scaled[:len(val)].reshape(-1,1)).ravel()
            svr_test_pred = y_scaler.inverse_transform(svr_pred_scaled[len(val):].reshape(-1,1)).ravel()

            # Próximo período
            future_index = make_future_index(ts.index[-1], 1, freq)
            X_future = future_index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            Xf = x_scaler.transform(X_future)
            svr_next = y_scaler.inverse_transform(svr_model.predict(Xf).reshape(-1,1)).ravel()[0]
        except Exception as e:
            st.warning(f"SVR falhou: {e}")
            svr_val_pred = svr_test_pred = np.zeros(len(val))
            svr_next = 0

        # -----------------------------
        # Métricas sobre Validação
        # -----------------------------
        m_arima = safe_mape(val.values, arima_val_pred)
        m_sarima = safe_mape(val.values, sarima_val_pred)
        m_svr = safe_mape(val.values, svr_val_pred)

        st.subheader("MAPE (%) sobre Validação")
        st.table({
            "Modelo": ["ARIMA", "SARIMA", "SVR"],
            "MAPE (%)": [round(m_arima,2), round(m_sarima,2), round(m_svr,2)]
        })

        # -----------------------------
        # Gráfico Evolução
        # -----------------------------
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(series.index, series.values, label="Real", color="black")
        ax.plot(val.index, arima_val_pred, label="ARIMA (Val)", linestyle="--")
        ax.plot(val.index, sarima_val_pred, label="SARIMA (Val)", linestyle="--")
        ax.plot(val.index, svr_val_pred, label="SVR (Val)", linestyle="--")

        ax.plot(test.index, arima_test_pred, label="ARIMA (Teste)", linestyle=":")
        ax.plot(test.index, sarima_test_pred, label="SARIMA (Teste)", linestyle=":")
        ax.plot(test.index, svr_test_pred, label="SVR (Teste)", linestyle=":")

        # Próximo período
        ax.scatter(make_future_index(ts.index[-1], 1, freq), [arima_next], color="blue", label="ARIMA Próx.", zorder=5)
        ax.scatter(make_future_index(ts.index[-1], 1, freq), [sarima_next], color="orange", label="SARIMA Próx.", zorder=5)
        ax.scatter(make_future_index(ts.index[-1], 1, freq), [svr_next], color="green", label="SVR Próx.", zorder=5)

        ax.legend()
        chart_placeholder.pyplot(fig)
        plt.close(fig)
        
        # Cria DataFrame de previsão consolidado
        forecast_table = pd.DataFrame({
            "Data": list(val.index) + list(test.index) + list(make_future_index(ts.index[-1],1,freq)),
            "Tipo": ["Validação"]*len(val) + ["Teste"]*len(test) + ["Próximo Período"],
            "ARIMA": list(arima_val_pred) + list(arima_test_pred) + [arima_next],
            "SARIMA": list(sarima_val_pred) + list(sarima_test_pred) + [sarima_next],
            "SVR": list(svr_val_pred) + list(svr_test_pred) + [svr_next]
        })

        st.subheader("Previsões por Modelo")
        st.dataframe(forecast_table)


