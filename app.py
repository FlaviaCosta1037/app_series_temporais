# app.py
import streamlit as st
import pandas as pd
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error
from services.auth import login, signup, login_google, logout, supabase
from utils.preprocessing import validate_and_prepare

# -------------------------------
# CONTROLE DE LOGIN
# -------------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
if "show_signup" not in st.session_state:
    st.session_state["show_signup"] = False

# -------------------------------
# FUNÇÃO AUXILIAR PARA LOGIN
# -------------------------------
def perform_login(email, password):
    user = login(email, password)
    if user:
        st.session_state["user"] = user
    else:
        st.error("Credenciais inválidas")

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

    # Link para abrir criação de conta
    if st.button("Criar Conta"):
        st.session_state["show_signup"] = True

    # -------------------------------
    # FORMULÁRIO DE CRIAÇÃO DE CONTA
    # -------------------------------
    if st.session_state["show_signup"]:
        st.subheader("Preencha seus dados para criar a conta")

        # Inicializa session_state para endereço se não existir
        for campo in ["rua", "bairro", "cidade", "uf"]:
            if campo not in st.session_state:
                st.session_state[campo] = ""

        # Campos básicos
        first_name = st.text_input("Nome", key="first_name")
        last_name = st.text_input("Sobrenome", key="last_name")
        email_new = st.text_input("Email", key="email_new")
        password_new = st.text_input("Senha", type="password", key="password_new")
        phone = st.text_input("Telefone (Ex: 11999999999)", key="phone")
        cep = st.text_input("CEP", key="cep")

        # Botão para buscar CEP
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
            except:
                st.error("Erro ao buscar CEP")

        # Campos de endereço usando session_state
        rua = st.text_input("Rua", value=st.session_state["rua"], key="rua")
        numero = st.text_input("Número", key="numero")
        bairro = st.text_input("Bairro", value=st.session_state["bairro"], key="bairro")
        cidade = st.text_input("Cidade", value=st.session_state["cidade"], key="cidade")
        uf = st.text_input("UF", value=st.session_state["uf"], key="uf")
        complemento = st.text_input("Complemento", key="complemento")

        # Botões Salvar / Limpar
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
                st.experimental_rerun()

        # Link para redefinir senha
        st.markdown("[Redefinir senha](#)")

    st.stop()

# -------------------------------
# HOME
# -------------------------------
st.sidebar.write(f"👤 Usuário: {st.session_state['user'].email}")
if st.sidebar.button("Sair", key="logout"):
    logout()
    st.session_state["user"] = None
    st.experimental_rerun()

st.header("Upload da Base de Dados")
uploaded_file = st.file_uploader("Carregue sua planilha (.csv ou .xlsx)", type=["csv", "xlsx"], key="upload_file")

if uploaded_file:
    # 1. Leitura do arquivo
    if uploaded_file.name.endswith(".csv"):
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) == 1:
                # se só tem 1 coluna, tenta ponto e vírgula
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
    else:
        df = pd.read_excel(uploaded_file)

    # Limpa nomes de colunas
    df.columns = [str(c).strip() for c in df.columns]

    st.write("Colunas detectadas:", df.columns.tolist())  

    # 2. Seleção de colunas
    st.subheader("Seleção de colunas")
    col_data = st.selectbox("Selecione a coluna de Data", df.columns, key="col_data")
    col_target = st.selectbox("Selecione a coluna Numérica", df.columns, key="col_target")

    # Converte datas, valores inválidos viram NaT
    df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
    n_invalid = df[col_data].isna().sum()
    if n_invalid > 0:
        st.warning(f"{n_invalid} datas inválidas foram encontradas e serão removidas")
        df = df.dropna(subset=[col_data])

    # Ordena pela data
    df = df.sort_values(col_data)
    ts = df[[col_data, col_target]].copy()
    ts.set_index(col_data, inplace=True)

    st.success("✅ Dados carregados e validados com sucesso!")
    st.dataframe(ts.head())

    # 3. Validação adicional
    df, error = validate_and_prepare(df, col_data, col_target)
    if error:
        st.error(error)
        st.stop()

    # Gráfico simples
    st.line_chart(ts)

    # -------------------------------
    # Botão Fazer Previsão
    # -------------------------------
    if st.button("Fazer Previsão"):
        n_forecast = 30

        # ARIMA
        try:
            arima_model = auto_arima(ts[col_target], seasonal=False, stepwise=True)
            arima_pred = arima_model.predict(n_periods=n_forecast)
        except Exception as e:
            st.warning(f"ARIMA falhou: {e}")
            arima_pred = np.zeros(n_forecast)

        # SARIMA
        try:
            sarima_model = auto_arima(ts[col_target], seasonal=True, m=12, stepwise=True)
            sarima_pred = sarima_model.predict(n_periods=n_forecast)
        except Exception as e:
            st.warning(f"SARIMA falhou: {e}")
            sarima_pred = np.zeros(n_forecast)

        # SVR
        try:
            ts_reset = ts.reset_index()
            ts_reset['date_ordinal'] = ts_reset[col_data].map(pd.Timestamp.toordinal)
            X = ts_reset['date_ordinal'].values.reshape(-1,1)
            y = ts_reset[col_target].values
            svr_model = SVR(kernel='rbf', C=100, gamma=000.1, epsilon=.1)
            svr_model.fit(X, y)
            future_dates = pd.date_range(ts.index[-1]+pd.Timedelta(days=1), periods=n_forecast)
            X_future = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1,1)
            svr_pred = svr_model.predict(X_future)
        except Exception as e:
            st.warning(f"SVR falhou: {e}")
            svr_pred = np.zeros(n_forecast)

        # Métricas MAPE
        y_true = ts[col_target].values[-n_forecast:] if len(ts) >= n_forecast else ts[col_target].values
        mape_arima = mean_absolute_percentage_error(y_true, arima_pred[:len(y_true)])*100
        mape_sarima = mean_absolute_percentage_error(y_true, sarima_pred[:len(y_true)])*100
        mape_svr = mean_absolute_percentage_error(y_true, svr_pred[:len(y_true)])*100

        st.subheader("MAPE (%) de cada modelo")
        st.table({
            "Modelo": ["ARIMA", "SARIMA", "SVR"],
            "MAPE (%)": [round(mape_arima,2), round(mape_sarima,2), round(mape_svr,2)]
        })

        # Previsão para os próximos dias
        st.subheader("Previsão para os próximos dias")
        forecast_dates = pd.date_range(ts.index[-1]+pd.Timedelta(days=1), periods=n_forecast)
        forecast_df = pd.DataFrame({
            "Data": forecast_dates,
            "ARIMA": arima_pred,
            "SARIMA": sarima_pred,
            "SVR": svr_pred
        })
        st.dataframe(forecast_df)

        # Total do próximo mês
        st.subheader("Total previsto do próximo mês")
        st.write({
            "ARIMA": round(arima_pred.sum(),2),
            "SARIMA": round(sarima_pred.sum(),2),
            "SVR": round(svr_pred.sum(),2)
        })

        # Gráfico das previsões
        plt.figure(figsize=(10,5))
        plt.plot(ts.index, ts[col_target], label="Real")
        plt.plot(forecast_dates, arima_pred, label="ARIMA")
        plt.plot(forecast_dates, sarima_pred, label="SARIMA")
        plt.plot(forecast_dates, svr_pred, label="SVR")
        plt.legend()
        st.pyplot(plt)
