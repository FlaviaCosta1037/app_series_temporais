# app.py
import streamlit as st
import pandas as pd
import requests
from services.auth import login, signup, login_google, logout, supabase
from utils.preprocessing import validate_and_prepare

# -------------------------------
# LOGIN
# -------------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None

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
        user = login(email, password)
        if user:
            st.session_state["user"] = user
            st.experimental_rerun()
        else:
            st.error("Credenciais inválidas")

    # -------------------------------
    # FORMULÁRIO DE CRIAÇÃO DE CONTA
    # -------------------------------
    with st.expander("Criar Conta"):
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

        # Campo CEP
        cep = st.text_input("CEP", key="cep")

        # Botão fora do form para buscar CEP
        buscar_cep = st.button("Buscar endereço pelo CEP", key="buscar_cep")
        if buscar_cep and cep:
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

        # Campos de endereço no formulário usando session_state
        rua = st.text_input("Rua", value=st.session_state["rua"], key="rua")
        numero = st.text_input("Número", key="numero")
        bairro = st.text_input("Bairro", value=st.session_state["bairro"], key="bairro")
        cidade = st.text_input("Cidade", value=st.session_state["cidade"], key="cidade")
        uf = st.text_input("UF", value=st.session_state["uf"], key="uf")
        complemento = st.text_input("Complemento", key="complemento")

        # FORMULÁRIO PRINCIPAL
        with st.form("signup_form"):
            submitted = st.form_submit_button("Salvar")
            limpar = st.form_submit_button("Limpar Dados")

            if limpar:
                st.experimental_rerun()  # reinicia o app para limpar tudo

            if submitted:
                # Verifica campos obrigatórios
                if not all([first_name, last_name, email_new, password_new, phone, cep, rua, numero, bairro, cidade, uf]):
                    st.error("Preencha todos os campos obrigatórios")
                else:
                    # Criar usuário no Supabase Auth
                    new_user = signup(email_new, password_new)
                    if new_user:
                        st.success("Conta criada! Agora faça login.")

                        # Persistir dados adicionais no Supabase
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
                    else:
                        st.error("Erro ao criar conta")

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
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df, error = validate_and_prepare(df)

    if error:
        st.error(error)
        st.stop()
    else:
        st.success("✅ Dados validados com sucesso!")
        st.dataframe(df.head())
