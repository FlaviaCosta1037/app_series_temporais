import streamlit as st
from services.auth import logout

def sidebar():
    st.sidebar.write(f"👤 Usuário: {st.session_state['user'].email}")
    if st.sidebar.button("Sair", key="logout"):
        logout()
        st.session_state["user"] = None
        st.rerun()
    