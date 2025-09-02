import streamlit as st
from services.auth import logout

def sidebar():
    if st.session_state.get("user") is None:
        return

    with st.sidebar:
        st.write(f"👤 Usuário: {st.session_state['user'].email}")
        if st.button("Sair", key="logout"):
            logout()
            st.session_state["user"] = None
            # Substitua por:
            st.session_state["sidebar_hidden"] = True
            st.session_state["user"] = None
            st.session_state["show_signup"] = False
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else None
