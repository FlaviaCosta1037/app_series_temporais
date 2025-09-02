import streamlit as st

def navbar():
    st.markdown(
        """
        <style>
        .navbar {
            display: flex;
            justify-content: space-around;
            background-color: #2c3e50;
            padding: 10px;
            border-radius: 8px;
        }
        .navbar a {
            color: white;
            text-decoration: none;
            font-weight: bold;
        }
        .navbar a:hover {
            color: #1abc9c;
        }
        </style>
        <div class="navbar">
            <a href="?page=home">🏠 Home</a>
            <a href="?page=relatorios">📊 Relatórios</a>
            <a href="?page=configuracoes">⚙️ Configurações</a>
        </div>
        """,
        unsafe_allow_html=True
    )
