# services/auth.py
from supabase import create_client, Client
import os
import socket

# Configure suas chaves do Supabase
SUPABASE_URL = "https://ttqwqucyuzkydybavvtg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0cXdxdWN5dXpreWR5YmF2dnRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU0NDEzMjEsImV4cCI6MjA3MTAxNzMyMX0.66cv8h5PBrdKSjd56lsjLl-9CGByndUgzbo3IF-ryhA"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_redirect_url():
    """Detecta se está rodando local ou em produção e retorna a URL correta"""
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # Se rodar localmente (localhost ou 127.0.0.1)
    if "localhost" in hostname or local_ip.startswith("127.") or local_ip.startswith("192."):
        return "http://localhost:8501"

    # Se for produção → usa variável de ambiente (melhor prática)
    return os.getenv("APP_URL", "https://seriestemporais.streamlit.app")

def signup(email: str, password: str):
    try:
        user = supabase.auth.sign_up({"email": email, "password": password})
        return user
    except Exception as e:
        print(f"Erro no signup: {e}")
        return None

def login(email: str, password: str):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return user.user if user else None
    except Exception as e:
        print(f"Erro no login: {e}")
        return None

def login_google():
    """Inicia login com Google (local ou deploy)"""
    try:
        redirect_url = get_redirect_url()
        res = supabase.auth.sign_in_with_oauth(
            {"provider": "google", "options": {"redirect_to": redirect_url}}
        )
        return res.url
    except Exception as e:
        print(f"Erro no login Google: {e}")
        return None

def logout():
    try:
        supabase.auth.sign_out()
        return True
    except Exception as e:
        print(f"Erro no logout: {e}")
        return False