# utils/preprocessing.py
import pandas as pd

def validate_and_prepare(df, col_data, col_target, date_format=None):
    """
    Valida e prepara os dados da série temporal.
    - col_data: nome da coluna de datas
    - col_target: nome da coluna numérica
    - date_format: formato da data (ex: "%d/%m/%Y"). Se None, tenta converter automaticamente
    """
    required_cols = [col_data, col_target]

    # Verifica se as colunas existem
    for col in required_cols:
        if col not in df.columns:
            return None, f"⚠️ Sua base precisa ter as colunas: {required_cols}, mas as colunas são: {list(df.columns)}"

    # 🔹 Converte todas as colunas categóricas / texto para maiúsculo
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype(str).str.upper()

    try:
        if date_format:
            df[col_data] = pd.to_datetime(df[col_data], format=date_format, errors="coerce")
        else:
            df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
    except Exception as e:
        return None, f"Erro ao converter a coluna de data: {e}"

    if df[col_data].isna().any():
        return None, f"Algumas datas na coluna '{col_data}' são inválidas. Verifique o formato."

    df[col_target] = pd.to_numeric(df[col_target], errors="coerce")
    if df[col_target].isna().any():
        return None, f"Alguns valores na coluna '{col_target}' não são numéricos."

    df = df.sort_values(col_data).reset_index(drop=True)

    return df, None
