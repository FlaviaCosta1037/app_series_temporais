# utils/preprocessing.py
import pandas as pd

REQUIRED_COLS = ["data", "target"]

def validate_and_prepare(df: pd.DataFrame):
    """
    Valida e prepara a base de dados para os modelos de previsão.
    Retorna: (df_preparado, mensagem_erro)
    """
    # 1. Verificar se as colunas obrigatórias existem
    if not all(col in df.columns for col in REQUIRED_COLS):
        return None, f"⚠️ Sua base precisa ter as colunas: {REQUIRED_COLS}"

    # 2. Converter coluna data
    try:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
    except Exception:
        return None, "⚠️ Não foi possível converter a coluna 'data' para datas."

    # 3. Checar se tem datas inválidas
    if df["data"].isna().any():
        return None, "⚠️ Existem valores inválidos na coluna 'data'."

    # 4. Checar se target é numérico
    if not pd.api.types.is_numeric_dtype(df["target"]):
        return None, "⚠️ A coluna 'target' precisa ser numérica (int ou float)."

    # 5. Ordenar por data
    df = df.sort_values("data").reset_index(drop=True)

    return df, None
