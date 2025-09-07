import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import itertools

def create_forecast_df(start_date, steps, forecast):
    """Cria DataFrame padronizado de previsão mensal"""
    return pd.DataFrame({
        "data": pd.date_range(start=start_date + pd.offsets.MonthBegin(1), periods=steps, freq="M"),
        "yhat": forecast
    })

def best_arima(y, p_range=(0,3), d_range=(0,2), q_range=(0,3)):
    """Busca simples para melhor ARIMA pelo menor AIC"""
    best_aic, best_order, best_model = np.inf, None, None
    for p,d,q in itertools.product(range(*p_range), range(*d_range), range(*q_range)):
        try:
            model = ARIMA(y, order=(p,d,q)).fit()
            if model.aic < best_aic:
                best_aic, best_order, best_model = model.aic, (p,d,q), model
        except:
            continue
    return best_model, best_order

def best_sarima(y, p_range=(0,3), d_range=(0,2), q_range=(0,3), s=12):
    """Busca simples para melhor SARIMA pelo menor AIC"""
    best_aic, best_order, best_model = np.inf, None, None
    for p,d,q in itertools.product(range(*p_range), range(*d_range), range(*q_range)):
        try:
            model = SARIMAX(y, order=(p,d,q), seasonal_order=(p,d,q,s)).fit(disp=False)
            if model.aic < best_aic:
                best_aic, best_order, best_model = model.aic, (p,d,q), model
        except:
            continue
    return best_model, best_order

def walk_forward_validation(y, model_func, steps=1, min_train_size=24):
    """
    Walk-forward validation mensal
    y: série temporal
    model_func: função que recebe y_train e retorna previsões
    steps: horizonte de previsão em meses
    min_train_size: mínimo de meses usados para treinar
    """
    preds, actuals = [], []
    for t in range(min_train_size, len(y) - steps):
        train, test = y[:t], y[t:t+steps]
        forecast = model_func(train, steps)
        preds.extend(forecast)
        actuals.extend(test)
    return mean_absolute_percentage_error(actuals, preds) * 100

def train_and_forecast(df, steps=12):
    """
    Treina ARIMA, SARIMA e SVR para previsão mensal
    steps: horizonte de previsão em meses
    """
    y = df["target"].values
    X = np.arange(len(y)).reshape(-1, 1)

    results = {}
    mape_scores = {}

    # Melhor ARIMA
    model_arima, order_arima = best_arima(y)
    forecast_arima = model_arima.forecast(steps=steps)
    results["ARIMA"] = create_forecast_df(df["data"].iloc[-1], steps, forecast_arima)

    mape_scores["ARIMA"] = walk_forward_validation(
        y,
        lambda train, s: ARIMA(train, order=order_arima).fit().forecast(steps=s),
        steps=1
    )

    # Melhor SARIMA
    model_sarima, order_sarima = best_sarima(y)
    forecast_sarima = model_sarima.forecast(steps=steps)
    results["SARIMA"] = create_forecast_df(df["data"].iloc[-1], steps, forecast_sarima)

    mape_scores["SARIMA"] = walk_forward_validation(
        y,
        lambda train, s: SARIMAX(train, order=order_sarima, seasonal_order=(order_sarima[0], order_sarima[1], order_sarima[2], 12)).fit(disp=False).forecast(steps=s),
        steps=1
    )

    # SVR com normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_scaled, y)
    future_X = np.arange(len(y), len(y)+steps).reshape(-1, 1)
    forecast_svr = svr.predict(scaler.transform(future_X))
    results["SVR"] = create_forecast_df(df["data"].iloc[-1], steps, forecast_svr)

    def svr_func(train, s):
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_future = np.arange(len(train), len(train)+s).reshape(-1, 1)
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
        svr.fit(X_train_scaled, train)
        return svr.predict(X_scaler.transform(X_future))

    mape_scores["SVR"] = walk_forward_validation(y, svr_func, steps=1)

    # Tabela de métricas
    mape_table = pd.DataFrame(mape_scores, index=["MAPE (walk-forward %)"]).T

    return results, mape_table
