import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error

def train_and_forecast(df):
    y = df["target"].values
    X = np.arange(len(y)).reshape(-1, 1)

    results = {}
    mape_scores = {}

    # ARIMA
    model_arima = ARIMA(y, order=(1,1,1)).fit()
    forecast_arima = model_arima.forecast(steps=30)
    results["ARIMA"] = pd.DataFrame({"data": pd.date_range(df["data"].iloc[-1], periods=30, freq="D"), "yhat": forecast_arima})
    mape_scores["ARIMA"] = mean_absolute_percentage_error(y[-30:], forecast_arima[:30]) * 100

    # SARIMA
    model_sarima = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    forecast_sarima = model_sarima.forecast(steps=30)
    results["SARIMA"] = pd.DataFrame({"data": pd.date_range(df["data"].iloc[-1], periods=30, freq="D"), "yhat": forecast_sarima})
    mape_scores["SARIMA"] = mean_absolute_percentage_error(y[-30:], forecast_sarima[:30]) * 100

    # SVR
    svr = SVR(kernel="rbf")
    svr.fit(X, y)
    forecast_svr = svr.predict(np.arange(len(y), len(y)+30).reshape(-1, 1))
    results["SVR"] = pd.DataFrame({"data": pd.date_range(df["data"].iloc[-1], periods=30, freq="D"), "yhat": forecast_svr})
    mape_scores["SVR"] = mean_absolute_percentage_error(y[-30:], forecast_svr[:30]) * 100

    # Tabela de métricas
    mape_table = pd.DataFrame(mape_scores, index=["MAPE (%)"]).T

    return results, mape_table
