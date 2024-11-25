import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

def calcular_var(tickers, start_date, end_date, monto, confidence_level):
    """Calcula el VaR de una cartera.

    Args:
        tickers (list): Lista de tickers de los activos.
        start_date (datetime): Fecha de inicio.
        end_date (datetime): Fecha de fin.
        monto (float): Monto a invertir.
        confidence_level (float): Nivel de confianza.

    Returns:
        float: Valor en riesgo de la cartera.
    """

    # Descargar datos y calcular retornos
    data = yf.download(tickers, start=start_date, end=end_date)
    returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

    # Calcular la matriz de covarianzas
    cov_matrix = returns.cov()

    # Calcular el retorno esperado de la cartera
    portfolio_return = np.sum(returns.mean()) / len(returns.columns)

    # Calcular la volatilidad de la cartera
    portfolio_std = np.sqrt(np.dot(np.ones(len(returns.columns)).T, np.dot(cov_matrix, np.ones(len(returns.columns))))) / len(returns.columns))

    # Calcular el VaR de la cartera
    VaR = norm.ppf(1 - confidence_level, loc=portfolio_return, scale=portfolio_std) * monto

    return VaR

# Título de la aplicación
st.title("Calculadora de VaR")

# Parámetros de entrada
tickers = st.text_input("Ingrese los tickers separados por comas (ej: AAPL,GOOGL,MSFT)", value="AAPL")
start_date = st.date_input("Fecha de inicio", value=pd.to_datetime('2023-01-01'))
end_date = st.date_input("Fecha de fin", value=pd.to_datetime('today'))
monto = st.number_input("Monto a invertir", value=10000.0)
confidence_level = st.slider("Nivel de confianza", min_value=0.90, max_value=0.99, value=0.95, step=0.01)

# Calcular el VaR
if st.button("Calcular VaR"):
    var = calcular_var(tickers.split(','), start_date, end_date, monto, confidence_level)
    st.write("El VaR de la cartera es:", var)
