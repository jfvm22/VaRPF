import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm 1 

def calcular_var_cartera(tickers, start_date, end_date, tipo_cambio, confidence_level, horizon, pesos):
    """Calcula el VaR de una cartera.

    Args:
        tickers (list): Lista de tickers de los activos.
        start_date (datetime): Fecha de inicio.
        end_date (datetime): Fecha de fin.
        tipo_cambio (float): Tipo de cambio.
        confidence_level (float): Nivel de confianza.
        horizon (int): Horizonte de tiempo en días.
        pesos (list): Lista de pesos de cada activo.

    Returns:
        float: Valor en riesgo de la cartera.
    """

    # Descargar datos y calcular retornos
    data = yf.download(tickers, start=start_date, end=end_date)
    data['Adj Close'] *= tipo_cambio
    returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

    # Calcular la matriz de covarianzas
    cov_matrix = returns.cov()

    # Calcular el retorno esperado de la cartera
    portfolio_return = np.sum(returns.mean() * pesos)

    # Calcular la volatilidad de la cartera
    portfolio_std = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))

    # Calcular el VaR de la cartera
    VaR = norm.ppf(1 - confidence_level, loc=portfolio_return, scale=portfolio_std * np.sqrt(horizon)) * np.sum(pesos * data['Adj Close'].iloc[-1])

    return VaR

# Título de la aplicación
st.title("Calculadora de VaR para Carteras de Acciones")

# Parámetros de entrada
tickers = st.text_input("Ingrese los tickers separados por comas (ej: AAPL,GOOGL,MSFT)", value="AAPL")
start_date = st.date_input("Fecha de inicio", value=pd.to_datetime('2023-01-01'))
end_date = st.date_input("Fecha de fin", value=pd.to_datetime('today'))
tipo_cambio = st.number_input("Tipo de cambio MXN/USD", value=20.0)
confidence_level = st.slider("Nivel de confianza", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
horizon = st.number_input("Horizonte de tiempo (días)", min_value=1, value=1)

# Pesos de la cartera
pesos_str = st.text_input("Ingrese los pesos de cada activo separados por comas (ej: 0.3,0.4,0.3)")
pesos = np.array([float(peso) for peso in pesos_str.split(',')])

# Calcular el VaR
VaR = calcular_var_cartera(tickers.split(','), start_date, end_date, tipo_cambio, confidence_level, horizon, pesos)

# Mostrar los resultados
st.write("## Resultados")
st.write(f"VaR de la cartera: {VaR:.2f} MXN")
