import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt  # Corrected import statement

def calcular_var(tickers, start_date, end_date, monto, nivel_confianza, pesos=None):
    """Calcula el Valor en Riesgo (VaR) de una cartera.

    Args:
        tickers (list): Lista de tickers de los activos.
        start_date (datetime): Fecha de inicio.
        end_date (datetime): Fecha de fin.
        monto (float): Monto a invertir.
        nivel_confianza (float): Nivel de confianza.
        pesos (list, optional): Pesos de cada activo en la cartera. Por defecto, None.

    Returns:
        float: Valor en riesgo de la cartera.
    """

    try:
        # Descargar datos y calcular retornos
        datos = yf.download(tickers, start=start_date, end=end_date)
        retornos = np.log(datos['Adj Close'] / datos['Adj Close'].shift(1)).dropna()

        # Calcular la matriz de covarianzas
        matriz_covarianza = retornos.cov()

        # Verificar si la matriz de covarianzas es definida positiva
        if not np.all(np.linalg.eigvals(matriz_covarianza) > 0):
            st.error("La matriz de covarianzas no es definida positiva. Verifica los datos.")
            return

        # Calcular el retorno esperado de la cartera
        retorno_esperado = np.sum(retornos.mean() * pesos) if pesos is not None else np.mean(retornos.mean())

        # Calcular la volatilidad de la cartera
        volatilidad_cartera = np.sqrt(np.dot(pesos, np.dot(matriz_covarianza, pesos))) if pesos is not None else np.std(retornos)

        # Calcular el VaR de la cartera
        VaR = norm.ppf(1 - nivel_confianza, loc=retorno_esperado, scale=volatilidad_cartera) * monto

        return VaR
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

# Título de la aplicación
st.title("Calculadora de Valor en Riesgo (VaR)")

# Parámetros de entrada
tickers = st.text_input("Ingrese los tickers separados por comas (ej: AAPL,GOOGL,MSFT)", value="AAPL")
fecha_inicio = st.date_input("Fecha de inicio", value=pd.to_datetime('2023-01-01'))
fecha_fin = st.date_input("Fecha de fin", value=pd.to_datetime('today'))
monto = st.number_input("Monto a invertir", value=10000.0)
nivel_confianza = st.slider("Nivel de confianza", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
pesos_str = st.text_input("Ingrese los pesos de los activos separados por comas (opcional)", value="")

# Convertir los pesos a un array de numpy
pesos = None
if pesos_str:
    try:
        pesos = np.array([float(w) for w in pesos_str.split(',')])
        if not np.isclose(np.sum(pesos), 1):
            st.error("La suma de los pesos debe ser igual a 1.")
            pesos = None
    except ValueError:
        st.error("Los pesos deben ser números.")

# Calcular el VaR
if st.button("Calcular VaR"):
    if pesos is None:
        pesos = np.ones(len(tickers.split(','))) / len(tickers.split(','))  # Pesos iguales por defecto
    var = calcular_var(tickers.split(','), fecha_inicio, fecha_fin, monto, nivel_confianza, pesos)
    if var is not None:
        st.write("El VaR de la cartera es:", var)

    # Visualizaciones
    st.subheader("Precios Históricos")
    st.line_chart(datos['Adj Close'])

    st.subheader("Distribución de Retornos")
    fig, ax = plt.subplots()
    ax.hist(retornos.flatten(), bins=50, density=True)
    ax.set_xlabel('Retornos')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Retornos')
    st.pyplot(fig)

    st.subheader("Valor en Riesgo (VaR)")
    fig, ax = plt.subplots()
    x = np.linspace(retornos.min(), retornos.max(), 1000)
    y = norm.pdf(x, loc=retorno_esperado, scale=volatilidad_cartera)
    ax.plot(x, y)
    ax.fill_between(x, y, where=(x <= norm.ppf(1 - nivel_confianza, loc=retorno_esperado, scale=volatilidad_cartera)), alpha=0.5)
    ax.set_xlabel('Retornos')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Retornos y VaR')
    st.pyplot(fig)
