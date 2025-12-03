import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Lab Finanças - Victor Hugo", layout="wide")
np.random.seed(42) # Garante resultados idênticos

st.title("A1 - Lab Finanças: Otimização de Portfólio")
st.markdown("**Aluno:** Victor Hugo Lemos")

# --- BARRA LATERAL ---
st.sidebar.header("Parâmetros do Investidor")
investment_amount = st.sidebar.number_input("Valor a Investir (US$)", min_value=100.0, value=10000.0, step=100.0)
risk_free_annual = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", value=4.0, step=0.1) / 100
test_days = st.sidebar.number_input("Dias de Backtest (Out-of-Sample)", value=252, step=1)
periodo_download = st.sidebar.selectbox("Período de Dados Históricos", ["2y", "5y", "10y"], index=1)

# Ativos
mag7 = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA"]
us_indices = ["SPY","QQQ","IWM"]
intl = ["VXUS","IEFA","EEM","ACWX"]
sectors = ["XLV","XLF","XLE","XLK","XLY"]
bonds = ["TLT","LQD","HYG"]
alts = ["GLD","VNQ","DBC"]
TICKERS = list(set(mag7 + us_indices + intl + sectors + bonds + alts))

# --- FUNÇÕES ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_width(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return (upper - lower)
