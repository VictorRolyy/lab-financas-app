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

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Lab Finanças - Victor Hugo", layout="wide")

st.title("A1 - Lab Finanças: Otimização de Portfólio (K-Means vs Markowitz)")
st.markdown("**Aluno:** Victor Hugo Lemos")

# --- BARRA LATERAL (Parâmetros) ---
st.sidebar.header("Configurações")
risk_free_annual = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", value=4.0, step=0.1) / 100
test_days = st.sidebar.number_input("Dias de Backtest (Out-of-Sample)", value=252, step=1)
periodo_download = st.sidebar.selectbox("Período de Dados Históricos", ["2y", "5y", "10y"], index=1)

# Ativos Fixos (Conforme justificativa)
mag7 = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA"]
us_indices = ["SPY","QQQ","IWM"]
intl = ["VXUS","IEFA","EEM","ACWX"]
sectors = ["XLV","XLF","XLE","XLK","XLY"]
bonds = ["TLT","LQD","HYG"]
alts = ["GLD","VNQ","DBC"]
TICKERS = list(set(mag7 + us_indices + intl + sectors + bonds + alts))

# --- FUNÇÕES DE CÁLCULO ---
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
    return (upper - lower) / sma

def annualize_return(daily_returns):
    return (1 + daily_returns.mean())**252 - 1

def annualize_vol(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def sharpe_ratio_annual(daily_returns, rf):
    rf_daily = (1 + rf)**(1/252) - 1
    excess = daily_returns - rf_daily
    return (excess.mean() / (daily_returns.std() + 1e-12)) * np.sqrt(252)

def solve_max_sharpe(mu, cov, rf):
    n = len(mu)
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -((ret - rf) / (vol + 1e-12))
    
    w0 = np.ones(n)/n
    bounds = [(0.0, 1.0)] * n
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

# --- LÓGICA PRINCIPAL (CACHED) ---
@st.cache_data
def load_data(tickers, period):
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    
    # Tratamento para diferentes versões do yfinance
    try:
        prices = data['Close']
    except KeyError:
        # Se for MultiIndex ou outra estrutura
        if isinstance(data.columns, pd.MultiIndex):
            # Tenta pegar nível 0 ou procurar 'Close'
            prices = data.xs('Close', axis=1, level=0, drop_level=True)
        else:
            prices = data
            
    # Limpeza básica
    prices = prices.dropna(axis=1, how='all').ffill().bfill()
    return prices

# Botão para iniciar
if st.sidebar.button("Rodar Análise"):
    with st.spinner('Baixando dados e calculando indicadores...'):
        prices = load_data(TICKERS, periodo_download)
        
        # Separação Treino vs Teste
        if len(prices) < test_days * 2:
            st.error("Dados insuficientes para o período de teste selecionado.")
            st.stop()
            
        train_prices = prices.iloc[:-test_days]
        test_prices = prices.iloc[-test_days:]
        
        train_rets = train_prices.pct_change().dropna()
        test_rets = test_prices.pct_change().dropna()
        
        # --- CÁLCULO DAS FEATURES (In-Sample) ---
        summary = []
        for t in train_prices.columns:
            r = train_rets[t]
            p = train_prices[t]
            
            # Indicadores (último valor do treino)
            try:
                curr_rsi = calculate_rsi(p).iloc[-1]
                curr_bb = calculate_bollinger_width(p).iloc[-1
