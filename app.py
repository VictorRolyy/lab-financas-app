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

st.title("Lab Finanças: Otimização de Portfólio (K-Means vs Markowitz)")
st.markdown("**Aluno:** Victor Hugo Lemos")

# --- BARRA LATERAL (Parâmetros) ---
st.sidebar.header("Configurações")
risk_free_annual = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", value=4.0, step=0.1) / 100
test_days = st.sidebar.number_input("Dias de Backtest (Out-of-Sample)", value=252, step=1)
periodo_download = st.sidebar.selectbox("Período de Dados Históricos", ["2y", "5y", "10y"], index=1)

# Ativos Fixos
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
    try:
        prices = data['Close']
    except KeyError:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data.xs('Close', axis=1, level=0, drop_level=True)
        else:
            prices = data
    prices = prices.dropna(axis=1, how='all').ffill().bfill()
    return prices

# Botão para iniciar
if st.sidebar.button("Rodar Análise"):
    with st.spinner('Baixando dados e calculando indicadores...'):
        prices = load_data(TICKERS, periodo_download)
        
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
            
            # BLOCO QUE ESTAVA DANDO ERRO (CORRIGIDO)
            try:
                curr_rsi = calculate_rsi(p).iloc[-1]
                curr_bb = calculate_bollinger_width(p).iloc[-1]
            except:
                curr_rsi, curr_bb = 50, 0
                
            summary.append([
                t,
                annualize_return(r),
                annualize_vol(r),
                sharpe_ratio_annual(r, risk_free_annual),
                curr_rsi,
                curr_bb
            ])
            
        metrics = pd.DataFrame(summary, columns=["Ticker","Ret","Vol","Sharpe","RSI","BB"])
        
        # --- TAB 1: Justificativa e Dados ---
        tab1, tab2, tab3, tab4 = st.tabs(["1. Universo e Dados", "2. Técnica A (Cluster)", "3. Técnica B (Markowitz)", "4. Backtest e Conclusão"])
        
        with tab1:
            st.markdown("### Justificativa Formal dos Ativos")
            st.info("""
            O universo de 30 ativos foi selecionado para garantir diversificação global e multi-fatorial:
            - **Growth/Tech:** Magnificent 7 (NVDA, AAPL...) e QQQ.
            - **Defensivos:** XLV (Saúde), TLT (Bonds Longos).
            - **Internacional:** EEM (Emergentes), VEA (Desenvolvidos).
            - **Alternativos:** GLD (Ouro), VNQ (Imóveis).
            """)
            st.dataframe(metrics.style.format("{:.2f}"))
            
        # --- TÉCNICA A: K-Means ---
        with tab2:
            st.markdown("### Clustering com K-Means")
            scaler = StandardScaler()
            X = scaler.fit_transform(metrics[["Ret","Vol","Sharpe","RSI","BB"]].fillna(0))
            
            sil_scores = []
            k_range = range(2, 10)
            for k in k_range:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(X)
                sil_scores.append(silhouette_score(X, labels))
            
            best_k = k_range[np.argmax(sil_scores)]
            st.metric("Melhor k (Silhouette)", best_k)
            
            kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
            metrics["Cluster"] = kmeans.fit_predict(X)
            
            sel_a = []
            for c in sorted(metrics["Cluster"].unique()):
                top = metrics[metrics["Cluster"]==c].sort_values("Sharpe", ascending=False).iloc[0]["Ticker"]
                sel_a.append(top)
                
            if len(sel_a) < 5:
                rest = metrics[~metrics["Ticker"].isin(sel_a)].sort_values("Sharpe", ascending=False)
                sel_a.extend(rest["Ticker"].head(5 - len(sel_a)).tolist())
            sel_a = sel_a[:5]
            
            st.success(f"Carteira Cluster (5 ativos): {', '.join(sel_a)}")
            
            fig, ax = plt.subplots()
            scatter = ax.scatter(metrics["Vol"], metrics["Ret"], c=metrics["Cluster"], cmap="viridis")
            plt.colorbar(scatter, label="Cluster")
            ax.set_xlabel("Volatilidade")
            ax.set_ylabel("Retorno")
            ax.set_title("Mapa de Clusters (Risco x Retorno)")
            st.pyplot(fig)

        # --- TÉCNICA B: Markowitz ---
        with tab3:
            st.markdown("### Markowitz (Max Sharpe)")
            top_10 = metrics.sort_values("Sharpe", ascending=False).head(10)["Ticker"].tolist()
            mu_sub = train_rets[top_10].mean() * 252
            cov_sub = train_rets[top_10].cov() * 252
            
            w_opt = solve_max_sharpe(mu_sub.values, cov_sub.values, risk_free_annual)
            
            df_weights = pd.DataFrame({"Ticker": top_10, "Peso": w_opt})
            df_weights = df_weights.sort_values("Peso", ascending=False).head(5)
            df_weights["Peso"] = df_weights["Peso"] / df_weights["Peso"].sum()
            
            sel_b = df_weights["Ticker"].tolist()
            weights_b = df_weights["Peso"].values
            
            st.success(f"Carteira Markowitz (Top 5 pesos): {', '.join(sel_b)}")
            st.bar_chart(df_weights.set_index("Ticker"))

        # --- TÉCNICA 4: Backtest ---
        with tab4:
            st.markdown("### Validação Out-of-Sample")
            st.write(f"Período de Teste: Últimos {test_days} dias")
            
            r_test_a = test_rets[sel_a].mean(axis=1).fillna(0)
            r_test_b = test_rets[sel_b].mul(weights_b, axis=1).sum(axis=1).fillna(0)
            r_test_bench = test_rets.mean(axis=1).fillna(0)
            
            cum_a = (1 + r_test_a).cumprod()
            cum_b = (1 + r_test_b).cumprod()
            cum_bench = (1 + r_test_bench).cumprod()
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(cum_a.index, cum_a, label="Téc A (Cluster)")
            ax2.plot(cum_b.index, cum_b, label="Téc B (Markowitz)")
            ax2.plot(cum_bench.index, cum_bench, label="Benchmark", linestyle="--", color="gray")
            ax2.set_title("Performance Acumulada (Dados Novos)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Retorno Total (Cluster)", f"{(cum_a.iloc[-1]-1)*100:.2f}%")
            col2.metric("Retorno Total (Markowitz)", f"{(cum_b.iloc[-1]-1)*100:.2f}%")
            col3.metric("Retorno Total (Benchmark)", f"{(cum_bench.iloc[-1]-1)*100:.2f}%")

else:
    st.info("Configure os parâmetros na barra lateral e clique em 'Rodar Análise'.")
