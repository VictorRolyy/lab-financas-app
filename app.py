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
np.random.seed(42) # Garante resultados idênticos ao Colab

st.title("A1 - Lab Finanças: Otimização de Portfólio")
st.markdown("**Aluno:** Victor Hugo Lemos")

# --- BARRA LATERAL ---
st.sidebar.header("Parâmetros do Modelo")
risk_free_annual = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", value=4.0, step=0.1) / 100
test_days = st.sidebar.number_input("Dias de Backtest (Out-of-Sample)", value=252, step=1)
periodo_download = st.sidebar.selectbox("Período de Dados Históricos", ["2y", "5y", "10y"], index=1)

# Ativos (Cópia exata do Notebook)
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

def calculate_max_drawdown(daily_returns):
    # Reconstrói a curva de patrimônio
    cum_rets = (1 + daily_returns).cumprod()
    # Calcula o pico histórico acumulado até cada dia
    peak = cum_rets.cummax()
    # Calcula a queda percentual em relação ao pico
    drawdown = (cum_rets - peak) / peak
    # Pega a pior queda (o valor mínimo)
    return drawdown.min()

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

# --- CACHE DE DADOS ---
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

# --- BOTÃO DE EXECUÇÃO ---
if st.sidebar.button("Rodar Análise Completa"):
    with st.spinner('Processando dados, calculando indicadores e otimizando...'):
        
        # 1. Carregar Dados
        prices = load_data(TICKERS, periodo_download)
        
        # Dividir Treino e Teste
        if len(prices) < test_days * 2:
            st.error("Dados insuficientes.")
            st.stop()
            
        train_prices = prices.iloc[:-test_days]
        test_prices = prices.iloc[-test_days:]
        split_date = train_prices.index[-1]
        
        train_rets = train_prices.pct_change().dropna()
        test_rets = test_prices.pct_change().dropna()
        
        # 2. Calcular Métricas no TREINO (Para o modelo aprender)
        summary = []
        for t in train_prices.columns:
            r = train_rets[t]
            p = train_prices[t]
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
        
        # --- ABAS DE VISUALIZAÇÃO ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Introdução e Dados", 
            "2. Técnica A (K-Means)", 
            "3. Técnica B (Markowitz)", 
            "4. Backtest (Validação)"
        ])
        
        # === TAB 1: JUSTIFICATIVA ===
        with tab1:
            st.markdown("""
            ### Justificativa da Escolha dos Ativos
            A seleção dos ativos foi estruturada para simular um universo de investimento global e diversificado.
            
            * **Renda Variável EUA:** Magnificent 7 (NVDA, AAPL...) e Índices (SPY, QQQ).
            * **Internacional:** Exposição global (VXUS, EEM).
            * **Fatores Setoriais:** Tecnologia (XLK), Saúde (XLV), etc.
            * **Proteção:** Bonds (TLT, LQD) e Ouro (GLD).
            """)
            st.write("### Base de Dados Calculada (Treino)")
            # Fix de formatação: Indexando pelo Ticker
            st.dataframe(metrics.set_index("Ticker").style.format("{:.2f}"))

        # === TAB 2: TÉCNICA A (K-MEANS) ===
        with tab2:
            st.markdown("### Clusterização (Machine Learning)")
            
            scaler = StandardScaler()
            X = scaler.fit_transform(metrics[["Ret","Vol","Sharpe","RSI","BB"]].fillna(0))
            
            col_a, col_b = st.columns(2)
            
            # Gráfico do Cotovelo
            with col_a:
                st.write("**Método do Cotovelo**")
                inertias = []
                k_grid = range(2, 11)
                for k in k_grid:
                    km = KMeans(n_clusters=k, n_init=20, random_state=42)
                    km.fit(X)
                    inertias.append(km.inertia_)
                
                fig_elbow, ax_el = plt.subplots(figsize=(5,3))
                ax_el.plot(k_grid, inertias, marker="o")
                ax_el.set_xlabel("k clusters")
                ax_el.set_ylabel("Inércia")
                ax_el.grid(True, alpha=0.3)
                st.pyplot(fig_elbow)

            # Gráfico Silhouette
            with col_b:
                st.write("**Silhouette Score**")
                sil_scores = []
                for k in k_grid:
                    km = KMeans(n_clusters=k, n_init=20, random_state=42)
                    labels = km.fit_predict(X)
                    sil_scores.append(silhouette_score(X, labels))
                
                best_k = k_grid[np.argmax(sil_scores)]
                
                fig_sil, ax_sil = plt.subplots(figsize=(5,3))
                ax_sil.plot(k_grid, sil_scores, marker="o", color='green')
                ax_sil.axvline(x=best_k, color='r', linestyle='--', label=f'Sugestão k={best_k}')
                ax_sil.set_xlabel("k clusters")
                ax_sil.set_ylabel("Silhouette")
                ax_sil.legend()
                ax_sil.grid(True, alpha=0.3)
                st.pyplot(fig_sil)
            
            st.info(f"O algoritmo definiu **k={best_k}** como o número ideal de clusters.")
            
            # K-Means Final
            kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
            metrics["Cluster"] = kmeans.fit_predict(X)
            
            # Seleção
            sel_a = []
            for c in sorted(metrics["Cluster"].unique()):
                top = metrics[metrics["Cluster"]==c].sort_values("Sharpe", ascending=False).iloc[0]["Ticker"]
                sel_a.append(top)
            
            if len(sel_a) < 5:
                rest = metrics[~metrics["Ticker"].isin(sel_a)].sort_values("Sharpe", ascending=False)
                sel_a.extend(rest["Ticker"].head(5 - len(sel_a)).tolist())
            sel_a = sel_a[:5]
            
            st.success(f"**Carteira Selecionada (Técnica A):** {', '.join(sel_a)}")
            
            # Scatter Plot 
            fig_scat, ax_scat = plt.subplots(figsize=(8,5))
            scatter = ax_scat.scatter(metrics["Vol"], metrics["Ret"], c=metrics["Cluster"], cmap="viridis", s=100, alpha=0.8)
            plt.colorbar(scatter, label="Cluster")
            sel_data = metrics[metrics["Ticker"].isin(sel_a)]
            ax_scat.scatter(sel_data["Vol"], sel_data["Ret"], color='red', marker='*', s=200, label="Selecionadas")
            ax_scat.set_xlabel("Volatilidade")
            ax_scat.set_ylabel("Retorno")
            ax_scat.legend()
            ax_scat.grid(True, alpha=0.3)
            st.pyplot(fig_scat)

        # === TAB 3: TÉCNICA B (MARKOWITZ) ===
        with tab3:
            st.markdown("### Otimização de Markowitz (Max Sharpe)")
            
            top_10 = metrics.sort_values("Sharpe", ascending=False).head(10)["Ticker"].tolist()
            mu_sub = train_rets[top_10].mean() * 252
            cov_sub = train_rets[top_10].cov() * 252
            
            w_opt = solve_max_sharpe(mu_sub.values, cov_sub.values, risk_free_annual)
            
            df_weights = pd.DataFrame({"Ticker": top_10, "Peso": w_opt})
            df_weights = df_weights.sort_values("Peso", ascending=False).head(5)
            df_weights["Peso"] = df_weights["Peso"] / df_weights["Peso"].sum() # Renormaliza
            
            sel_b = df_weights["Ticker"].tolist()
            weights_b = df_weights["Peso"].values
            
            st.success(f"**Carteira Selecionada (Técnica B):** {', '.join(sel_b)}")
            
            fig_bar, ax_bar = plt.subplots(figsize=(8,4))
            ax_bar.bar(df_weights["Ticker"], df_weights["Peso"], color='orange')
            ax_bar.set_title("Pesos Alocados (Top 5)")
            st.pyplot(fig_bar)

        # === TAB 4: BACKTEST ===
        with tab4:
            st.markdown("### Validação Out-of-Sample (O Verdadeiro Teste)")
            st.write(f"Os modelos foram treinados com dados até **{split_date.date()}**. O gráfico abaixo mostra como eles performaram DEPOIS dessa data.")
            
            # Calcular Performance
            r_test_a = test_rets[sel_a].mean(axis=1).fillna(0)
            r_test_b = test_rets[sel_b].mul(weights_b, axis=1).sum(axis=1).fillna(0)
            r_test_bench = test_rets.mean(axis=1).fillna(0)
            
            cum_a = (1 + r_test_a).cumprod()
            cum_b = (1 + r_test_b).cumprod()
            cum_bench = (1 + r_test_bench).cumprod()
            
            # Gráfico ZOOM
            st.write("#### 1. Zoom no Período de Teste")
            fig_zoom, ax_z = plt.subplots(figsize=(10, 5))
            ax_z.plot(cum_a.index, cum_a, label="Téc A (Cluster)")
            ax_z.plot(cum_b.index, cum_b, label="Téc B (Markowitz)")
            ax_z.plot(cum_bench.index, cum_bench, label="Benchmark", linestyle="--", color="gray")
            ax_z.legend()
            ax_z.grid(True, alpha=0.3)
            st.pyplot(fig_zoom)
            
            # Gráfico FULL HISTORY
            st.write("#### 2. Histórico Completo (Treino + Teste)")
            full_ret_a = prices[sel_a].pct_change().mean(axis=1).fillna(0)
            full_ret_b = prices[sel_b].pct_change().mul(weights_b, axis=1).sum(axis=1).fillna(0)
            full_ret_bench = prices.pct_change().mean(axis=1).fillna(0)
            
            full_cum_a = (1 + full_ret_a).cumprod()
            full_cum_b = (1 + full_ret_b).cumprod()
            full_cum_bench = (1 + full_ret_bench).cumprod()
            
            fig_full, ax_f = plt.subplots(figsize=(12, 6))
            ax_f.plot(full_cum_a.index, full_cum_a, label="Téc A", linewidth=1.5)
            ax_f.plot(full_cum_b.index, full_cum_b, label="Téc B", linewidth=1.5)
            ax_f.plot(full_cum_bench.index, full_cum_bench, label="Benchmark", color='gray', linestyle='--', alpha=0.5)
            ax_f.axvline(x=split_date, color='red', linestyle=':', linewidth=2, label="Divisão Treino/Teste")
            ax_f.axvspan(split_date, full_cum_a.index[-1], color='gray', alpha=0.1)
            ax_f.legend()
            ax_f.grid(True, alpha=0.3)
            ax_f.set_title("Performance Histórica Completa")
            st.pyplot(fig_full)
            
            # --- MÉTRICAS FINAIS COM DRAWDOWN ---
            st.write("#### Resumo de Performance (Teste)")
            
            # Cálculo dos Drawdowns
            dd_a = calculate_max_drawdown(r_test_a)
            dd_b = calculate_max_drawdown(r_test_b)
            dd_bench = calculate_max_drawdown(r_test_bench)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Retorno Total (Cluster)", f"{(cum_a.iloc[-1]-1)*100:.2f}%")
            col2.metric("Retorno Total (Markowitz)", f"{(cum_b.iloc[-1]-1)*100:.2f}%")
            col3.metric("Retorno Total (Benchmark)", f"{(cum_bench.iloc[-1]-1)*100:.2f}%")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Max Drawdown (Cluster)", f"{dd_a*100:.2f}%")
            col5.metric("Max Drawdown (Markowitz)", f"{dd_b*100:.2f}%")
            col6.metric("Max Drawdown (Benchmark)", f"{dd_bench*100:.2f}%")

else:
    st.info("👆 Clique no botão 'Rodar Análise Completa' na barra lateral para iniciar.")
