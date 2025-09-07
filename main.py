import yfinance as yf
import pandas as pd
import streamlit as st
import datetime as dt
import requests
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

# ---------------- TELEGRAM CONFIG ----------------
TELEGRAM_TOKEN = "8318189112:AAFcnSVKkCC5Sd7d5wKz3rg72l2LbpG5_uA"
CHAT_ID = "ETF_Screener_bot"

def send_telegram(msg):
    """Sends a message to a Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Error enviando mensaje Telegram: {e}")

# ---------------- INDICATORS FUNCTIONS ----------------
def compute_rsi(data, window=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------- STRATEGY FUNCTIONS ----------------
def check_buy_signal(df):
    """
    Checks for a buy signal based on the defined conditions.
    Assumes df has 'Close', 'SMA50', 'SMA200', and 'RSI' columns.
    """
    if df.empty:
        return False
    
    last_price = df["Close"].iloc[-1]
    
    # Condition 1: Drawdown from all-time high
    max_price = df["Close"].max()
    drawdown = (last_price / max_price - 1) * 100
    cond1 = drawdown <= -10
    
    # Condition 2: Last week was bearish (simple check)
    if len(df) >= 5:
        last_week = df["Close"].iloc[-5:]
        cond2 = last_week.iloc[-1] < last_week.iloc[0]
    else:
        cond2 = False
    
    # Condition 3: Price is between SMA50 and SMA200
    sma50 = df["SMA50"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]
    cond3 = (last_price < sma50) and (last_price > sma200)
    
    # Condition 4: RSI is oversold
    rsi = df["RSI"].iloc[-1]
    cond4 = rsi < 30
    
    return cond1 and cond2 and cond3 and cond4

def run_backtest(df):
    """
    Performs a backtest of the strategy with new sell conditions.
    """
    capital = 100000
    positions = []
    
    # Calculate indicators for the whole period
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = compute_rsi(df["Close"])
    
    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df["Close"].iloc[i]

        # Check for open positions to close
        for pos in positions.copy(): # Iterate over a copy to allow modification
            entry_price = pos['price']
            entry_date = pos['date']
            
            # --- SELL CONDITIONS ---
            
            # Condition 1: Take-Profit (5% profit)
            take_profit_level = entry_price * 1.05
            if current_price >= take_profit_level:
                gain = (current_price - entry_price) / entry_price
                capital += pos['value'] * (1 + gain)
                positions.remove(pos)
                continue
                
            # Condition 2: Stop-Loss (5% loss)
            stop_loss_level = entry_price * 0.95
            if current_price <= stop_loss_level:
                gain = (current_price - entry_price) / entry_price
                capital += pos['value'] * (1 + gain)
                positions.remove(pos)
                continue
            
            # Condition 3: RSI oversold
            current_rsi = df["RSI"].iloc[i]
            if current_rsi > 70:
                gain = (current_price - entry_price) / entry_price
                capital += pos['value'] * (1 + gain)
                positions.remove(pos)
                continue
            
            # Condition 4: Time limit (14 days)
            if (current_date - entry_date).days >= 30:
                gain = (current_price - entry_price) / entry_price
                capital += pos['value'] * (1 + gain)
                positions.remove(pos)
                continue

        # Check for new buy signal
        df_slice = df.iloc[:i+1] # Look at data up to the current day
        if check_buy_signal(df_slice) and capital > 0:
            buy_price = df["Open"].iloc[i+1] if i+1 < len(df) else df["Close"].iloc[i]
            investment = capital * 0.25 # Invest 25% of capital
            positions.append({
                'date': current_date,
                'price': buy_price,
                'value': investment
            })
            capital -= investment
    
    # Close all remaining positions
    for pos in positions:
        sell_price = df["Close"].iloc[-1]
        gain = (sell_price - pos['price']) / pos['price']
        capital += pos['value'] * (1 + gain)
        
    return capital

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="ETF Screener", layout="wide")
st.title("🚀 Screener ETFs con Alertas Telegram y Backtesting")

# Ticker list
etfs = ["XLK", "XLY", "XLE", "ARKK", "GLD", "MSOS","SPY", "QQQ", "VTI", "IWM", "EEM", "VNQ", "LQD", "BND","CS1.PA"]

# Dates
end = dt.date.today()
start = end - dt.timedelta(days=365*5)  # 5 years for backtesting

# Main functionality
if st.button("Analizar ETFs"):
    if not etfs:
        st.warning("Por favor, introduce al menos un ticker.")
    else:
        st.info("Descargando datos...")
        try:
            df_full = yf.download(etfs, start=start, end=end, progress=False)
            
            # Get ETF names
            etf_names = {}
            for ticker in etfs:
                try:
                    info = yf.Ticker(ticker).info
                    etf_names[ticker] = info.get('longName', ticker)
                except Exception:
                    etf_names[ticker] = ticker
                    
        except Exception as e:
            st.error(f"Error al descargar datos: {e}")
            df_full = pd.DataFrame()
            etf_names = {ticker: ticker for ticker in etfs}

        if df_full.empty:
            st.warning("No se encontraron datos para los tickers.")
        else:
            data = {}
            alerts = []
            results_backtest = {}
            
            for ticker in etfs:
                # Handle single ticker data
                if len(etfs) > 1:
                    df = df_full['Close'][ticker].dropna().to_frame()
                    df.columns = ["Close"]
                    if 'Open' in df_full.columns:
                        df['Open'] = df_full['Open'][ticker].dropna()
                else:
                    df = df_full.dropna().copy()
                
                if df.empty or len(df) < 200:
                    st.warning(f"No hay suficientes datos para {ticker}.")
                    continue
                
                # Run screener logic on the most recent data
                df["SMA50"] = df["Close"].rolling(50).mean()
                df["SMA200"] = df["Close"].rolling(200).mean()
                df["RSI"] = compute_rsi(df["Close"])

                # Check for a buy signal today
                if check_buy_signal(df):
                    last_price = df["Close"].iloc[-1]
                    max_price = df["Close"].max()
                    drawdown = (last_price / max_price - 1) * 100
                    sma50 = df["SMA50"].iloc[-1]
                    sma200 = df["SMA200"].iloc[-1]
                    rsi = df["RSI"].iloc[-1]
                    alerts.append(
                        f"📉 ETF {ticker} ({etf_names.get(ticker, ticker)}) en señal de compra:\n"
                        f"Precio: {last_price:.2f}\n"
                        f"Drawdown: {drawdown:.2f}%\n"
                        f"SMA50: {sma50:.2f}, SMA200: {sma200:.2f}\n"
                        f"RSI: {rsi:.2f}"
                    )
                
                # Prepare data for summary table
                last_price = df["Close"].iloc[-1]
                max_price = df["Close"].max()
                drawdown = (last_price / max_price - 1) * 100
                sma50 = df["SMA50"].iloc[-1]
                sma200 = df["SMA200"].iloc[-1]
                rsi = df["RSI"].iloc[-1]

                # Use the ETF name in the data dictionary key
                display_name = f"{ticker} ({etf_names.get(ticker, ticker)})"
                data[display_name] = {
                    "Precio actual": round(last_price, 2),
                    "Máximo histórico": round(max_price, 2),
                    "Drawdown (%)": round(drawdown, 2),
                    "SMA50": round(sma50, 2),
                    "SMA200": round(sma200, 2),
                    "RSI": round(rsi, 2),
                    "Candidato compra": check_buy_signal(df)
                }

                # Run backtest
                initial_capital = 100000
                final_capital = run_backtest(df.copy())
                results_backtest[display_name] = {
                    "Rendimiento (%)": (final_capital / initial_capital - 1) * 100
                }
            
            # Move the summary tables to the top
            st.subheader("📊 Resultados del Screener")
            df_out = pd.DataFrame(data).T
            st.dataframe(df_out)

            st.subheader("🧪 Resultados del Backtesting")
            df_backtest = pd.DataFrame(results_backtest).T
            st.dataframe(df_backtest.style.format({"Rendimiento (%)": "{:.2f}%"}))
            
            # Send Telegram alert if there are candidates
            if alerts:
                for msg in alerts:
                    send_telegram(msg)
                st.success("✅ Alertas enviadas por Telegram")
            else:
                st.info("Ningún ETF cumple las condiciones hoy.")

            # ---------------- GRÁFICOS ----------------
            st.subheader("📈 Gráficos de ETFs")
            for ticker in etfs:
                # Handle single ticker data
                if len(etfs) > 1:
                    df = df_full['Close'][ticker].dropna().to_frame()
                    df.columns = ["Close"]
                    if 'Open' in df_full.columns:
                        df['Open'] = df_full['Open'][ticker].dropna()
                else:
                    df = df_full.dropna().copy()

                if df.empty or len(df) < 200:
                    continue
                
                df["SMA50"] = df["Close"].rolling(50).mean()
                df["SMA200"] = df["Close"].rolling(200).mean()
                df["RSI"] = compute_rsi(df["Close"])

                st.subheader(f"📈 {ticker} ({etf_names.get(ticker, ticker)})")
                col1, col2 = st.columns([2,1])

                # Filter data for the last 2 years
                df_last_2_years = df.tail(365 * 2)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_last_2_years.index, y=df_last_2_years["Close"], name="Precio", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df_last_2_years.index, y=df_last_2_years["SMA50"], name="SMA50", line=dict(color="orange")))
                fig.add_trace(go.Scatter(x=df_last_2_years.index, y=df_last_2_years["SMA200"], name="SMA200", line=dict(color="red")))
                fig.update_layout(title=f"{ticker} ({etf_names.get(ticker, ticker)}) - Precio y Medias (Últimos 2 años)", xaxis_title="Fecha", yaxis_title="Precio")
                col1.plotly_chart(fig, use_container_width=True)

                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df_last_2_years.index, y=df_last_2_years["RSI"], name="RSI", line=dict(color="purple")))
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title=f"{ticker} ({etf_names.get(ticker, ticker)}) - RSI (Últimos 2 años)", xaxis_title="Fecha", yaxis_title="RSI (14)")
                col2.plotly_chart(fig_rsi, use_container_width=True)
