import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="QuantEdge Risk Terminal", layout="wide")

st.title("🛡️ QuantEdge: Dynamic Risk & Backtesting Engine")
st.caption("Statistical Directional Models & Automated Volatility Controls")

# Input Sanitization (Prevents Injection Attacks)
raw_input = st.sidebar.text_input("NSE Stock Symbol:", "JPPOWER").strip().upper()
sanitized_symbol = "".join(e for e in raw_input if e.isalnum())
ticker_symbol = f"{sanitized_symbol}.NS" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio:", 1.0, 4.0, 2.0, 0.5)
atr_multiplier = st.sidebar.slider("Stop-Loss ATR Multiplier:", 1.0, 3.0, 1.5, 0.25)

@st.cache_data(ttl=300)
def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y", interval="1d")
    return df, stock.info if not df.empty else (None, None)

df, info = fetch_data(ticker_symbol)

if df is None or df.empty:
    st.error("Invalid symbol or no market data found. Please enter a valid NSE ticker.")
else:
    # Feature Engineering
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    clean_df = df.dropna().copy()
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'ATR']
    
    X = clean_df[features]
    y = clean_df['Target_Direction']

    # Backtest Split
    split = int(len(clean_df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    accuracy = (model.predict(X_test) == y_test).mean() * 100
    prob_up = model.predict_proba(X.tail(1))[0][1] * 100

    current_price = float(df['Close'].iloc[-1])
    atr_val = float(df['ATR'].iloc[-1])
    stop_loss = current_price - (atr_val * atr_multiplier)
    take_profit = current_price + ((current_price - stop_loss) * risk_reward_ratio)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"₹{current_price:.2f}")
    m2.metric("Directional Edge", "BULLISH" if prob_up >= 55 else ("BEARISH" if prob_up <= 45 else "NEUTRAL"), f"{prob_up:.1f}% Confidence")
    m3.metric("Backtest Accuracy", f"{accuracy:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.error(f"**Automated Stop-Loss:** ₹{stop_loss:.2f}")
    c2.success(f"**Take-Profit Target:** ₹{take_profit:.2f}")
