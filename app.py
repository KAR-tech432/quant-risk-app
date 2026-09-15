import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Page Config
st.set_page_config(page_title="QuantEdge Risk Terminal", layout="wide")

st.title("🛡️ QuantEdge: Dynamic Risk & Backtesting Engine")
st.caption("Statistical Directional Models & Automated Volatility Controls")

# Input Sanitization
raw_input = st.sidebar.text_input("NSE Stock Symbol:", "JPPOWER").strip().upper()
sanitized_symbol = "".join(e for e in raw_input if e.isalnum())
ticker_symbol = f"{sanitized_symbol}.NS" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio:", 1.0, 4.0, 2.0, 0.5)
atr_multiplier = st.sidebar.slider("Stop-Loss ATR Multiplier:", 1.0, 3.0, 1.5, 0.25)

# 1. Cache Data Fetching (5-minute TTL to prevent API rate limits)
@st.cache_data(ttl=300)
def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y", interval="1d")
    return df if not df.empty else None

# 2. Cache ML Model Training (Prevents CPU throttling on slider moves)
@st.cache_resource
def train_xgboost(clean_data):
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'ATR']
    X = clean_data[features]
    y = clean_data['Target_Direction']

    # 80/20 Train-Test Split
    split = int(len(clean_data) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)

    accuracy = (model.predict(X_test) == y_test).mean() * 100
    return model, accuracy, features

# Fetch Data
df = fetch_data(ticker_symbol)

if df is None or df.empty:
    st.error("Invalid symbol or no market data found. Please enter a valid NSE ticker.")
else:
    # Feature Engineering
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Binary Classification Target: 1 if Next Close > Current Close else 0
    df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    clean_df = df.dropna().copy()
    
    # Run Cached Model
    model, accuracy, features = train_xgboost(clean_df)

    latest_features = clean_df[features].tail(1)
    prob_up = model.predict_proba(latest_features)[0][1] * 100

    current_price = float(df['Close'].iloc[-1])
    atr_val = float(df['ATR'].iloc[-1])
    
    # Volatility Risk Levels
    stop_loss = current_price - (atr_val * atr_multiplier)
    take_profit = current_price + ((current_price - stop_loss) * risk_reward_ratio)

    # Top Metric Bar
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"₹{current_price:.2f}")
    
    if prob_up >= 55:
        signal = "BULLISH (BUY EDGE)"
    elif prob_up <= 45:
        signal = "BEARISH (SELL EDGE)"
    else:
        signal = "NEUTRAL (NO EDGE)"
        
    m2.metric("Directional Edge", signal, f"{prob_up:.1f}% Confidence")
    m3.metric("Backtest Accuracy", f"{accuracy:.1f}%")

    st.markdown("---")
    
    # Risk Management Cards
    c1, c2, c3 = st.columns(3)
    max_risk_per_share = current_price - stop_loss
    
    with c1:
        st.error(f"**Automated Stop-Loss:** ₹{stop_loss:.2f}")
        st.caption(f"Risk per share: ₹{max_risk_per_share:.2f} ({atr_multiplier}x ATR)")
    
    with c2:
        st.success(f"**Take-Profit Target:** ₹{take_profit:.2f}")
        st.caption(f"Target reward per share: ₹{take_profit - current_price:.2f}")
        
    with c3:
        st.info(f"**Risk-to-Reward Ratio:** 1 : {risk_reward_ratio}")
        portfolio_val = st.number_input("Capital to Risk (₹):", value=10000, step=1000)
        max_shares = int(portfolio_val / max_risk_per_share) if max_risk_per_share > 0 else 0
        st.caption(f"Max Recommended Position: **{max_shares} shares**")

    st.markdown("---")

    # Interactive Charting
    st.subheader("📊 Price Action & Risk Overlay")
    
    hist_df = df.tail(90)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist_df.index,
        open=hist_df['Open'],
        high=hist_df['High'],
        low=hist_df['Low'],
        close=hist_df['Close'],
        name="Price"
    ))

    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#FF9800', width=1)))
    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#2196F3', width=1)))

    fig.add_hline(y=stop_loss, line_dash="dash", line_color="#FF5252", annotation_text="Stop-Loss Level")
    fig.add_hline(y=take_profit, line_dash="dash", line_color="#00E676", annotation_text="Take-Profit Level")

    fig.update_layout(template="plotly_dark", height=550, xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)
