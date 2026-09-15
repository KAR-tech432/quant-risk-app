import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="QuantEdge Risk & Strategy Terminal", layout="wide")

st.title("🛡️ QuantEdge: Probabilistic Strategy & Risk Terminal")
st.caption("Engineered for Risk Controls, Backtesting & Directional Signal Precision")

# Sidebar Controls
st.sidebar.header("Asset & Strategy Parameters")
stock_input = st.sidebar.text_input("NSE Stock Symbol:", "JPPOWER").strip().upper()
ticker_symbol = stock_input if stock_input.endswith((".NS", ".BO")) else f"{stock_input}.NS"

risk_reward_ratio = st.sidebar.slider("Target Risk-to-Reward Ratio:", 1.0, 4.0, 2.0, 0.5)
atr_multiplier = st.sidebar.slider("Stop-Loss ATR Multiplier:", 1.0, 3.0, 1.5, 0.25)

@st.cache_data(ttl=300)
def fetch_and_process_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y", interval="1d")
    
    if df.empty:
        return None, None
    
    info = stock.info
    
    # Feature Engineering
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = MACD(df['Close']).macd()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Direction Target
    df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df, info

df, info = fetch_and_process_data(ticker_symbol)

if df is None or df.empty:
    st.error(f"Unable to load data for '{stock_input}'. Please check if the symbol is listed on NSE.")
else:
    current_price = float(df['Close'].iloc[-1])
    atr_val = float(df['ATR'].iloc[-1])
    
    # Risk Management Rules
    stop_loss = current_price - (atr_val * atr_multiplier)
    take_profit = current_price + ((current_price - stop_loss) * risk_reward_ratio)
    max_risk_per_share = current_price - stop_loss
    
    # ML Feature Prep
    clean_df = df.dropna().copy()
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR']
    
    X = clean_df[features]
    y = clean_df['Target_Direction']

    # Historical Backtest Split
    split = int(len(clean_df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    
    # Train Model
    model = XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Backtest Accuracy
    test_preds = model.predict(X_test)
    accuracy = (test_preds == y_test).mean() * 100
    
    # Current Signal
    latest_features = X.tail(1)
    up_probability = model.predict_proba(latest_features)[0][1] * 100
    signal = "BULLISH (BUY EDGE)" if up_probability >= 55 else ("BEARISH (SELL EDGE)" if up_probability <= 45 else "NEUTRAL (NO EDGE)")

    # Dashboard Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"₹{current_price:.2f}")
    m2.metric("Signal Direction", signal, f"{up_probability:.1f}% Confidence")
    m3.metric("Historical Win Rate", f"{accuracy:.1f}%", "Out-of-Sample Backtest")
    m4.metric("14-Day Volatility (ATR)", f"₹{atr_val:.2f}")

    st.markdown("---")

    # Position Sizing
    st.subheader("🎯 Position Sizing & Risk Management")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.error(f"**Stop-Loss:** ₹{stop_loss:.2f}")
        st.caption(f"Risk per share: ₹{max_risk_per_share:.2f} ({atr_multiplier}x ATR)")
    
    with c2:
        st.success(f"**Take-Profit Target:** ₹{take_profit:.2f}")
        st.caption(f"Target reward per share: ₹{take_profit - current_price:.2f}")
        
    with c3:
        st.info(f"**Risk-to-Reward Ratio:** 1 : {risk_reward_ratio}")
        portfolio_val = st.number_input("Enter Capital to Risk (₹):", value=10000, step=1000)
        max_shares = int(portfolio_val / max_risk_per_share) if max_risk_per_share > 0 else 0
        st.caption(f"Max Recommended Position: **{max_shares} shares**")

    st.markdown("---")

    # Interactive Chart
    st.subheader("📊 Price Action & Risk Zones")
    
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