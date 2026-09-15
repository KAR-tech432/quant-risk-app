import re
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Page Configuration
st.set_page_config(page_title="QuantEdge Risk Terminal", layout="wide")

st.title("🛡️ QuantEdge: Dynamic Risk & Backtesting Engine")
st.caption("Statistical Directional Models & Automated Volatility Controls")

# --- SECURE INPUT SANITIZATION & VALIDATION ---
st.sidebar.header("Asset & Strategy Parameters")
raw_input = st.sidebar.text_input("NSE Stock Symbol:", "JPPOWER").strip().upper()

# Strict Regex Whitelist: Allows only uppercase letters and numbers (1-10 chars)
sanitized_symbol = re.sub(r'[^A-Z0-9]', '', raw_input)[:10]

if not sanitized_symbol:
    st.error("Invalid ticker format. Please enter a valid stock symbol (e.g., RELIANCE, JPPOWER).")
    st.stop()

# Auto-append NSE suffix
ticker_symbol = f"{sanitized_symbol}.NS" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio:", 1.0, 4.0, 2.0, 0.5)
atr_multiplier = st.sidebar.slider("Stop-Loss ATR Multiplier:", 1.0, 3.0, 1.5, 0.25)


# --- CACHED DATA FETCHING ENGINE ---
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(symbol: str):
    """Fetches historical market data with caching to prevent excessive API calls."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="2y", interval="1d")
        if df.empty or len(df) < 60:
            return None, None
        return df, stock.info
    except Exception:
        return None, None


# --- CACHED MACHINE LEARNING TRAINER ---
@st.cache_resource(show_spinner=False)
def train_cached_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Caches model in memory to eliminate CPU re-training on slider/widget updates.
    Prevents Streamlit Cloud CPU throttling.
    """
    model = XGBClassifier(
        n_estimators=50,       # Optimized for low CPU footprint
        learning_rate=0.03,
        max_depth=3,           # Prevents overfitting
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


# Execute Data Fetch
with st.spinner(f"Loading market structure for {sanitized_symbol}..."):
    df, info = fetch_market_data(ticker_symbol)

if df is None or df.empty:
    st.error(f"Could not retrieve market data for symbol '{sanitized_symbol}'. Verify the ticker on NSE.")
else:
    # Feature Engineering
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = MACD(df['Close']).macd()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Target: 1 if next session close is higher than current close
    df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    clean_df = df.dropna().copy()
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR']
    
    X = clean_df[features]
    y = clean_df['Target_Direction']

    # Train/Test Backtest Split (80% Train, 20% Out-of-Sample Test)
    split = int(len(clean_df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    # Fetch Cached Model Instance
    model = train_cached_xgboost_model(X_train, y_train)

    # Calculate Out-of-Sample Historical Accuracy
    test_predictions = model.predict(X_test)
    backtest_accuracy = (test_predictions == y_test).mean() * 100

    # Current Live Directional Edge
    latest_features = X.tail(1)
    up_probability = model.predict_proba(latest_features)[0][1] * 100
    
    if up_probability >= 55:
        signal = "BULLISH (BUY EDGE)"
    elif up_probability <= 45:
        signal = "BEARISH (SELL EDGE)"
    else:
        signal = "NEUTRAL (NO EDGE)"

    # Live Price Metrics
    current_price = float(df['Close'].iloc[-1])
    atr_val = float(df['ATR'].iloc[-1])
    
    # Automated Risk Boundaries
    stop_loss = current_price - (atr_val * atr_multiplier)
    take_profit = current_price + ((current_price - stop_loss) * risk_reward_ratio)
    risk_per_share = current_price - stop_loss

    # Render Dashboard Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Market Price", f"₹{current_price:.2f}")
    m2.metric("Directional Edge Signal", signal, f"{up_probability:.1f}% Confidence")
    m3.metric("Backtest Win Rate", f"{backtest_accuracy:.1f}%", "Out-of-Sample Test")
    m4.metric("14-Day ATR (Volatility)", f"₹{atr_val:.2f}")

    st.markdown("---")

    # Risk Management & Position Sizing Section
    st.subheader("🎯 Position Sizing & Risk Rules")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.error(f"**Automated Stop-Loss:** ₹{stop_loss:.2f}")
        st.caption(f"Max Loss per Share: ₹{risk_per_share:.2f} ({atr_multiplier}x ATR)")
        
    with c2:
        st.success(f"**Take-Profit Target:** ₹{take_profit:.2f}")
        st.caption(f"Target Reward per Share: ₹{take_profit - current_price:.2f}")
        
    with c3:
        st.info(f"**Target Risk-to-Reward:** 1 : {risk_reward_ratio}")
        capital_to_risk = st.number_input("Enter Capital Risk Allocation (₹):", value=10000, step=1000)
        recommended_shares = int(capital_to_risk / risk_per_share) if risk_per_share > 0 else 0
        st.caption(f"Recommended Position: **{recommended_shares} shares**")

    st.markdown("---")

    # Interactive Chart Rendering
    st.subheader("📊 Price Action & Automated Risk Lines")
    
    chart_df = df.tail(90)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'],
        high=chart_df['High'],
        low=chart_df['Low'],
        close=chart_df['Close'],
        name="Price Action"
    ))

    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#FF9800', width=1)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#2196F3', width=1)))

    fig.add_hline(y=stop_loss, line_dash="dash", line_color="#FF5252", annotation_text="Stop-Loss")
    fig.add_hline(y=take_profit, line_dash="dash", line_color="#00E676", annotation_text="Take-Profit")

    fig.update_layout(template="plotly_dark", height=550, xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)
