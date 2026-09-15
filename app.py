import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel

# Page Config
st.set_page_config(page_title="QuantEdge Institutional Terminal", layout="wide")

st.title("🛡️ QuantEdge: Institutional Alpha Terminal")
st.caption("Advanced Market Microstructure, Volume Profiles & Statistical Reversion Models")

# Sidebar Controls
raw_input = st.sidebar.text_input("NSE Stock Symbol:", "JPPOWER").strip().upper()
sanitized_symbol = "".join(e for e in raw_input if e.isalnum())
ticker_symbol = f"{sanitized_symbol}.NS" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio:", 1.0, 4.0, 2.0, 0.5)
atr_multiplier = st.sidebar.slider("Stop-Loss ATR Multiplier:", 1.0, 3.0, 1.5, 0.25)

# Caching Layer
@st.cache_data(ttl=300)
def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y", interval="1d")
    return df if not df.empty else None

@st.cache_resource
def train_xgboost(clean_data):
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'ATR', 'Z_Score', 'OBV_Slope']
    X = clean_data[features]
    y = clean_data['Target_Direction']

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

df = fetch_data(ticker_symbol)

if df is None or df.empty:
    st.error("Invalid symbol or no market data found. Please enter a valid NSE ticker.")
else:
    # -------------------------------------------------------------
    # ADVANCED QUANT METRICS & FEATURE ENGINEERING
    # -------------------------------------------------------------
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # 1. Z-Score Mean Reversion (Distance from 50-day Mean in Standard Deviations)
    df['Rolling_Mean'] = df['Close'].rolling(50).mean()
    df['Rolling_Std'] = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']

    # 2. Institutional Smart Money Flow (OBV Slope over 10 Days)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = obv.diff(10)

    # 3. Volatility Squeeze Metric (Bollinger Inside Keltner Channels)
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['KC_Upper'] = kc.keltner_channel_hband()
    df['KC_Lower'] = kc.keltner_channel_lband()
    df['Squeeze_Active'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])

    # Target Definition
    df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    clean_df = df.dropna().copy()

    # Train Model
    model, accuracy, features = train_xgboost(clean_df)
    latest_features = clean_df[features].tail(1)
    prob_up = model.predict_proba(latest_features)[0][1] * 100

    current_price = float(df['Close'].iloc[-1])
    atr_val = float(df['ATR'].iloc[-1])
    z_score_val = float(clean_df['Z_Score'].iloc[-1])
    obv_slope_val = float(clean_df['OBV_Slope'].iloc[-1])
    squeeze_val = bool(clean_df['Squeeze_Active'].iloc[-1])

    # Dynamic Risk Levels
    stop_loss = current_price - (atr_val * atr_multiplier)
    take_profit = current_price + ((current_price - stop_loss) * risk_reward_ratio)

    # -------------------------------------------------------------
    # INSTITUTIONAL METRICS DASHBOARD
    # -------------------------------------------------------------
    st.subheader("⚡ Institutional Alpha Metrics")
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Price Z-Score", f"{z_score_val:+.2f} σ", 
              "Overbought (> +2)" if z_score_val > 2 else ("Oversold (< -2)" if z_score_val < -2 else "Fair Value"))

    m2.metric("Smart Money Flow", "ACCUMULATION" if obv_slope_val > 0 else "DISTRIBUTION", 
              f"{obv_slope_val:,.0f} Vol Delta")

    m3.metric("Volatility Squeeze", "FIRE READY" if squeeze_val else "NORMAL VOLATILITY", 
              "Pre-Breakout Squeeze" if squeeze_val else "Standard Expansion")

    m4.metric("Model Edge", f"{prob_up:.1f}% Bullish", f"Model Acc: {accuracy:.1f}%")

    st.markdown("---")

    # Risk Metrics
    c1, c2, c3 = st.columns(3)
    max_risk = current_price - stop_loss
    with c1:
        st.error(f"**Automated Stop-Loss:** ₹{stop_loss:.2f}")
        st.caption(f"Risk per share: ₹{max_risk:.2f}")
    with c2:
        st.success(f"**Take-Profit Target:** ₹{take_profit:.2f}")
        st.caption(f"Reward per share: ₹{take_profit - current_price:.2f}")
    with c3:
        st.info(f"**Risk-Reward Ratio:** 1 : {risk_reward_ratio}")
        capital = st.number_input("Capital to Risk (₹):", value=10000, step=1000)
        st.caption(f"Max Position Size: **{int(capital / max_risk) if max_risk > 0 else 0} shares**")

    st.markdown("---")

    # -------------------------------------------------------------
    # ADVANCED CHART: CANDLESTICKS + VOLUME PROFILE (POC)
    # -------------------------------------------------------------
    st.subheader("📊 Price Action & Point of Control (POC)")

    hist_df = df.tail(90)
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist_df.index,
        open=hist_df['Open'], high=hist_df['High'],
        low=hist_df['Low'], close=hist_df['Close'],
        name="Price"
    ))

    # Point of Control (Volume Profile calculation)
    price_bins = pd.cut(hist_df['Close'], bins=15)
    volume_profile = hist_df.groupby(price_bins, observed=False)['Volume'].sum()
    poc_bin = volume_profile.idxmax()
    poc_price = (poc_bin.left + poc_bin.right) / 2

    # Plot POC & Risk Lines
    fig.add_hline(y=poc_price, line_dash="solid", line_color="#E040FB", line_width=2,
                  annotation_text=f"Point of Control (POC): ₹{poc_price:.2f}")
    fig.add_hline(y=stop_loss, line_dash="dash", line_color="#FF5252", annotation_text="Stop-Loss")
    fig.add_hline(y=take_profit, line_dash="dash", line_color="#00E676", annotation_text="Take-Profit")

    fig.update_layout(template="plotly_dark", height=550, xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)
