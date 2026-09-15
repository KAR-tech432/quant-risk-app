import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel

# -------------------------------------------------------------
# PAGE CONFIGURATION & MOBILE STYLING
# -------------------------------------------------------------
st.set_page_config(page_title="QuantEdge 360° Terminal", layout="wide")

st.markdown("""
<style>
    /* Compact Margins for Mobile & Desktop */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    
    /* Compact Mobile-Friendly Verdict Banner */
    .decision-banner {
        padding: 6px 10px;
        border-radius: 6px;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 12px;
        line-height: 1.2;
    }
    
    .decision-subtext {
        font-size: 11px;
        font-weight: 400;
        opacity: 0.95;
        display: block;
        margin-top: 3px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.header("🎯 Asset Settings")
exchange = st.sidebar.radio("Select Exchange:", ["NSE (.NS)", "BSE (.BO)"])
raw_input = st.sidebar.text_input("Stock Symbol:", "JPPOWER").strip().upper()
sanitized_symbol = "".join(e for e in raw_input if e.isalnum())

suffix = ".NS" if exchange == "NSE (.NS)" else ".BO"
ticker_symbol = f"{sanitized_symbol}{suffix}" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio:", 1.0, 4.0, 2.0, 0.5)
atr_multiplier = st.sidebar.slider("Stop-Loss ATR Multiplier:", 1.0, 3.0, 1.5, 0.25)
capital_allocated = st.sidebar.number_input("Capital to Risk (₹):", value=50000, step=5000)

# -------------------------------------------------------------
# CACHED DATA FETCHING LAYER
# -------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_stock_master(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y", interval="1d")
    
    info = {}
    try:
        info = ticker.info
    except Exception:
        info = {}
        
    financials = ticker.quarterly_financials if hasattr(ticker, 'quarterly_financials') else pd.DataFrame()
    return df, info, financials

@st.cache_resource
def train_xgboost(clean_data):
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'ATR', 'Z_Score', 'OBV_Slope']
    X = clean_data[features]
    y = clean_data['Target_Direction']

    split = int(len(clean_data) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=4, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)

    accuracy = (model.predict(X_test) == y_test).mean() * 100
    return model, accuracy, features

# Load Data
df, info, financials = fetch_stock_master(ticker_symbol)

if df is None or df.empty:
    st.error(f"Could not load market data for **{ticker_symbol}**. Verify symbol or exchange configuration.")
else:
    # --- GUARANTEED STOCK NAME FETCHING ---
    company_name = info.get('longName') or info.get('shortName') or ticker_symbol
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    summary = info.get('longBusinessSummary', 'No detailed business summary available.')

    # -------------------------------------------------------------
    # FEATURE ENGINEERING & QUANT METRICS
    # -------------------------------------------------------------
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Z-Score
    df['Rolling_Mean'] = df['Close'].rolling(50).mean()
    df['Rolling_Std'] = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']

    # OBV Slope
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = obv.diff(10)

    # Volatility Squeeze
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
    df['Squeeze_Active'] = (bb.bollinger_hband() < kc.keltner_channel_hband()) & (bb.bollinger_lband() > kc.keltner_channel_lband())

    # Target Setup
    df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    clean_df = df.dropna().copy()

    # Model Execution
    model, accuracy, features = train_xgboost(clean_df)
    latest_features = clean_df[features].tail(1)
    prob_up = model.predict_proba(latest_features)[0][1] * 100

    # Key Values
    curr_price = float(info.get('currentPrice', df['Close'].iloc[-1]))
    prev_close = float(info.get('previousClose', df['Close'].iloc[-2]))
    price_change = curr_price - prev_close
    pct_change = (price_change / prev_close) * 100

    atr_val = float(df['ATR'].iloc[-1])
    z_score_val = float(clean_df['Z_Score'].iloc[-1])
    obv_slope_val = float(clean_df['OBV_Slope'].iloc[-1])
    squeeze_val = bool(clean_df['Squeeze_Active'].iloc[-1])
    rsi_val = float(clean_df['RSI'].iloc[-1])
    sma_200_val = float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else curr_price
    pe_ratio = info.get('trailingPE', None)

    # Risk Calculations
    stop_loss = curr_price - (atr_val * atr_multiplier)
    risk_per_share = curr_price - stop_loss
    take_profit = curr_price + (risk_per_share * risk_reward_ratio)
    max_shares = int(capital_allocated / risk_per_share) if risk_per_share > 0 else 0

    # -------------------------------------------------------------
    # SCORING ENGINE
    # -------------------------------------------------------------
    total_bullish_score = 0
    if curr_price > sma_200_val: total_bullish_score += 2
    if obv_slope_val > 0: total_bullish_score += 2
    if prob_up >= 55.0: total_bullish_score += 2
    if (rsi_val >= 50.0 and rsi_val <= 70.0) or (rsi_val <= 30.0): total_bullish_score += 1
    if pe_ratio is not None and pe_ratio < 25.0: total_bullish_score += 1

    if total_bullish_score >= 6 and z_score_val < 1.8:
        action_decision = "ACCUMULATE (STRONG BUY)"
        banner_color = "#00C853"
        action_summary = "Technical trend, money flow, and model probability are aligned."
    elif total_bullish_score < 4 or z_score_val > 2.2:
        action_decision = "SHORT / REDUCE (BEARISH)"
        banner_color = "#FF1744"
        action_summary = "Distribution pressure present or price expansion is overextended."
    elif total_bullish_score >= 4:
        action_decision = "HOLD (NEUTRAL BIASED)"
        banner_color = "#FF9100"
        action_summary = "Macro trend intact; momentum suggests holding existing positions."
    else:
        action_decision = "EXIT / AVOID (NO EDGE)"
        banner_color = "#D500F9"
        action_summary = "Conflicting signals between volume and technical structure."

    # -------------------------------------------------------------
    # 1. NATIVE STREAMLIT HEADER (ALWAYS VISIBLE & MOBILE RESPONSIVE)
    # -------------------------------------------------------------
    st.subheader(f"📈 {company_name} ({ticker_symbol})")
    st.caption(f"**Sector:** {sector} | **Industry:** {industry}")

    st.markdown(f"""
    <div class="decision-banner" style="background-color: {banner_color}; color: white;">
        VERDICT: {action_decision}
        <span class="decision-subtext">{action_summary}</span>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. MICROSTRUCTURE & INSTITUTIONAL FLOW
    # -------------------------------------------------------------
    st.subheader("⚡ Microstructure & Institutional Flow")
    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Price Z-Score", f"{z_score_val:+.2f} σ", 
              "Oversold" if z_score_val < -2 else ("Overbought" if z_score_val > 2 else "Fair Value"))
    q2.metric("Smart Money Flow", "ACCUMULATION" if obv_slope_val > 0 else "DISTRIBUTION", f"{obv_slope_val:,.0f} Delta")
    q3.metric("Volatility Squeeze", "FIRE READY" if squeeze_val else "EXPANDED", "Consolidation" if squeeze_val else "Active Trend")
    q4.metric("XGBoost Edge", f"{prob_up:.1f}% Bullish", f"Acc: {accuracy:.1f}%")

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. LIVE PRICE SNAPSHOT
    # -------------------------------------------------------------
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Live Price", f"₹{curr_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    h2.metric("Day High", f"₹{info.get('dayHigh', df['High'].iloc[-1]):.2f}")
    h3.metric("Day Low", f"₹{info.get('dayLow', df['Low'].iloc[-1]):.2f}")
    mcap = info.get('marketCap', 0)
    h4.metric("Market Cap", f"₹{mcap/1e7:,.0f} Cr" if mcap else "N/A")
    h5.metric("Volume Today", f"{int(info.get('volume', df['Volume'].iloc[-1])):,.0f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. FUNDAMENTALS & RISK CONTROL
    # -------------------------------------------------------------
    st.subheader("🏛️ Fundamentals & Risk Control")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Trailing P/E", f"{info.get('trailingPE', 'N/A')}")
    f2.metric("Price-to-Book", f"{info.get('priceToBook', 'N/A')}")
    f3.metric("Automated Stop-Loss", f"₹{stop_loss:.2f}", f"{atr_multiplier}x ATR")
    f4.metric("Take-Profit Target", f"₹{take_profit:.2f}", f"RR Ratio {risk_reward_ratio}:1")
    f5.metric("Max Share Size", f"{max_shares} Shares", f"Risk ₹{capital_allocated:,.0f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. CHARTS & STATEMENTS
    # -------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Price Action & POC", "📜 Quarterly Financials"])

    with tab1:
        hist_df = df.tail(120)
        price_bins = pd.cut(hist_df['Close'], bins=15)
        volume_profile = hist_df.groupby(price_bins, observed=False)['Volume'].sum()
        poc_bin = volume_profile.idxmax()
        poc_price = (poc_bin.left + poc_bin.right) / 2

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist_df.index, open=hist_df['Open'], high=hist_df['High'],
            low=hist_df['Low'], close=hist_df['Close'], name="Price"
        ))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#FF9800', width=1)))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#2196F3', width=1)))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_200'], mode='lines', name='SMA 200', line=dict(color='#E91E63', width=1.5)))

        fig.add_hline(y=poc_price, line_dash="solid", line_color="#E040FB", line_width=2, annotation_text=f"POC: ₹{poc_price:.2f}")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="#FF5252", annotation_text=f"Stop-Loss (₹{stop_loss:.2f})")
        fig.add_hline(y=take_profit, line_dash="dash", line_color="#00E676", annotation_text=f"Target (₹{take_profit:.2f})")

        fig.update_layout(template="plotly_dark", height=420, xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if isinstance(financials, pd.DataFrame) and not financials.empty:
            st.dataframe(financials, use_container_width=True)
        else:
            st.info("Quarterly financials data not available for this ticker.")

    with st.expander(f"ℹ️ Business Profile: {company_name}"):
        st.write(summary)
