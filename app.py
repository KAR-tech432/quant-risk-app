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
# 1. PAGE CONFIGURATION & HIGH-VISIBILITY DAYLIGHT STYLING
# -------------------------------------------------------------
st.set_page_config(page_title="QuantEdge 360° Terminal", layout="wide")

st.markdown("""
<style>
    /* Reset padding for maximum screen real estate */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    
    /* High-Contrast Dark Slate Terminal Card */
    .terminal-card {
        background-color: #161922;
        border: 1px solid #33394B;
        border-radius: 8px;
        padding: 12px 16px;
        min-height: 72px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.35);
    }
    
    .stock-title-main {
        font-size: 16px;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.2;
        margin: 0;
        word-break: break-word;
    }
    
    .stock-symbol-badge {
        color: #00E5FF !important;
        font-weight: 700;
        font-size: 14px;
    }

    .stock-meta-info {
        font-size: 12px;
        color: #E0E6ED !important;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Solid High-Visibility Verdict Badge with Outdoor Readability */
    .verdict-box-solid {
        border-radius: 6px;
        padding: 10px 14px;
        text-align: right;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        transition: background-color 0.3s ease;
    }
    
    .verdict-title {
        font-size: 14px;
        font-weight: 900;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #FFFFFF !important;
    }
    
    .verdict-desc {
        font-size: 11px;
        color: #FFFFFF !important;
        font-weight: 500;
        opacity: 0.95;
        margin-top: 2px;
        line-height: 1.2;
    }

    @media (max-width: 640px) {
        .verdict-box-solid {
            text-align: left;
            margin-top: 6px;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION CONTROLS
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
# 3. CACHED DATA FETCHING & HIGH-ACCURACY MODEL TRAINING
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
    df_feat = clean_data.copy()
    
    # Advanced Multi-Timeframe Feature Engineering
    df_feat['Ret_1D'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
    df_feat['Ret_5D'] = np.log(df_feat['Close'] / df_feat['Close'].shift(5))
    df_feat['Ret_20D'] = np.log(df_feat['Close'] / df_feat['Close'].shift(20))
    
    # Volatility & Volume Spread Indicators
    df_feat['HL_Spread'] = (df_feat['High'] - df_feat['Low']) / df_feat['Close']
    df_feat['Vol_ZScore'] = (df_feat['Volume'] - df_feat['Volume'].rolling(20).mean()) / df_feat['Volume'].rolling(20).std()
    
    # Lagged Momentum Dynamics
    df_feat['RSI_Lag1'] = df_feat['RSI'].shift(1)
    df_feat['RSI_Slope'] = df_feat['RSI'] - df_feat['RSI_Lag1']
    
    # Noise-Filtered Target Setup (0.75% threshold to eliminate coin-flip predictions)
    future_return = (df_feat['Close'].shift(-1) - df_feat['Close']) / df_feat['Close']
    df_feat['Target_Direction'] = np.where(future_return > 0.0075, 1, 0)
    
    df_feat = df_feat.dropna()
    
    feature_cols = [
        'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'ATR', 
        'Z_Score', 'OBV_Slope', 'Ret_1D', 'Ret_5D', 'Ret_20D', 
        'HL_Spread', 'Vol_ZScore', 'RSI_Slope'
    ]
    
    X = df_feat[feature_cols]
    y = df_feat['Target_Direction']

    # Chronological Time-Series Split (80% Train / 20% Out-of-Sample Test)
    split = int(len(df_feat) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    # Optimized Hyperparameters with Regularization
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.015,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    accuracy = (model.predict(X_test) == y_test).mean() * 100
    
    return model, accuracy, feature_cols

# Load Asset Data
df, info, financials = fetch_stock_master(ticker_symbol)

if df is None or df.empty:
    st.error(f"Could not load market data for **{ticker_symbol}**. Verify symbol or exchange configuration.")
else:
    company_name = info.get('longName') or info.get('shortName') or ticker_symbol
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    summary = info.get('longBusinessSummary', 'No detailed business summary available.')

    # -------------------------------------------------------------
    # 4. QUANTITATIVE FEATURE ENGINEERING
    # -------------------------------------------------------------
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Valuation Z-Score
    df['Rolling_Mean'] = df['Close'].rolling(50).mean()
    df['Rolling_Std'] = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']

    # Smart Money OBV Slope
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = obv.diff(10)

    # Volatility Squeeze State
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
    df['Squeeze_Active'] = (bb.bollinger_hband() < kc.keltner_channel_hband()) & (bb.bollinger_lband() > kc.keltner_channel_lband())

    clean_df = df.dropna().copy()

    # Model Execution
    model, accuracy, features = train_xgboost(clean_df)
    latest_features = clean_df[features].tail(1)
    prob_up = model.predict_proba(latest_features)[0][1] * 100

    # Real-Time Price Analytics
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

    # Automated Risk Parameters
    stop_loss = curr_price - (atr_val * atr_multiplier)
    risk_per_share = curr_price - stop_loss
    take_profit = curr_price + (risk_per_share * risk_reward_ratio)
    max_shares = int(capital_allocated / risk_per_share) if risk_per_share > 0 else 0

    # -------------------------------------------------------------
    # 5. DYNAMIC VERDICT & COLOR DECISION ENGINE
    # -------------------------------------------------------------
    total_bullish_score = 0
    if curr_price > sma_200_val: total_bullish_score += 2
    if obv_slope_val > 0: total_bullish_score += 2
    if prob_up >= 55.0: total_bullish_score += 2
    if (rsi_val >= 50.0 and rsi_val <= 70.0) or (rsi_val <= 30.0): total_bullish_score += 1
    if pe_ratio is not None and pe_ratio < 25.0: total_bullish_score += 1

    if pct_change > 0.0 and total_bullish_score >= 5 and z_score_val < 1.8:
        action_decision = "ACCUMULATE (BUY)"
        banner_bg = "#00C853"
        action_summary = f"Up {pct_change:+.2f}% today with strong institutional accumulation."
    elif pct_change < 0.0 or total_bullish_score < 4 or z_score_val > 2.0:
        action_decision = "SHORT / REDUCE"
        banner_bg = "#D50000"
        action_summary = f"Down {pct_change:+.2f}% today under heavy selling pressure or overextension."
    elif pct_change == 0.0 or total_bullish_score >= 4:
        action_decision = "HOLD (NEUTRAL)"
        banner_bg = "#FF6D00"
        action_summary = "Trading flat; overall market momentum remains balanced."
    else:
        action_decision = "EXIT / AVOID"
        banner_bg = "#AA00FF"
        action_summary = "High volatility squeeze active; await trend confirmation."

    # -------------------------------------------------------------
    # 6. HEADER CARDS (DAYLIGHT OPTIMIZED)
    # -------------------------------------------------------------
    head_col1, head_col2 = st.columns([1.6, 1])

    with head_col1:
        st.markdown(f"""
        <div class="terminal-card">
            <div class="stock-title-main">
                {company_name} <span class="stock-symbol-badge">({ticker_symbol})</span>
            </div>
            <div class="stock-meta-info">
                Sector: {sector} &nbsp;|&nbsp; Industry: {industry}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with head_col2:
        st.markdown(f"""
        <div class="terminal-card" style="padding: 0; background: transparent; border: none;">
            <div class="verdict-box-solid" style="background-color: {banner_bg};">
                <div class="verdict-title">
                    VERDICT: {action_decision}
                </div>
                <div class="verdict-desc">
                    {action_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 7. QUANT HEALTH CHECKS (SIMPLIFIED HEADERS & TECHNICAL DELTAS)
    # -------------------------------------------------------------
    st.subheader("⚡ Dashboard Health Checks")
    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        label="Price Valuation", 
        value="Fair Value" if -1.5 <= z_score_val <= 1.5 else ("Expensive" if z_score_val > 1.5 else "Cheap"), 
        delta=f"Z-Score: {z_score_val:+.2f} σ",
        help="Price Z-Score: Checks if the stock is priced normally (Fair Value), too high (Expensive), or deeply discounted (Cheap) compared to its recent average."
    )

    q2.metric(
        label="Big Money Activity", 
        value="BUYING" if obv_slope_val > 0 else "SELLING", 
        delta=f"OBV Delta: {obv_slope_val:,.0f}",
        help="Smart Money Flow: Tracks whether large institutional investors are accumulating shares or quietly dumping them."
    )

    q3.metric(
        label="Breakout Stage", 
        value="COILING / SQUEEZE" if squeeze_val else "ACTIVE MOVE", 
        delta="Consolidation" if squeeze_val else "Trending Now",
        help="Volatility Squeeze: 'Coiling' means price is compressed like a spring before a sharp breakout. 'Active Move' means the expansion is underway."
    )

    q4.metric(
        label="AI Upward Odds", 
        value=f"{prob_up:.1f}% Win Chance", 
        delta=f"XGBoost Acc: {accuracy:.1f}%",
        help="XGBoost Edge: An AI model that analyzes past price patterns to estimate the probability of the stock moving higher tomorrow by >0.75%."
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # 8. LIVE PRICE SNAPSHOT
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
    # 9. FUNDAMENTALS & RISK CONTROL
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
    # 10. INTERACTIVE CHARTS & FINANCIAL STATEMENTS
    # -------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Price Action & Volume Profile", "📜 Quarterly Financials"])

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
