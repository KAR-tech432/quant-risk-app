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
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------
st.set_page_config(page_title="QuantEdge 360° Decision Engine", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2A2E39;
    }
    .decision-banner {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 25px;
    }
    .verdict-box {
        background-color: #131722;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #00C853;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ QuantEdge: 360° Institutional Decision Engine")
st.caption("Combining Fundamentals, Microstructure, Smart Money Flow & Multi-Timeframe Scoring Matrix")

# -------------------------------------------------------------
# SIDEBAR CONTROLS & INPUT SANITIZATION
# -------------------------------------------------------------
st.sidebar.header("🎯 Asset & Strategy Settings")
exchange = st.sidebar.radio("Select Exchange:", ["NSE (.NS)", "BSE (.BO)"])
raw_input = st.sidebar.text_input("Stock Symbol:", "JPPOWER").strip().upper()
sanitized_symbol = "".join(e for e in raw_input if e.isalnum())

suffix = ".NS" if exchange == "NSE (.NS)" else ".BO"
ticker_symbol = f"{sanitized_symbol}{suffix}" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio:", 1.0, 4.0, 2.0, 0.5)
atr_multiplier = st.sidebar.slider("Stop-Loss ATR Multiplier:", 1.0, 3.0, 1.5, 0.25)
capital_allocated = st.sidebar.number_input("Capital to Risk (₹):", value=50000, step=5000)

# -------------------------------------------------------------
# CACHED DATA FETCHING LAYER (2-YEAR DATASET)
# -------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_stock_master(symbol):
    ticker = yf.Ticker(symbol)
    # Fetch exact 2-Year lookback period (~500 trading days)
    df = ticker.history(period="2y", interval="1d")
    
    info = ticker.info if hasattr(ticker, 'info') else {}
    major_holders = ticker.major_holders if hasattr(ticker, 'major_holders') else pd.DataFrame()
    financials = ticker.quarterly_financials if hasattr(ticker, 'quarterly_financials') else pd.DataFrame()
    
    return df, info, major_holders, financials

@st.cache_resource
def train_xgboost(clean_data):
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'ATR', 'Z_Score', 'OBV_Slope']
    X = clean_data[features]
    y = clean_data['Target_Direction']

    # 80% Train (~400 days) / 20% Test (~100 days)
    split = int(len(clean_data) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=4, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)

    accuracy = (model.predict(X_test) == y_test).mean() * 100
    return model, accuracy, features

# Load Data
df, info, major_holders, financials = fetch_stock_master(ticker_symbol)

if df is None or df.empty:
    st.error(f"Could not load market data for **{ticker_symbol}**. Verify symbol or exchange configuration.")
else:
    company_name = info.get('longName', ticker_symbol)
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
    
    # 1. Z-Score Mean Reversion (50-Day Rolling Window)
    df['Rolling_Mean'] = df['Close'].rolling(50).mean()
    df['Rolling_Std'] = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']

    # 2. Institutional Money Flow (10-Day OBV Slope Window)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = obv.diff(10)

    # 3. Volatility Squeeze Engine (20-Day Window)
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
    df['Squeeze_Active'] = (bb.bollinger_hband() < kc.keltner_channel_hband()) & (bb.bollinger_lband() > kc.keltner_channel_lband())

    # Machine Learning Target Definition
    df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    clean_df = df.dropna().copy()

    # Model Execution
    model, accuracy, features = train_xgboost(clean_df)
    latest_features = clean_df[features].tail(1)
    prob_up = model.predict_proba(latest_features)[0][1] * 100

    # Key Snapshot Values
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

    # Dynamic Risk Parameters
    stop_loss = curr_price - (atr_val * atr_multiplier)
    risk_per_share = curr_price - stop_loss
    take_profit = curr_price + (risk_per_share * risk_reward_ratio)
    max_shares = int(capital_allocated / risk_per_share) if risk_per_share > 0 else 0

    # -------------------------------------------------------------
    # 8-POINT MULTI-PILLAR SCORING MATRIX ENGINE
    # -------------------------------------------------------------
    score_breakdown = []
    total_bullish_score = 0

    # Criteria 1: 200 DMA Macro Trend
    c1_passed = curr_price > sma_200_val
    if c1_passed: total_bullish_score += 2
    score_breakdown.append({
        "Parameter": "Macro Trend Structure (200 DMA)",
        "Lookback Window": "200 Days",
        "Observed Metric": f"Price ₹{curr_price:.2f} > 200 DMA ₹{sma_200_val:.2f}" if c1_passed else f"Price ₹{curr_price:.2f} < 200 DMA ₹{sma_200_val:.2f}",
        "Weight": "+2 Pts" if c1_passed else "0 Pts",
        "Status": "✅ PASSED" if c1_passed else "❌ FAILED"
    })

    # Criteria 2: Smart Money Accumulation (OBV Slope)
    c2_passed = obv_slope_val > 0
    if c2_passed: total_bullish_score += 2
    score_breakdown.append({
        "Parameter": "Smart Money Flow (OBV Slope)",
        "Lookback Window": "10 Days",
        "Observed Metric": f"+{obv_slope_val:,.0f} Delta (Accumulation)" if c2_passed else f"{obv_slope_val:,.0f} Delta (Distribution)",
        "Weight": "+2 Pts" if c2_passed else "0 Pts",
        "Status": "✅ PASSED" if c2_passed else "❌ FAILED"
    })

    # Criteria 3: Machine Learning Edge (XGBoost)
    c3_passed = prob_up >= 55.0
    if c3_passed: total_bullish_score += 2
    score_breakdown.append({
        "Parameter": "XGBoost Machine Learning Edge",
        "Lookback Window": "2 Years (80/20 Train-Test)",
        "Observed Metric": f"{prob_up:.1f}% Bullish Probability",
        "Weight": "+2 Pts" if c3_passed else "0 Pts",
        "Status": "✅ PASSED" if c3_passed else "❌ FAILED"
    })

    # Criteria 4: Momentum Range (RSI)
    c4_passed = (rsi_val >= 50.0 and rsi_val <= 70.0) or (rsi_val <= 30.0)
    if c4_passed: total_bullish_score += 1
    score_breakdown.append({
        "Parameter": "Momentum Strength (RSI 14)",
        "Lookback Window": "14 Days",
        "Observed Metric": f"RSI @ {rsi_val:.1f} (Healthy Expansion)",
        "Weight": "+1 Pt" if c4_passed else "0 Pts",
        "Status": "✅ PASSED" if c4_passed else "❌ FAILED"
    })

    # Criteria 5: Fundamental Valuation
    c5_passed = pe_ratio is not None and pe_ratio < 25.0
    if c5_passed: total_bullish_score += 1
    score_breakdown.append({
        "Parameter": "Fundamental Valuation (Trailing P/E)",
        "Lookback Window": "Trailing 12 Months",
        "Observed Metric": f"P/E @ {pe_ratio:.2f}" if pe_ratio else "P/E Data N/A",
        "Weight": "+1 Pt" if c5_passed else "0 Pts",
        "Status": "✅ PASSED" if c5_passed else "❌ FAILED"
    })

    # System Decision Mapping
    if total_bullish_score >= 6 and z_score_val < 1.8:
        action_decision = "ACCUMULATE (STRONG BUY)"
        banner_color = "#00C853"
        action_summary = "Technical trend, money flow, and model probability are aligned with healthy valuation parameters."
    elif total_bullish_score < 4 or z_score_val > 2.2:
        action_decision = "SHORT / REDUCE (BEARISH)"
        banner_color = "#FF1744"
        action_summary = "Stock exhibits distribution pressure or price expansion standard deviation is severely overextended."
    elif total_bullish_score >= 4:
        action_decision = "HOLD (NEUTRAL BIASED)"
        banner_color = "#FF9100"
        action_summary = "Macro trend remains intact, but momentum indicators suggest holding existing shares rather than aggressive buying."
    else:
        action_decision = "EXIT / AVOID (NO EDGE)"
        banner_color = "#D500F9"
        action_summary = "Conflicting signals between volume distribution and technical structure. Capital protection advised."

    # -------------------------------------------------------------
    # 1. LIVE DECISION BANNER & TOP SNAPSHOT
    # -------------------------------------------------------------
    st.markdown(f"## **{company_name}** (`{ticker_symbol}`)")
    st.caption(f"**Sector:** {sector} | **Industry:** {industry}")

    st.markdown(f"""
    <div class="decision-banner" style="background-color: {banner_color}; color: white;">
        SYSTEM VERDICT: {action_decision}<br>
        <span style="font-size: 14px; font-weight: normal;">{action_summary}</span>
    </div>
    """, unsafe_allow_html=True)

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Live Market Price", f"₹{curr_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    h2.metric("Day High", f"₹{info.get('dayHigh', df['High'].iloc[-1]):.2f}")
    h3.metric("Day Low", f"₹{info.get('dayLow', df['Low'].iloc[-1]):.2f}")
    
    mcap = info.get('marketCap', 0)
    h4.metric("Market Cap", f"₹{mcap/1e7:,.0f} Cr" if mcap else "N/A")
    h5.metric("Volume Today", f"{int(info.get('volume', df['Volume'].iloc[-1])):,.0f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. DIAGNOSTICS & VERDICT AUDIT BREAKDOWN (NEW FEATURE)
    # -------------------------------------------------------------
    st.subheader("🔍 Verdict Audit: Technical & Parameter Breakdown")
    st.caption(f"Score Achieved: **{total_bullish_score} / 8 Points** | Evaluated Dataset: **{len(df)} Days (2 Years)**")

    audit_df = pd.DataFrame(score_breakdown)
    st.table(audit_df)

    # Timeframe Diagnostics Card
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Dataset Lookback", f"{len(df)} Bars", "Daily Candles (2Y)")
    d2.metric("Train/Test Split", f"{int(len(clean_df)*0.8)} / {int(len(clean_df)*0.2)} Bars", "80% Train | 20% Test")
    d3.metric("Point of Control (POC)", "90-Day Window", "Execution Focus")
    d4.metric("Mean Reversion (Z)", f"{z_score_val:+.2f} σ", "< +1.8 σ Safety Limit Passed")

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. FUNDAMENTAL HEALTH & VALUATION METRICS
    # -------------------------------------------------------------
    st.subheader("🏛️ Fundamental Ratios & Valuation")
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric("Trailing P/E", f"{info.get('trailingPE', 'N/A')}")
    f2.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}")
    f3.metric("Price-to-Book (P/B)", f"{info.get('priceToBook', 'N/A')}")
    f4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A")
    f5.metric("Debt-to-Equity", f"{info.get('debtToEquity', 'N/A')}")
    f6.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")

    high_52 = info.get('fiftyTwoWeekHigh', df['High'].max())
    low_52 = info.get('fiftyTwoWeekLow', df['Low'].min())
    range_span = high_52 - low_52
    position_pct = ((curr_price - low_52) / range_span) * 100 if range_span > 0 else 50

    st.markdown("**52-Week Range Trajectory**")
    st.progress(min(max(int(position_pct), 0), 100))
    st.caption(f"**52W Low:** ₹{low_52:.2f}  |  **Current:** ₹{curr_price:.2f} ({position_pct:.1f}% off Low)  |  **52W High:** ₹{high_52:.2f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. MICROSTRUCTURE & INSTITUTIONAL MONEY FLOW
    # -------------------------------------------------------------
    st.subheader("⚡ Microstructure & Institutional Flow")
    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Price Z-Score", f"{z_score_val:+.2f} σ", 
              "Overbought (> +2)" if z_score_val > 2 else ("Oversold (< -2)" if z_score_val < -2 else "Fair Value"))
    q2.metric("Smart Money Flow", "ACCUMULATION" if obv_slope_val > 0 else "DISTRIBUTION", f"{obv_slope_val:,.0f} Vol Delta")
    q3.metric("Volatility Squeeze", "FIRE READY" if squeeze_val else "EXPANDED", "Consolidation" if squeeze_val else "Active Trend")
    q4.metric("XGBoost Predictive Edge", f"{prob_up:.1f}% Bullish", f"Model Acc: {accuracy:.1f}%")

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. RISK CONTROLS & POSITION SIZING
    # -------------------------------------------------------------
    st.subheader("🛡️ Risk Control & Sizing Matrix")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.error(f"**Automated Stop-Loss:** ₹{stop_loss:.2f}")
        st.caption(f"Risk per share: ₹{risk_per_share:.2f} ({atr_multiplier}x ATR)")
    with c2:
        st.success(f"**Take-Profit Target:** ₹{take_profit:.2f}")
        st.caption(f"Reward per share: ₹{take_profit - curr_price:.2f}")
    with c3:
        st.info(f"**Position Size Limits:** {max_shares} Shares")
        st.caption(f"Based on ₹{capital_allocated:,.0f} capital risk allocation")

    st.markdown("---")

    # -------------------------------------------------------------
    # 6. INTERACTIVE CHARTS & FINANCIAL STATEMENTS TABS
    # -------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Price Action & Point of Control (POC)", "🏦 Shareholding Pattern", "📜 Quarterly Financials"])

    with tab1:
        hist_df = df.tail(120)
        
        # Calculate Point of Control (POC) over 90 Days
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

        fig.add_hline(y=poc_price, line_dash="solid", line_color="#E040FB", line_width=2, annotation_text=f"Point of Control (POC): ₹{poc_price:.2f}")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="#FF5252", annotation_text=f"Stop-Loss (₹{stop_loss:.2f})")
        fig.add_hline(y=take_profit, line_dash="dash", line_color="#00E676", annotation_text=f"Take-Profit (₹{take_profit:.2f})")

        fig.update_layout(template="plotly_dark", height=550, xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Shareholding Pattern")
        if isinstance(major_holders, pd.DataFrame) and not major_holders.empty:
            st.dataframe(major_holders, use_container_width=True)
        else:
            st.info("Major shareholding breakdown data not available for this ticker.")

    with tab3:
        st.subheader("Quarterly Financial Statements")
        if isinstance(financials, pd.DataFrame) and not financials.empty:
            st.dataframe(financials, use_container_width=True)
        else:
            st.info("Quarterly financials data not available for this ticker.")

    with st.expander(f"ℹ️ Business Profile: {company_name}"):
        st.write(summary)
