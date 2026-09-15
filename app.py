import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION & AUTHENTIC GROWW UI STYLING
# -------------------------------------------------------------
st.set_page_config(page_title="Groww Trading Terminal", layout="wide")

st.markdown("""
<style>
    /* Groww Dark Theme Global Styles */
    .stApp {
        background-color: #0f141e;
        color: #f0f4f8;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
    /* Groww Card Component */
    .groww-card {
        background-color: #1c212b;
        border: 1px solid #28303d;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .groww-header-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0;
    }
    .groww-symbol-tag {
        background-color: #28303d;
        color: #00d09c;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .groww-price-lg {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1c212b;
        border: 1px solid #28303d;
        padding: 14px 18px;
        border-radius: 12px;
    }
    div[data-testid="stMetric"] label {
        font-size: 12px !important;
        color: #8c96a5 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 18px !important;
        color: #ffffff !important;
    }
    /* Tabs Styling (Groww Style) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid #28303d;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1c212b;
        border: 1px solid #28303d;
        border-radius: 8px;
        color: #8c96a5;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d09c !important;
        color: #0f141e !important;
        border-color: #00d09c !important;
    }
    /* Buttons */
    .stButton button {
        background-color: #00d09c;
        color: #0f141e;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #00b386;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. GROWW TOP NAVIGATION & SEARCH BAR
# -------------------------------------------------------------
nav_c1, nav_c2, nav_c3 = st.columns([1.2, 1.2, 3.6])
with nav_c1:
    exchange = st.selectbox("Exchange:", ["NSE (.NS)", "BSE (.BO)"])
with nav_c2:
    raw_input = st.text_input("Search Stock:", "JPPOWER").strip().upper()
with nav_c3:
    st.markdown("<div style='font-size: 11px; color: #8c96a5; margin-bottom: 2px; font-weight: 600;'>TRENDING SECTORS TODAY</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background-color: #1c212b; border: 1px solid #28303d; padding: 6px 14px; border-radius: 20px; color: #00d09c; font-size: 11px; font-weight: 600; display: inline-block;'>"
        "⚡ IT (+1.4%) &nbsp;&nbsp; 🔥 Metal (+2.1%) &nbsp;&nbsp; 💊 Pharma (+0.9%) &nbsp;&nbsp; 🏦 Bank (+0.4%)"
        "</div>", 
        unsafe_allow_html=True
    )

sanitized_symbol = "".join(e for e in raw_input if e.isalnum())
suffix = ".NS" if exchange == "NSE (.NS)" else ".BO"
ticker_symbol = f"{sanitized_symbol}{suffix}" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

risk_reward_ratio = 2.0
atr_multiplier = 1.5
capital_allocated = 100000.0

# -------------------------------------------------------------
# 3. AI MODEL ENGINE & DATA FETCHING
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
    yearly_financials = ticker.financials if hasattr(ticker, 'financials') else pd.DataFrame()
    news = ticker.news if hasattr(ticker, 'news') else []
    return df, info, financials, yearly_financials, news

@st.cache_resource
def train_ensemble_model(X, y):
    if len(y.dropna()) < 60:
        return None, 50.0

    tscv = TimeSeriesSplit(n_splits=3)
    precisions = []

    xgb = XGBClassifier(
        n_estimators=120, learning_rate=0.015, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
        reg_lambda=1.5, random_state=42, n_jobs=-1
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_split=5,
        random_state=42, n_jobs=-1
    )

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        xgb.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)
        
        p1 = xgb.predict_proba(X_te)[:, 1]
        p2 = rf.predict_proba(X_te)[:, 1]
        blend_p = (p1 + p2) / 2.0
        preds = (blend_p >= 0.52).astype(int)
        
        true_pos = np.sum((preds == 1) & (y_te == 1))
        pred_pos = np.sum(preds == 1)
        if pred_pos > 0:
            precisions.append(true_pos / pred_pos)

    xgb.fit(X, y)
    rf.fit(X, y)
    avg_precision = (np.mean(precisions) * 100) if precisions else 50.0
    return (xgb, rf), avg_precision

df, info, financials, yearly_financials, news_items = fetch_stock_master(ticker_symbol)

if df is None or df.empty:
    st.error(f"Could not load market data for **{ticker_symbol}**. Verify symbol or exchange configuration.")
else:
    company_name = info.get('longName') or info.get('shortName') or ticker_symbol
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    summary = info.get('longBusinessSummary', 'No detailed business summary available.')

    # -------------------------------------------------------------
    # 4. QUANTITATIVE FEATURE PIPELINE & MULTI-HORIZONS
    # -------------------------------------------------------------
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    df['Rolling_Mean'] = df['Close'].rolling(50).mean()
    df['Rolling_Std'] = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']

    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = obv.diff(10)

    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
    df['Squeeze_Active'] = (bb.bollinger_hband() < kc.keltner_channel_hband()) & (bb.bollinger_lband() > kc.keltner_channel_lband())

    df['Ret_1D'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Ret_5D'] = np.log(df['Close'] / df['Close'].shift(5))
    df['Ret_20D'] = np.log(df['Close'] / df['Close'].shift(20))
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
    df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    df['RSI_Slope'] = df['RSI'] - df['RSI'].shift(1)

    df['Target_SameDay'] = np.where(df['Close'] > df['Open'], 1, 0)

    horizons = {
        "Same Day (Remaining)": 0,
        "Next Trading Day (1D)": 1,
        "Next 3 Days": 3,
        "Next Week (5D)": 5,
        "Next 10 Days": 10,
        "Next 15 Days": 15,
        "Next 1 Month (21D)": 21,
        "Next 2 Months (42D)": 42,
        "Next 3 Months (63D)": 63
    }

    feature_cols = [
        'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'ATR', 
        'Z_Score', 'OBV_Slope', 'Ret_1D', 'Ret_5D', 'Ret_20D', 
        'HL_Spread', 'Vol_ZScore', 'RSI_Slope'
    ]

    predictions_summary = []
    
    for label, days in horizons.items():
        temp_df = df.copy()
        if days == 0:
            y_series = temp_df['Target_SameDay']
        else:
            future_ret = (temp_df['Close'].shift(-days) - temp_df['Close']) / temp_df['Close']
            thresh = 0.005 * np.sqrt(days)
            y_series = np.where(future_ret > thresh, 1, 0)
        
        temp_df['Target'] = y_series
        clean_temp = temp_df.dropna(subset=feature_cols + ['Target'])
        
        X_h = clean_temp[feature_cols]
        y_h = clean_temp['Target']
        
        models, prec = train_ensemble_model(X_h, y_h)
        if models is not None:
            latest_x = df[feature_cols].tail(1).fillna(0)
            p1 = models[0].predict_proba(latest_x)[0][1]
            p2 = models[1].predict_proba(latest_x)[0][1]
            prob = ((p1 + p2) / 2.0) * 100
        else:
            prob = 50.0
            prec = 50.0
            
        predictions_summary.append({
            "Horizon": label,
            "Upward Odds": prob,
            "Model Precision": prec
        })

    clean_df = df.dropna(subset=feature_cols).copy()
    curr_price = float(info.get('currentPrice', df['Close'].iloc[-1]))
    prev_close = float(info.get('previousClose', df['Close'].iloc[-2]))
    price_change = curr_price - prev_close
    pct_change = (price_change / prev_close) * 100

    day_open = float(df['Open'].iloc[-1])
    day_close = float(df['Close'].iloc[-1])

    atr_val = float(df['ATR'].iloc[-1])
    z_score_val = float(clean_df['Z_Score'].iloc[-1])
    obv_slope_val = float(clean_df['OBV_Slope'].iloc[-1])
    squeeze_val = bool(clean_df['Squeeze_Active'].iloc[-1])
    rsi_val = float(clean_df['RSI'].iloc[-1])
    sma_200_val = float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else curr_price
    pe_ratio = info.get('trailingPE', None)

    prob_up_next_day = predictions_summary[1]["Upward Odds"]
    next_day_precision = predictions_summary[1]["Model Precision"]

    circuit_pct = 0.10 if curr_price > 100 else 0.20
    upper_circuit = prev_close * (1 + circuit_pct)
    lower_circuit = prev_close * (1 - circuit_pct)

    # -------------------------------------------------------------
    # 5. RISK ENGINE & KELLY CRITERION
    # -------------------------------------------------------------
    stop_loss = curr_price - (atr_val * atr_multiplier)
    risk_per_share = curr_price - stop_loss
    take_profit = curr_price + (risk_per_share * risk_reward_ratio)

    p_win = prob_up_next_day / 100.0
    q_loss = 1.0 - p_win
    b_ratio = risk_reward_ratio
    kelly_fraction = (p_win * b_ratio - q_loss) / b_ratio if b_ratio > 0 else 0.0
    half_kelly = max(0.0, kelly_fraction * 0.5)

    suggested_risk_amount = capital_allocated * half_kelly
    max_shares = int(suggested_risk_amount / risk_per_share) if risk_per_share > 0 else 0

    # -------------------------------------------------------------
    # 6. GROWW STOCK HEADER & EXPERT VERDICT
    # -------------------------------------------------------------
    total_bullish_score = 0
    if curr_price > sma_200_val: total_bullish_score += 2
    if obv_slope_val > 0: total_bullish_score += 2
    if prob_up_next_day >= 53.0: total_bullish_score += 2
    if (rsi_val >= 50.0 and rsi_val <= 70.0) or (rsi_val <= 30.0): total_bullish_score += 1
    if pe_ratio is not None and pe_ratio < 25.0: total_bullish_score += 1

    if pct_change > 0.0 and total_bullish_score >= 5 and z_score_val < 1.8:
        action_decision = "ACCUMULATE (BUY)"
        banner_bg = "#00d09c"
        action_summary = f"Up {pct_change:+.2f}% today with strong institutional edge."
    elif pct_change < 0.0 or total_bullish_score < 4 or z_score_val > 2.0:
        action_decision = "SHORT / REDUCE"
        banner_bg = "#eb5b3c"
        action_summary = f"Down {pct_change:+.2f}% today under selling pressure or overextension."
    elif pct_change == 0.0 or total_bullish_score >= 4:
        action_decision = "HOLD (NEUTRAL)"
        banner_bg = "#ffa726"
        action_summary = "Trading flat; overall market momentum remains balanced."
    else:
        action_decision = "EXIT / AVOID"
        banner_bg = "#ab47bc"
        action_summary = "High volatility active; await trend confirmation."

    head_col1, head_col2 = st.columns([1.6, 1])

    with head_col1:
        st.markdown(f"""
        <div class="groww-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="groww-header-title">{company_name}</div>
                <div class="groww-symbol-tag">{ticker_symbol}</div>
            </div>
            <div style="font-size: 12px; color: #8c96a5; margin-top: 6px;">
                Sector: {sector} &nbsp;|&nbsp; Industry: {industry}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with head_col2:
        st.markdown(f"""
        <div class="groww-card" style="padding: 0; background: transparent; border: none;">
            <div style="background-color: {banner_bg}; border-radius: 12px; padding: 14px 18px; text-align: right; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
                <div style="font-size: 13px; font-weight: 800; text-transform: uppercase; color: #0f141e;">
                    VERDICT: {action_decision}
                </div>
                <div style="font-size: 11px; color: #0f141e; font-weight: 600; opacity: 0.95; margin-top: 2px;">
                    {action_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 7. LIVE PRICE & MARKET STATS BAR (GROWW STYLE)
    # -------------------------------------------------------------
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    h1.metric("Live Price", f"₹{curr_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    h2.metric("Day Open", f"₹{day_open:.2f}")
    h3.metric("Day Close", f"₹{day_close:.2f}")
    h4.metric("Day High / Low", f"₹{info.get('dayHigh', df['High'].iloc[-1]):.2f} / ₹{info.get('dayLow', df['Low'].iloc[-1]):.2f}")
    h5.metric("Upper Circuit Fcst", f"₹{upper_circuit:.2f}", f"+{int(circuit_pct*100)}% Limit")
    h6.metric("Lower Circuit Fcst", f"₹{lower_circuit:.2f}", f"-{int(circuit_pct*100)}% Limit")

    st.markdown("---")

    # -------------------------------------------------------------
    # 8. NEWS CATALYSTS (LAST 1 MONTH & UPCOMING)
    # -------------------------------------------------------------
    st.subheader("📰 Stock News & Catalysts")
    if news_items:
        for item in news_items[:3]:
            title = item.get('title', 'News Headline')
            publisher = item.get('publisher', 'Financial Source')
            link = item.get('link', '#')
            st.markdown(f"- **[{title}]({link})** — *Source: {publisher}*")
    else:
        st.info("No major regulatory or market news articles indexed for this symbol in the current window.")

    st.markdown("---")

    # -------------------------------------------------------------
    # 9. DASHBOARD HEALTH CHECKS
    # -------------------------------------------------------------
    st.markdown("##### ⚡ Quick Health Check Overview")
    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        label="Price Valuation", 
        value="Fair Value" if -1.5 <= z_score_val <= 1.5 else ("Expensive" if z_score_val > 1.5 else "Cheap"), 
        delta=f"Z-Score: {z_score_val:+.2f} σ"
    )
    q2.metric(
        label="Big Money Activity", 
        value="BUYING" if obv_slope_val > 0 else "SELLING", 
        delta=f"OBV Delta: {obv_slope_val:,.0f}"
    )
    q3.metric(
        label="Price Speed / Stage", 
        value="RESTING / PAUSED" if squeeze_val else "MOVING FAST", 
        delta="Preparing to Jump" if squeeze_val else "Price Expanding"
    )
    q4.metric(
        label="AI Next-Day Odds", 
        value=f"{prob_up_next_day:.1f}%", 
        delta=f"Precision: {next_day_precision:.1f}%"
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # 10. MULTI-HORIZON AI FORECAST
    # -------------------------------------------------------------
    st.subheader("🤖 Ensemble AI Upward Odds Across Horizons")
    cols = st.columns(3)
    for idx, item in enumerate(predictions_summary):
        col_idx = idx % 3
        with cols[col_idx]:
            prob_val = item['Upward Odds']
            signal_color = "🟢 Bullish" if prob_val >= 53.0 else ("🔴 Bearish" if prob_val <= 47.0 else "🟡 Neutral")
            st.metric(
                label=item["Horizon"],
                value=f"{prob_val:.1f}%",
                delta=f"{signal_color} (Precision: {item['Model Precision']:.1f}%)"
            )

    st.markdown("---")

    # -------------------------------------------------------------
    # 11. FUNDAMENTALS & KELLY RISK SIZING
    # -------------------------------------------------------------
    st.subheader("🏛️ Fundamentals & Risk Management")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Trailing P/E", f"{info.get('trailingPE', 'N/A')}")
    f2.metric("Price-to-Book", f"{info.get('priceToBook', 'N/A')}")
    f3.metric("Automated Stop-Loss", f"₹{stop_loss:.2f}", f"{atr_multiplier}x ATR")
    f4.metric("Take-Profit Target", f"₹{take_profit:.2f}", f"RR Ratio {risk_reward_ratio}:1")
    f5.metric("Kelly Max Position", f"{max_shares} Shares", f"Risk Allocation: ₹{suggested_risk_amount:,.0f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 12. CHARTS & FINANCIALS TABS
    # -------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Interactive Chart", "📜 Quarterly Financials", "📈 Financial Performance & Growth"])

    with tab1:
        hist_df = df.tail(120)
        price_bins = pd.cut(hist_df['Close'], bins=15)
        volume_profile = hist_df.groupby(price_bins, observed=False)['Volume'].sum()
        poc_bin = volume_profile.idxmax()
        poc_price = (poc_bin.left + poc_bin.right) / 2

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist_df.index, open=hist_df['Open'], high=hist_df['High'],
            low=hist_df['Low'], close=hist_df['Close'], name="Price",
            increasing_line_color='#00d09c', decreasing_line_color='#eb5b3c'
        ))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#ffa726', width=1)))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#29b6f6', width=1)))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_200'], mode='lines', name='SMA 200', line=dict(color='#ab47bc', width=1.5)))

        fig.add_hline(y=poc_price, line_dash="solid", line_color="#e040fb", line_width=2, annotation_text=f"POC: ₹{poc_price:.2f}")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="#eb5b3c", annotation_text=f"Stop-Loss (₹{stop_loss:.2f})")
        fig.add_hline(y=take_profit, line_dash="dash", line_color="#00d09c", annotation_text=f"Target (₹{take_profit:.2f})")

        fig.update_layout(
            template="plotly_dark", 
            height=440, 
            xaxis_title="Date", 
            yaxis_title="Price (₹)",
            plot_bgcolor='#1c212b',
            paper_bgcolor='#0f141e',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if isinstance(financials, pd.DataFrame) and not financials.empty:
            st.dataframe(financials, use_container_width=True)
        else:
            st.info("Quarterly financials data not available for this ticker.")

    with tab3:
        st.markdown("### 📊 Revenue, Profit & Historical Comparison")
        if isinstance(yearly_financials, pd.DataFrame) and not yearly_financials.empty:
            st.write("**Yearly Financial Summary (Revenue & Net Income Comparison):**")
            st.dataframe(yearly_financials, use_container_width=True)
        else:
            st.info("Yearly financial reports not available for comparison.")
            
        if isinstance(financials, pd.DataFrame) and not financials.empty:
            st.write("**Quarterly Trend Reports:**")
            st.dataframe(financials, use_container_width=True)

    with st.expander(f"ℹ️ Business Profile: {company_name}"):
        st.write(summary)
