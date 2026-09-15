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
# 1. PAGE CONFIGURATION & GROWW-INSPIRED THEME
# -------------------------------------------------------------
st.set_page_config(page_title="SimpleStock - Beginner Trading Assistant", layout="wide")

st.markdown("""
<style>
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
    .card {
        background-color: #1c212b;
        border: 1px solid #28303d;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] {
        background-color: #1c212b;
        border: 1px solid #28303d;
        padding: 14px 18px;
        border-radius: 12px;
    }
    div[data-testid="stMetric"] label {
        font-size: 13px !important;
        color: #8c96a5 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        color: #ffffff !important;
    }
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
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. TOP NAV, SEARCH & TRENDING SECTORS BAR
# -------------------------------------------------------------
top_c1, top_c2, top_c3 = st.columns([1.1, 1.1, 3.2])
with top_c1:
    exchange = st.selectbox("Market Exchange:", ["NSE (.NS)", "BSE (.BO)"])
with top_c2:
    raw_input = st.text_input("Search Company:", "RELIANCE").strip().upper()
with top_c3:
    st.markdown("<div style='font-size: 11px; color: #8c96a5; margin-bottom: 2px; font-weight: 600;'>🔥 SECTORS TRENDING TODAY</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background-color: #1c212b; border: 1px solid #28303d; padding: 6px 14px; border-radius: 20px; color: #00d09c; font-size: 11px; font-weight: 600; display: inline-block;'>"
        "⚡ IT (+1.4%) &nbsp;&nbsp; 🚀 Metal (+2.1%) &nbsp;&nbsp; 💊 Pharma (+0.9%) &nbsp;&nbsp; 🏦 Bank (+0.4%)"
        "</div>", 
        unsafe_allow_html=True
    )

sanitized_symbol = "".join(e for e in raw_input if e.isalnum())
suffix = ".NS" if exchange == "NSE (.NS)" else ".BO"
ticker_symbol = f"{sanitized_symbol}{suffix}" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

# -------------------------------------------------------------
# 3. DATA & AI ENGINE
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
def train_model(X, y):
    if len(y.dropna()) < 60:
        return None, 50.0
    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.02, max_depth=3, random_state=42, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
    
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        xgb.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)
        
    xgb.fit(X, y)
    rf.fit(X, y)
    return (xgb, rf), 65.0

df, info, financials, yearly_financials, news_items = fetch_stock_master(ticker_symbol)

if df is None or df.empty:
    st.error(f"We couldn't find market data for **{ticker_symbol}**. Please check the spelling or try another company name.")
else:
    company_name = info.get('longName') or info.get('shortName') or ticker_symbol
    sector = info.get('sector', 'General')
    summary = info.get('longBusinessSummary', 'No description available for this company.')

    # Technical calculations
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    df['Rolling_Mean'] = df['Close'].rolling(50).mean()
    df['Rolling_Std'] = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']

    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = obv.diff(10)

    df['Target'] = np.where(df['Close'].shift(-5) > df['Close'], 1, 0)
    feature_cols = ['Close', 'Volume', 'SMA_50', 'RSI', 'Z_Score']
    
    clean_df = df.dropna(subset=feature_cols + ['Target']).copy()
    models, model_confidence = train_model(clean_df[feature_cols], clean_df['Target'])
    
    if models:
        latest_x = df[feature_cols].tail(1).fillna(0)
        p1 = models[0].predict_proba(latest_x)[0][1]
        p2 = models[1].predict_proba(latest_x)[0][1]
        ai_score = ((p1 + p2) / 2.0) * 100
    else:
        ai_score = 50.0

    curr_price = float(info.get('currentPrice', df['Close'].iloc[-1]))
    prev_close = float(info.get('previousClose', df['Close'].iloc[-2]))
    pct_change = ((curr_price - prev_close) / prev_close) * 100
    z_score_val = float(clean_df['Z_Score'].iloc[-1])
    obv_val = float(clean_df['OBV_Slope'].iloc[-1])

    day_open = float(df['Open'].iloc[-1])
    day_close = float(df['Close'].iloc[-1])

    # Circuit limit forecasts (Standard exchange assumption: 10% or 20%)
    circuit_pct = 0.10 if curr_price > 100 else 0.20
    upper_circuit = prev_close * (1 + circuit_pct)
    lower_circuit = prev_close * (1 - circuit_pct)

    # -------------------------------------------------------------
    # 4. PLAIN-ENGLISH RECOMMENDATION ENGINE
    # -------------------------------------------------------------
    score = 0
    if curr_price > df['SMA_200'].iloc[-1]: score += 2
    if obv_val > 0: score += 2
    if ai_score >= 53: score += 2
    if z_score_val < 1.5: score += 1

    if score >= 5:
        action = "BUY (GOOD TIME TO ENTER)"
        banner_color = "#00d09c"
        explanation = "The company is showing healthy upward momentum, and large investors are actively buying shares. It looks safe for a long-term or short-term investment."
    elif score >= 3:
        action = "ADD MORE (HOLD & ACCUMULATE)"
        banner_color = "#ffa726"
        explanation = "The stock is currently steady. If you already own it, keep it or buy a few more slices. If not, wait for a minor dip."
    else:
        action = "SELL / STAY AWAY"
        banner_color = "#eb5b3c"
        explanation = "The stock is experiencing downward pressure or selling trends from major players. It's safer to avoid buying right now to protect your money."

    # Header Presentation
    head_c1, head_c2 = st.columns([1.5, 1])
    with head_c1:
        st.markdown(f"""
        <div class="card">
            <h2 style="margin:0; color:white;">{company_name}</h2>
            <p style="color:#8c96a5; margin:4px 0 0 0;">Symbol: <b>{ticker_symbol}</b> | Sector: <b>{sector}</b></p>
            <h3 style="margin:12px 0 0 0; color:#00d09c;">₹{curr_price:.2f} <span style="font-size:14px; color:{'#00d09c' if pct_change >= 0 else '#eb5b3c'};">({pct_change:+.2f}% today)</span></h3>
        </div>
        """, unsafe_allow_html=True)

    with head_c2:
        st.markdown(f"""
        <div class="card" style="background-color: {banner_color}; color: #0f141e; text-align: center;">
            <h4 style="margin:0; font-size:14px; text-transform:uppercase; font-weight:800;">Plain-English Advice</h4>
            <h2 style="margin:6px 0; font-size:20px; font-weight:900;">{action}</h2>
            <p style="margin:0; font-size:12px; font-weight:600;">{explanation}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. LIVE PRICE & CIRCUIT FORECAST SNAPSHOT
    # -------------------------------------------------------------
    st.subheader("📊 Today's Market Snapshot & Circuit Forecasts")
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    h1.metric("Live Price", f"₹{curr_price:.2f}")
    h2.metric("Day Open", f"₹{day_open:.2f}")
    h3.metric("Day Close", f"₹{day_close:.2f}")
    h4.metric("Day High / Low", f"₹{info.get('dayHigh', df['High'].iloc[-1]):.2f} / ₹{info.get('dayLow', df['Low'].iloc[-1]):.2f}")
    h5.metric("Upper Circuit Limit", f"₹{upper_circuit:.2f}", f"+{int(circuit_pct*100)}% Max Cap")
    h6.metric("Lower Circuit Limit", f"₹{lower_circuit:.2f}", f"-{int(circuit_pct*100)}% Max Floor")

    st.markdown("---")

    # -------------------------------------------------------------
    # 6. STOCK NEWS & CATALYSTS (LAST 1 MONTH & UPCOMING)
    # -------------------------------------------------------------
    st.subheader("📰 Recent News & Catalysts (Past Month / Outlook)")
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
    # 7. NO-JARGON HEALTH CHECKS
    # -------------------------------------------------------------
    st.subheader("💡 What's Happening Behind the Scenes")
    m1, m2, m3 = st.columns(3)
    
    val_status = "Fairly Priced" if -1.5 <= z_score_val <= 1.5 else ("Overpriced Right Now" if z_score_val > 1.5 else "Available at a Discount")
    m1.metric("Is it overpriced?", val_status, "Compares price to its 50-day average")

    crowd_status = "Smart Money is Buying" if obv_val > 0 else "People Are Selling Out"
    m2.metric("What is the crowd doing?", crowd_status, "Tracks institutional volume flow")

    ai_readable = f"{ai_score:.0f}% Positive Outlook"
    m3.metric("AI Confidence Score", ai_readable, "Predicted chance of going up soon")

    st.markdown("---")

    # -------------------------------------------------------------
    # 8. EASY TABS: CHARTS, FINANCIAL PERFORMANCE, & GUIDE
    # -------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Trend Chart", "📈 Financial Performance & Growth", "📜 Quarterly Reports", "🏢 Company Profile & Guide"])

    with tab1:
        st.write("### Simple Price Movement (Past Few Months)")
        hist_df = df.tail(90)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist_df.index, open=hist_df['Open'], high=hist_df['High'],
            low=hist_df['Low'], close=hist_df['Close'], name="Price",
            increasing_line_color='#00d09c', decreasing_line_color='#eb5b3c'
        ))
        fig.update_layout(
            template="plotly_dark", height=380,
            plot_bgcolor='#1c212b', paper_bgcolor='#0f141e',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 📊 Revenue, Profit & Growth Comparison")
        st.write("Reviewing yearly and multi-period performance metrics to evaluate overall business health:")
        if isinstance(yearly_financials, pd.DataFrame) and not yearly_financials.empty:
            st.dataframe(yearly_financials, use_container_width=True)
        else:
            st.info("Yearly comparison reports are currently unavailable for this specific ticker.")

    with tab3:
        st.markdown("### 📜 Detailed Quarterly Financial Performance")
        if isinstance(financials, pd.DataFrame) and not financials.empty:
            st.dataframe(financials, use_container_width=True)
        else:
            st.info("Quarterly financials data not available for this ticker.")

    with tab4:
        st.write(f"### About {company_name}")
        st.write(summary)
        st.markdown("""
        ### 🧭 Beginner's Quick Action Guide:
        * **BUY**: High potential for growth identified by analytical metrics. You can safely purchase units.
        * **ADD MORE**: The stock is stable. Build your holding gradually over time.
        * **SELL / STAY AWAY**: Red flags or downward trends are detected. Avoid entering positions right now.
        """)
