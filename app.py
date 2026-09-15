import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel

# Page Config
st.set_page_config(page_title="QuantEdge 360° Institutional Terminal", layout="wide")

st.title("🏛️ QuantEdge 360° Stock Intelligence Terminal")
st.caption("Real-Time NSE/BSE Fundamentals, Ownership Structure, Valuation Ratios & Quantitative Signals")

# Sidebar Exchange Selector & Input
exchange = st.sidebar.radio("Select Exchange:", ["NSE (.NS)", "BSE (.BO)"])
raw_input = st.sidebar.text_input("Enter Stock Symbol:", "JPPOWER").strip().upper()
sanitized_symbol = "".join(e for e in raw_input if e.isalnum())

# Auto-format Ticker Symbol
suffix = ".NS" if exchange == "NSE (.NS)" else ".BO"
ticker_symbol = f"{sanitized_symbol}{suffix}" if not sanitized_symbol.endswith((".NS", ".BO")) else sanitized_symbol

risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio:", 1.0, 4.0, 2.0, 0.5)
atr_multiplier = st.sidebar.slider("Stop-Loss ATR Multiplier:", 1.0, 3.0, 1.5, 0.25)

# -------------------------------------------------------------
# CACHED DATA FETCHING ENGINE
# -------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_stock_master(symbol):
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="2y", interval="1d")
    
    # Extract Metadata Safely
    info = ticker.info if hasattr(ticker, 'info') else {}
    major_holders = ticker.major_holders if hasattr(ticker, 'major_holders') else pd.DataFrame()
    financials = ticker.quarterly_financials if hasattr(ticker, 'quarterly_financials') else pd.DataFrame()
    
    return history, info, major_holders, financials

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

# Load Everything
df, info, major_holders, financials = fetch_stock_master(ticker_symbol)

if df is None or df.empty:
    st.error(f"Could not load data for **{ticker_symbol}**. Verify symbol spelling or exchange selection.")
else:
    company_name = info.get('longName', ticker_symbol)
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    summary = info.get('longBusinessSummary', 'No detailed business summary available.')

    # -------------------------------------------------------------
    # 1. LIVE HEADER & CORE MARKET SNAPSHOT
    # -------------------------------------------------------------
    st.markdown(f"## **{company_name}** (`{ticker_symbol}`)")
    st.caption(f"**Sector:** {sector} | **Industry:** {industry}")

    curr_price = info.get('currentPrice', df['Close'].iloc[-1])
    prev_close = info.get('previousClose', df['Close'].iloc[-2])
    price_change = curr_price - prev_close
    pct_change = (price_change / prev_close) * 100

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Live Market Price", f"₹{curr_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    h2.metric("Day High", f"₹{info.get('dayHigh', df['High'].iloc[-1]):.2f}")
    h3.metric("Day Low", f"₹{info.get('dayLow', df['Low'].iloc[-1]):.2f}")
    
    mcap = info.get('marketCap', 0)
    mcap_crores = mcap / 1e7 if mcap else 0
    h4.metric("Market Cap", f"₹{mcap_crores:,.0f} Cr" if mcap_crores > 0 else "N/A")
    h5.metric("Volume Today", f"{int(info.get('volume', df['Volume'].iloc[-1])):,.0f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. FUNDAMENTAL VALUATION & METRICS GRID
    # -------------------------------------------------------------
    st.subheader("📋 Fundamental Ratios & Valuation Data")
    
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric("Trailing P/E", f"{info.get('trailingPE', 'N/A')}")
    f2.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}")
    f3.metric("Price-to-Book (P/B)", f"{info.get('priceToBook', 'N/A')}")
    f4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A")
    f5.metric("Debt to Equity", f"{info.get('debtToEquity', 'N/A')}")
    f6.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")

    # -------------------------------------------------------------
    # 3. 52-WEEK RANGE & VOLATILITY GAUGES
    # -------------------------------------------------------------
    high_52 = info.get('fiftyTwoWeekHigh', df['High'].max())
    low_52 = info.get('fiftyTwoWeekLow', df['Low'].min())
    range_span = high_52 - low_52
    position_pct = ((curr_price - low_52) / range_span) * 100 if range_span > 0 else 50

    st.markdown("### **52-Week Range Trajectory**")
    st.progress(min(max(int(position_pct), 0), 100))
    st.caption(f"**52W Low:** ₹{low_52:.2f}  |  **Current:** ₹{curr_price:.2f} ({position_pct:.1f}% off Low)  |  **52W High:** ₹{high_52:.2f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. QUANTITATIVE & MICROSTRUCTURE INDICATORS
    # -------------------------------------------------------------
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
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

    df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    clean_df = df.dropna().copy()

    model, accuracy, features = train_xgboost(clean_df)
    latest_features = clean_df[features].tail(1)
    prob_up = model.predict_proba(latest_features)[0][1] * 100

    atr_val = float(df['ATR'].iloc[-1])
    z_score_val = float(clean_df['Z_Score'].iloc[-1])
    obv_slope_val = float(clean_df['OBV_Slope'].iloc[-1])
    squeeze_val = bool(clean_df['Squeeze_Active'].iloc[-1])

    stop_loss = curr_price - (atr_val * atr_multiplier)
    take_profit = curr_price + ((curr_price - stop_loss) * risk_reward_ratio)

    st.subheader("⚡ Institutional Quantitative Signals")
    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Price Z-Score", f"{z_score_val:+.2f} σ", 
              "Overbought (> +2)" if z_score_val > 2 else ("Oversold (< -2)" if z_score_val < -2 else "Fair Value"))
    q2.metric("Smart Money Flow", "ACCUMULATION" if obv_slope_val > 0 else "DISTRIBUTION", f"{obv_slope_val:,.0f} Vol Delta")
    q3.metric("Volatility Squeeze", "FIRE READY" if squeeze_val else "EXPANDED", "Consolidation" if squeeze_val else "Active Trend")
    q4.metric("XGBoost Predictive Edge", f"{prob_up:.1f}% Bullish", f"Accuracy: {accuracy:.1f}%")

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. DEEP TABS: FINANCIALS, HOLDINGS & BUSINESS DETAILS
    # -------------------------------------------------------------
    t1, t2, t3, t4 = st.columns(4)
    tab1, tab2, tab3 = st.tabs(["📊 Interactive Technical Chart", "🏦 Shareholding Pattern", "📜 Quarterly Financials"])

    with tab1:
        hist_df = df.tail(120)
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=hist_df.index, open=hist_df['Open'], high=hist_df['High'],
            low=hist_df['Low'], close=hist_df['Close'], name="Price"
        ))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#FF9800', width=1)))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#2196F3', width=1)))

        fig.add_hline(y=stop_loss, line_dash="dash", line_color="#FF5252", annotation_text=f"Stop-Loss (₹{stop_loss:.2f})")
        fig.add_hline(y=take_profit, line_dash="dash", line_color="#00E676", annotation_text=f"Take-Profit (₹{take_profit:.2f})")

        fig.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Institutional & Major Holders Breakdown")
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

    # Business Overview Footer
    with st.expander(f"ℹ️ About {company_name}"):
        st.write(summary)
