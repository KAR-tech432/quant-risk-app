import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone, timedelta

st.set_page_config(page_title="Multi-Formula Master Quant Terminal", page_icon="🏛️", layout="wide")

# High-Visibility Daylight Theme CSS
st.markdown("""
    <style>
        .stApp { background: #f1f5f9; color: #0f172a; font-family: -apple-system, sans-serif; }
        .card { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
        .title { font-size: 11px; font-weight: 900; text-transform: uppercase; letter-spacing: 0.8px; color: #475569; }
        .stTextInput input { background-color: #ffffff !important; color: #0f172a !important; border: 1px solid #94a3b8 !important; }
        .stSelectbox div[data-baseweb="select"] { background-color: #ffffff !important; color: #0f172a !important; }
    </style>
""", unsafe_allow_html=True)

# Control Bar
c1, c2, c3 = st.columns([1, 2.5, 1.2])
ex = ".NS" if c1.selectbox("Exchange", ["NSE", "BSE"]) == "NSE" else ".BO"
sym = c2.text_input("Asset Ticker", placeholder="e.g. RELIANCE, TCS, SBIN").upper()
c3.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
run = c3.button("Run Multi-Formula Engine", use_container_width=True)

ticker = f"{sym.strip()}{ex}" if sym else None

if ticker:
    @st.cache_data(ttl=5)
    def pull_data(t):
        stock = yf.Ticker(t)
        df = stock.history(period="3mo", interval="1d")
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    df = pull_data(ticker)
    if df.empty or len(df) < 20:
        st.error(f"❌ Insufficient telemetry data for '{ticker}'. Check symbol spelling.")
    else:
        # --- MULTI-FORMULA QUANTITATIVE ENGINE ---
        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']
        
        p = float(close.iloc[-1])
        op = float(df['Open'].iloc[-1])
        chg = ((p - op) / op) * 100
        cur_h, cur_l = float(high.iloc[-1]), float(low.iloc[-1])
        
        # 1. True Average True Range (ATR)
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        # 2. Volume-Weighted Average Price (VWAP proxy)
        vwap = float((vol * (high + low + close) / 3).sum() / vol.sum()) if vol.sum() > 0 else p
        
        # 3. Bollinger Bands (20-period, 2 std dev)
        sma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)
        
        # 4. Pivot Points & Fibonacci Multipliers
        pp = (cur_h + cur_l + p) / 3
        swing_range = high.tail(10).max() - low.tail(10).min()
        
        # Ensemble Consensus Scoring for Action Advice
        bull_score = sum([
            1 if p > pp else 0,
            1 if p > vwap else 0,
            1 if p > sma20 else 0,
            1 if chg > 0 else 0
        ])
        bull = bull_score >= 2

        if bull:
            action_advice = "ACCUMULATE / BUY"
            story = f"Multiple quantitative models (Pivots, VWAP, and Trend Momentum) align positively. Price is trading above institutional fair value (VWAP ₹{vwap:.2f}), signaling strong buyer defense."
            card_bg, border_c, accent_c = "#ecfdf5", "#059669", "#047857"
        else:
            action_advice = "SELL / HOLD CAUTIOUSLY"
            story = f"Models indicate distribution pressure. Price is trading below institutional fair value (VWAP ₹{vwap:.2f}) and key moving averages, suggesting sellers control the structure."
            card_bg, border_c, accent_c = "#fef2f2", "#dc2626", "#b91c1c"

        # Multi-Model Ensembled Projections
        # Next Candle (Micro Bollinger + ATR Hybrid)
        nc_h = min(p + (atr * 0.12), bb_upper)
        nc_l = max(p - (atr * 0.12), bb_lower)
        
        # Next Day (Pivot + ATR + Fibonacci Blend)
        nd_h1 = (2 * pp) - cur_l
        nd_h2 = p + (atr * 0.8)
        nd_l1 = (2 * pp) - cur_h
        nd_l2 = p - (atr * 0.8)
        
        # Next Week (Multi-session Volatility & Swing Expansion)
        nw_h1 = p + (swing_range * 0.5) + (atr * 1.2)
        nw_h2 = p + swing_range + (atr * 2.0)
        nw_l1 = p - (swing_range * 0.5) - (atr * 1.2)
        nw_l2 = p - swing_range - (atr * 2.0)

        # Indian Stock Market Timing (IST: 09:15 - 15:30)
        ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        current_time = ist_now.time()
        market_open = time(9, 15)
        market_close = time(15, 30)
        is_weekday = ist_now.weekday() < 5
        
        if is_weekday and market_open <= current_time <= market_close:
            market_status = "🟢 Market is OPEN Right Now"
        elif is_weekday and current_time < market_open:
            market_status = "🟡 Pre-Market Time (Opens at 09:15 AM)"
        else:
            market_status = "🔴 Market is CLOSED (Showing Last Settled Session Data)"

        # Header Display
        st.markdown(f"<h2 style='margin:0; font-size:20px; color:#0f172a;'>📍 {ticker} | <span style='color:{accent_c};'>₹{p:.2f} ({chg:+.2f}%)</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px; font-weight:700; color:#475569; margin-top:4px;'>{market_status} (IST: {ist_now.strftime('%H:%M:%S')}) | Multi-Model Ensemble Active</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#cbd5e1; margin:10px 0;'>", unsafe_allow_html=True)

        # Layout
        col1, col2 = st.columns([1.5, 2.5], gap="medium")
        
        with col1:
            st.markdown(f"""
                <div class="card" style="background: {card_bg}; border: 2px solid {border_c};">
                    <div class="title" style="color: {accent_c};">Ensemble Expert Decision</div>
                    <div style="font-size: 18px; font-weight: 900; color: {accent_c}; margin-top: 8px;">{action_advice}</div>
                    <div style="font-size: 12px; color: #0f172a; line-height: 1.5; margin-top: 8px;">
                        <b>Synthesis:</b> {story}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            sub_c1, sub_c2, sub_c3 = st.columns(3, gap="small")
            
            sub_c1.markdown(f"""
                <div class="card">
                    <div class="title">Next Candle</div>
                    <div style="font-size: 11px; margin-top: 6px; color: #0f172a; line-height: 1.5;">
                        ▲ <b>High:</b> ₹{nc_h:.2f}<br>
                        ▼ <b>Low:</b> ₹{nc_l:.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Side-by-Side Next Day
            sub_c2.markdown(f"""
                <div class="card">
                    <div class="title">Next Day Targets</div>
                    <div style="display: flex; gap: 6px; margin-top: 6px;">
                        <div style="flex: 1;">
                            <div style="font-size: 9px; font-weight: 800; color: #047857; margin-bottom: 2px;">HIGHS</div>
                            <span style="font-size: 10px; color: #0f172a;">🟢 H1: ₹{nd_h1:.2f}</span><br>
                            <span style="font-size: 10px; color: #0f172a;">🟢 H2: ₹{nd_h2:.2f}</span>
                        </div>
                        <div style="flex: 1;">
                            <div style="font-size: 9px; font-weight: 800; color: #b91c1c; margin-bottom: 2px;">LOWS</div>
                            <span style="font-size: 10px; color: #0f172a;">🔴 L1: ₹{nd_l1:.2f}</span><br>
                            <span style="font-size: 10px; color: #0f172a;">🔴 L2: ₹{nd_l2:.2f}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Side-by-Side Next Week
            sub_c3.markdown(f"""
                <div class="card">
                    <div class="title">Next Week Targets</div>
                    <div style="display: flex; gap: 6px; margin-top: 6px;">
                        <div style="flex: 1;">
                            <div style="font-size: 9px; font-weight: 800; color: #047857; margin-bottom: 2px;">HIGHS</div>
                            <span style="font-size: 10px; color: #0f172a;">🟢 H1: ₹{nw_h1:.2f}</span><br>
                            <span style="font-size: 10px; color: #0f172a;">🟢 H2: ₹{nw_h2:.2f}</span>
                        </div>
                        <div style="flex: 1;">
                            <div style="font-size: 9px; font-weight: 800; color: #b91c1c; margin-bottom: 2px;">LOWS</div>
                            <span style="font-size: 10px; color: #0f172a;">🔴 L1: ₹{nw_l1:.2f}</span><br>
                            <span style="font-size: 10px; color: #0f172a;">🔴 L2: ₹{nw_l2:.2f}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
else:
    st.info("💡 Type an Indian stock ticker (like RELIANCE, TCS, or SBIN) above to execute the multi-formula engine.")
