import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone, timedelta

st.set_page_config(page_title="Master Quant Terminal - IST Edition", page_icon="🏛️", layout="wide")

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
run = c3.button("Run Simple Analysis", use_container_width=True)

ticker = f"{sym.strip()}{ex}" if sym else None

if ticker:
    @st.cache_data(ttl=5)
    def pull_data(t):
        stock = yf.Ticker(t)
        df = stock.history(period="2mo", interval="1d")
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    df = pull_data(ticker)
    if df.empty or len(df) < 15:
        st.error(f"❌ Could not load sufficient data for '{ticker}'. Please check the spelling.")
    else:
        # True ATR Calculation (14 periods) to prevent wild scaling
        df['HL'] = df['High'] - df['Low']
        df['HC'] = abs(df['High'] - df['Close'].shift(1))
        df['LC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
        atr = float(df['TR'].rolling(14).mean().iloc[-1])
        
        p = float(df['Close'].iloc[-1])
        op = float(df['Open'].iloc[-1])
        chg = ((p - op) / op) * 100
        cur_h, cur_l = float(df['High'].iloc[-1]), float(df['Low'].iloc[-1])
        
        # Indian Stock Market Timing Logic (IST: 09:15 - 15:30)
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
            market_status = "🔴 Market is CLOSED (Showing Last Saved Prices)"

        # Classic Pivot Calculation using latest session
        pp = (cur_h + cur_l + p) / 3
        bull = p >= pp
        
        # Plain English Recommendations & Colors
        if bull:
            action_advice = "ACCUMULATE / BUY"
            story = "Buyers are actively stepping in and pushing the price upward. The overall mood for this stock is positive, making it a good candidate to buy or slowly build a position."
            card_bg, border_c, accent_c = "#ecfdf5", "#059669", "#047857"
        else:
            action_advice = "SELL / HOLD CAUTIOUSLY"
            story = "Sellers are currently in control and pushing prices down. It is safer to wait, hold off on buying, or exit existing positions to protect your money."
            card_bg, border_c, accent_c = "#fef2f2", "#dc2626", "#b91c1c"

        # Realistic Projections using True ATR & Pivots
        nc_h, nc_l = p + (atr * 0.15), p - (atr * 0.15)
        nd_h1, nd_h2 = p + (atr * 0.5), p + (atr * 1.0)
        nd_l1, nd_l2 = p - (atr * 0.5), p - (atr * 1.0)
        nw_h1, nw_h2 = p + (atr * 1.5), p + (atr * 2.5)
        nw_l1, nw_l2 = p - (atr * 1.5), p - (atr * 2.5)

        # Header Display
        st.markdown(f"<h2 style='margin:0; font-size:20px; color:#0f172a;'>📍 {ticker} | <span style='color:{accent_c};'>₹{p:.2f} ({chg:+.2f}%)</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px; font-weight:700; color:#475569; margin-top:4px;'>{market_status} (IST: {ist_now.strftime('%H:%M:%S')})</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#cbd5e1; margin:10px 0;'>", unsafe_allow_html=True)

        # Layout for Clear Reading
        col1, col2 = st.columns([1.5, 2.5], gap="medium")
        
        with col1:
            st.markdown(f"""
                <div class="card" style="background: {card_bg}; border: 2px solid {border_c};">
                    <div class="title" style="color: {accent_c};">Expert Decision & Advice</div>
                    <div style="font-size: 18px; font-weight: 900; color: {accent_c}; margin-top: 8px;">{action_advice}</div>
                    <div style="font-size: 12px; color: #0f172a; line-height: 1.5; margin-top: 8px;">
                        <b>What this means for you:</b> {story}
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

            # Next Day Targets (Side-by-Side Highs & Lows - Realistic Scale)
            sub_c2.markdown(f"""
                <div class="card">
                    <div class="title">Next Day Targets</div>
                    <div style="display: flex; gap: 8px; margin-top: 6px;">
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

            # Next Week Targets (Side-by-Side Highs & Lows - Realistic Scale)
            sub_c3.markdown(f"""
                <div class="card">
                    <div class="title">Next Week Targets</div>
                    <div style="display: flex; gap: 8px; margin-top: 6px;">
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
    st.info("💡 Type an Indian stock ticker (like RELIANCE, TCS, or SBIN) above to see the simple analysis.")
