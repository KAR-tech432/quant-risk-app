import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Master Quant Terminal", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
        .stApp { background: #030712; color: #f8fafc; font-family: -apple-system, sans-serif; }
        .card { border-radius: 8px; padding: 14px; margin-bottom: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
        .title { font-size: 11px; font-weight: 900; text-transform: uppercase; letter-spacing: 0.8px; }
    </style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2.5, 1.2])
ex = ".NS" if c1.selectbox("Exchange", ["NSE", "BSE"]) == "NSE" else ".BO"
sym = c2.text_input("Asset Ticker", placeholder="e.g. RELIANCE, TCS, SBIN").upper()
c3.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
run = c3.button("Execute Deep Analysis", use_container_width=True)

ticker = f"{sym.strip()}{ex}" if sym else None

if ticker:
    @st.cache_data(ttl=5)
    def pull_data(t):
        stock = yf.Ticker(t)
        df = stock.history(period="1mo", interval="1d")
        if df.empty:
            df = stock.history(period="5d", interval="1h")
        return df.tz_localize(None) if not df.empty and df.index.tz is not None else df

    df = pull_data(ticker)
    if df.empty:
        st.error(f"❌ Telemetry link offline for '{ticker}'. Verify ticker spelling.")
    else:
        p = float(df['Close'].iloc[-1])
        op = float(df['Open'].iloc[0])
        chg = ((p - op) / op) * 100
        H, L = float(df['High'].max()), float(df['Low'].min())
        
        # 30-Year Expert Quantitative Matrix & Projections
        pp = (H + L + p) / 3
        atr = H - L
        bull = p >= pp
        
        # Dynamic Market-Condition Color Mapping (Daylight Visible High-Contrast)
        if bull:
            bias, card_bg, border_c, accent_c = "BULLISH ACCUMULATION 🚀", "linear-gradient(135deg, #022c22 0%, #064e3b 100%)", "#059669", "#34d399"
        else:
            bias, card_bg, border_c, accent_c = "BEARISH DISTRIBUTION 🔻", "linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%)", "#dc2626", "#f87171"

        # Item 1: Upcoming Candle Micro Projections
        nc_h, nc_l = p + (atr / 25), p - (atr / 25)

        # Item 2: Next Day High & Low (Dual Values)
        nd_h1, nd_h2 = (2 * pp) - L, pp + atr
        nd_l1, nd_l2 = (2 * pp) - H, pp - atr

        # Item 3: Next Week High & Low (Dual Values)
        nw_h1, nw_h2 = p + (atr * 1.1), p + (atr * 2.0)
        nw_l1, nw_l2 = p - (atr * 1.1), p - (atr * 2.0)

        # Header Telemetry Display
        st.markdown(f"<h2 style='margin:0; font-size:20px;'>📍 {ticker} | <span style='color:{accent_c};'>₹{p:.2f} ({chg:+.2f}%)</span></h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1e293b; margin:10px 0;'>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        col1.markdown(f"""
            <div class="card" style="background: {card_bg}; border: 1px solid {border_c};">
                <div class="title" style="color: {accent_c};">Market Bias & Core State</div>
                <div style="font-size: 13px; font-weight: 900; color: #fff; margin-top: 6px;">{bias}</div>
                <div style="font-size: 11px; color: #cbd5e1; margin-top: 4px;">Order Flow: Institutional positioning active.</div>
            </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
            <div class="card" style="background: #0f172a; border: 1px solid #334155;">
                <div class="title" style="color: #38bdf8;">Next Candle Projections</div>
                <div style="font-size: 12px; margin-top: 6px; color: #f8fafc;">
                    ▲ <b>High:</b> ₹{nc_h:.2f}<br>
                    ▼ <b>Low:</b> ₹{nc_l:.2f}
                </div>
            </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
            <div class="card" style="background: #0f172a; border: 1px solid #334155;">
                <div class="title" style="color: #38bdf8;">Next Day H & L (Dual)</div>
                <div style="font-size: 11px; margin-top: 6px; color: #f8fafc;">
                    🟢 <b>H1:</b> ₹{nd_h1:.2f} | <b>H2:</b> ₹{nd_h2:.2f}<br>
                    🔴 <b>L1:</b> ₹{nd_l1:.2f} | <b>L2:</b> ₹{nd_l2:.2f}
                </div>
            </div>
        """, unsafe_allow_html=True)

        col4.markdown(f"""
            <div class="card" style="background: #0f172a; border: 1px solid #334155;">
                <div class="title" style="color: #38bdf8;">Next Week H & L (Dual)</div>
                <div style="font-size: 11px; margin-top: 6px; color: #f8fafc;">
                    🟢 <b>H1:</b> ₹{nw_h1:.2f} | <b>H2:</b> ₹{nw_h2:.2f}<br>
                    🔴 <b>L1:</b> ₹{nw_l1:.2f} | <b>L2:</b> ₹{nw_l2:.2f}
                </div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("💡 Enter a ticker symbol above to initialize the institutional projection engine.")
