import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Enterprise Quant Terminal", page_icon="🏛️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .stApp { background-color: #090d16; color: #e2e8f0; font-family: sans-serif; }
        .block-container { padding: 1.2rem 1rem !important; max-width: 100%; }
        .card { background: linear-gradient(135deg, #131b2e 0%, #0f1726 100%); border: 1px solid #1e293b; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
        .title { font-size: 11px; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; }
        div.stButton > button { background: #1e293b; color: #38bdf8; border: 1px solid #38bdf8; font-weight: 700; border-radius: 6px; width: 100%; padding: 8px; }
        div.stButton > button:hover { background: #38bdf8; color: #090d16; }
    </style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2.5, 1.2], gap="medium")
ex = ".NS" if c1.selectbox("Exchange", ["NSE", "BSE"]) == "NSE" else ".BO"
inp = c2.text_input("Symbol Ticker", placeholder="e.g. RELIANCE, SBIN").upper()
c3.markdown("<div style='height: 27px;'></div>", unsafe_allow_html=True)
run_btn = c3.button("🔄 Analyze", use_container_width=True)

ticker = f"{inp.strip()}{ex}" if inp else None

hc1, hc2 = st.columns([1.5, 2.5])
hc1.markdown("<h3 style='color: #f8fafc; margin: 0; font-size: 18px;'>Quant Terminal</h3>", unsafe_allow_html=True)
header_box = hc2.empty()

if not ticker:
    header_box.markdown("<div style='text-align: right; color: #64748b; font-size: 13px;'>📍 Standby - Awaiting Ticker</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 10px 0; border-color: #1e293b;'>", unsafe_allow_html=True)

if ticker:
    @st.cache_data(ttl=10)
    def fetch_data(t):
        df = yf.Ticker(t).history(period="2d", interval="1m")
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    @st.fragment(run_every=1)
    def live_engine():
        df_raw = fetch_data(ticker)
        if df_raw.empty:
            header_box.markdown("<div style='text-align: right; color: #f87171; font-size: 15px; font-weight: 700;'>📍 Invalid Ticker or Data Offline</div>", unsafe_allow_html=True)
            return
        
        # Resample to 10-Minute Candles for Advanced R:R & Volatility Calculations
        df_10m = df_raw.resample('10min', closed='left', label='left').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        if df_10m.empty:
            df_10m = df_raw.tail(10) # Fallback

        c_price = float(df_raw['Close'].iloc[-1])
        op_price = float(df_raw['Open'].iloc[0])
        chg = ((c_price - op_price) / op_price) * 100
        color = "#34d399" if chg >= 0 else "#f87171"
        
        header_box.markdown(
            f"<div style='text-align: right; font-size: 20px; font-weight: 900; color: #f8fafc;'>"
            f"📍 {ticker.split('.')[0]} <span style='color: #34d399;'>₹{c_price:.2f}</span> "
            f"<span style='font-size: 15px; color: {color};'>({chg:+.2f}%)</span>"
            f"</div>", unsafe_allow_html=True
        )

        H, L, C = float(df_raw['High'].max()), float(df_raw['Low'].min()), c_price
        rng = H - L
        mid = (H + L) / 2
        bull = df_raw['Close'].iloc[-1] >= df_raw['Open'].iloc[-1]

        # Latest 10-Minute Candle Metrics for R:R
        latest_10m_high = float(df_10m['High'].iloc[-1])
        latest_10m_low = float(df_10m['Low'].iloc[-1])
        ten_min_range = latest_10m_high - latest_10m_low

        if bull and C >= mid:
            bias, dec, horizon, bg, border = "BULLISH ACCUMULATION 📈", "ACCUMULATE / BUY", "3 to 5 Trading Sessions", "linear-gradient(135deg, #064e3b 0%, #022c22 100%H)", "#059669"
            reason = "Institutional accumulation evident; buyers defending upper range boundaries."
            # Dynamic 10-min R:R Setup for Buy
            entry_price = C
            stop_loss = latest_10m_low - (ten_min_range * 0.2)
            risk = entry_price - stop_loss
            reward = risk * 2.5 # Professional 1:2.5 R:R Ratio
            target_price = entry_price + reward
            rr_text = f"1 : 2.50"
        elif not bull and C < mid:
            bias, dec, horizon, bg, border = "BEARISH DISTRIBUTION 📉", "SELL / REDUCE", "2 to 3 Trading Sessions", "linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%)", "#dc2626"
            reason = "Distribution dominating order flow; sellers active near resistance."
            entry_price = C
            stop_loss = latest_10m_high + (ten_min_range * 0.2)
            risk = stop_loss - entry_price
            reward = risk * 2.5
            target_price = entry_price - reward
            rr_text = f"1 : 2.50"
        else:
            bias, dec, horizon, bg, border = "EQUILIBRIUM ⚖️", "HOLD / WAIT", "1 to 2 Trading Sessions", "linear-gradient(135deg, #78350f 100%, #451a03 100%)", "#d97706"
            reason = "Consolidation zone; awaiting directional breakout trigger."
            entry_price = C
            stop_loss = latest_10m_low
            target_price = latest_10m_high
            rr_text = f"1 : 1.00 (Neutral)"

        pp = (H + L + C) / 3
        nc_h, nc_l = C + (rng/20), C - (rng/20)
        nd_h1, nd_h2, nd_l1, nd_l2 = (2*pp)-L, pp+rng, (2*pp)-H, pp-rng
        nw_h1, nw_h2, nw_l1, nw_l2 = C+(rng*1.1), C+(rng*2.0), C-(rng*1.1), C-(rng*2.0)
        r1, r2, s1, s2 = (2*pp)-L, H+(pp-L), (2*pp)-H, L-(H-pp)

        r1_c1, r1_c2, r1_c3 = st.columns(3, gap="medium")
        
        r1_c1.markdown(f"""<div class="card" style="background: {bg}; border: 1px solid {border};">
            <div class="title">Expert Market Bias & Decision</div>
            <div style="font-size: 15px; font-weight: 900; color: #fff;">{bias}</div>
            <div style="font-size: 11px; font-weight: 800; color: #38bdf8; margin-top: 4px;">Rec: {dec} | <span style="color: #fbbf24;">Valid: {horizon}</span></div>
            <div style="font-size: 10px; margin-top: 4px; padding: 4px; background: rgba(0,0,0,0.25); border-radius: 4px; color: #f1f5f9;">{reason}</div>
        </div>""", unsafe_allow_html=True)

        r1_c2.markdown(f"""<div class="card">
            <div class="title">10-Min Candle Risk : Reward Matrix</div>
            <div style="font-size: 13px; font-weight: 900; color: #38bdf8; margin-top: 2px;">Ratio: {rr_text}</div>
            <div style="font-size: 11px; margin-top: 4px; color: #f1f5f9;">
                🛡️ <b>SL:</b> ₹{stop_loss:.2f}<br>
                🎯 <b>Target:</b> ₹{target_price:.2f}
            </div>
        </div>""", unsafe_allow_html=True)

        r1_c3.markdown(f"""<div class="card">
            <div class="title">Next Candle Micro Targets</div>
            <div style="font-size: 11px; margin-top: 6px;"><span style="color: #34d399;">▲ High: ₹{nc_h:.2f}</span><br><span style="color: #f87171;">▼ Low: ₹{nc_l:.2f}</span></div>
        </div>""", unsafe_allow_html=True)

        r2_c1, r2_c2, r2_c3 = st.columns(3, gap="medium")
        r2_c1.markdown(f"""<div class="card"><div class="title">Next Day Targets</div>
            <div style="font-size: 11px;"><span style="color: #34d399; font-weight: 700;">H1: ₹{nd_h1:.2f} | H2: ₹{nd_h2:.2f}</span><br><span style="color: #f87171; font-weight: 700;">L1: ₹{nd_l1:.2f} | L2: ₹{nd_l2:.2f}</span></div></div>""", unsafe_allow_html=True)
        r2_c2.markdown(f"""<div class="card"><div class="title">Next Week Targets</div>
            <div style="font-size: 11px;"><span style="color: #34d399; font-weight: 700;">H1: ₹{nw_h1:.2f} | H2: ₹{nw_h2:.2f}</span><br><span style="color: #f87171; font-weight: 700;">L1: ₹{nw_l1:.2f} | L2: ₹{nw_l2:.2f}</span></div></div>""", unsafe_allow_html=True)
        r2_c3.markdown(f"""<div class="card"><div class="title">Session S/R Matrix</div>
            <div style="font-size: 11px;"><span style="color: #f87171; font-weight: 700;">R1: ₹{r1:.2f} | R2: ₹{r2:.2f}</span><br><span style="color: #34d399; font-weight: 700;">S1: ₹{s1:.2f} | S2: ₹{s2:.2f}</span></div></div>""", unsafe_allow_html=True)

    live_engine()
else:
    st.info("💡 **Terminal Ready:** Select market exchange and enter ticker above.")
