import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone, timedelta

st.set_page_config(page_title="Live Multi-Formula Quant Terminal", page_icon="🏛️", layout="wide")

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
sym = c2.text_input("Asset Ticker", value="TEJASNET", placeholder="e.g. RELIANCE, TCS, TEJASNET").upper()
c3.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
run = c3.button("Run Live Engine", use_container_width=True)

ticker = f"{sym.strip()}{ex}" if sym else None

if ticker:
    @st.cache_data(ttl=5)
    def pull_data(t):
        stock = yf.Ticker(t)
        # Daily history for metrics and formulas
        df = stock.history(period="3mo", interval="1d")
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Intraday history for today's live chart (5m intervals)
        idf = stock.history(period="1d", interval="5m")
        if not idf.empty and idf.index.tz is not None:
            idf.index = idf.index.tz_localize(None)
            
        return df, idf

    df, idf = pull_data(ticker)
    
    if df.empty or len(df) < 20:
        st.error(f"❌ Insufficient live telemetry data for '{ticker}'. Check symbol spelling.")
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
        
        prev_h = float(high.iloc[-2]) if len(high) > 1 else cur_h
        prev_l = float(low.iloc[-2]) if len(low) > 1 else cur_l
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        vwap = float((vol * (high + low + close) / 3).sum() / vol.sum()) if vol.sum() > 0 else p
        
        sma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)
        
        pp = (cur_h + cur_l + p) / 3
        swing_range = high.tail(10).max() - low.tail(10).min()
        
        bull_score = sum([
            1 if p > pp else 0,
            1 if p > vwap else 0,
            1 if p > sma20 else 0,
            1 if chg > 0 else 0
        ])
        bull = bull_score >= 2

        # Plain English Decision & Advice
        if bull:
            action_advice = "ACCUMULATE / BUY"
            story = f"Buyers are active and stepping in to push the price up. The stock is trading safely above the fair average price (₹{vwap:.2f}), showing strong buyer support. <b>Decision:</b> Good time to consider buying or building a position."
            card_bg, border_c, accent_c = "#ecfdf5", "#059669", "#047857"
        else:
            action_advice = "DO NOT BUY / HOLD OFF"
            story = f"Sellers are currently in control and pushing the price down. The stock is trading below the fair average price (₹{vwap:.2f}), showing that big buyers are absent. <b>Decision:</b> Better to wait and protect your money."
            card_bg, border_c, accent_c = "#fef2f2", "#dc2626", "#b91c1c"

        nc_h = min(p + (atr * 0.12), bb_upper)
        nc_l = max(p - (atr * 0.12), bb_lower)
        
        nd_h1 = (2 * pp) - cur_l
        nd_h2 = p + (atr * 0.8)
        nd_l1 = (2 * pp) - cur_h
        nd_l2 = p - (atr * 0.8)
        
        nw_h1 = p + (swing_range * 0.5) + (atr * 1.2)
        nw_h2 = p + swing_range + (atr * 2.0)
        nw_l1 = p - (swing_range * 0.5) - (atr * 1.2)
        nw_l2 = p - swing_range - (atr * 2.0)

        ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        current_time = ist_now.time()
        market_open = time(9, 15)
        market_close = time(15, 30)
        is_weekday = ist_now.weekday() < 5
        
        if is_weekday and market_open <= current_time <= market_close:
            market_status = "🟢 Market is OPEN (Live Sync Active)"
        elif is_weekday and current_time < market_open:
            market_status = "🟡 Pre-Market Time (Opens at 09:15 AM)"
        else:
            market_status = "🔴 Market is CLOSED (Showing Last Settled Session Data)"

        st.markdown(f"<h2 style='margin:0; font-size:20px; color:#0f172a;'>📍 {ticker} | <span style='color:{accent_c};'>₹{p:.2f} ({chg:+.2f}%)</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px; font-weight:700; color:#475569; margin-top:4px;'>{market_status} (IST: {ist_now.strftime('%H:%M:%S')})</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#cbd5e1; margin:10px 0;'>", unsafe_allow_html=True)

        # TOP ROW: Expert Decision, Previous Candle, Next Candle
        top_c1, top_c2, top_c3 = st.columns([2, 1, 1], gap="medium")
        
        with top_c1:
            st.markdown(f"""
                <div class="card" style="background: {card_bg}; border: 2px solid {border_c};">
                    <div class="title" style="color: {accent_c};">Simple Plain-English Review</div>
                    <div style="font-size: 18px; font-weight: 900; color: {accent_c}; margin-top: 8px;">{action_advice}</div>
                    <div style="font-size: 12px; color: #0f172a; line-height: 1.5; margin-top: 8px;">
                        {story}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with top_c2:
            st.markdown(f"""
                <div class="card">
                    <div class="title">Previous Candle</div>
                    <div style="font-size: 11px; margin-top: 6px; color: #0f172a; line-height: 1.5;">
                        ▲ <b>High:</b> ₹{prev_h:.2f}<br>
                        ▼ <b>Low:</b> ₹{prev_l:.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with top_c3:
            st.markdown(f"""
                <div class="card">
                    <div class="title">Next Candle</div>
                    <div style="font-size: 11px; margin-top: 6px; color: #0f172a; line-height: 1.5;">
                        ▲ <b>High:</b> ₹{nc_h:.2f}<br>
                        ▼ <b>Low:</b> ₹{nc_l:.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # BOTTOM ROW: Next Day Targets & Next Week Targets Side-by-Side
        bot_c1, bot_c2 = st.columns(2, gap="medium")

        with bot_c1:
            st.markdown(f"""
                <div class="card">
                    <div class="title">Next Day Targets</div>
                    <div style="display: flex; gap: 12px; margin-top: 6px;">
                        <div style="flex: 1;">
                            <div style="font-size: 9px; font-weight: 800; color: #047857; margin-bottom: 2px;">HIGHS</div>
                            <span style="font-size: 11px; color: #0f172a;">🟢 H1: ₹{nd_h1:.2f}</span><br>
                            <span style="font-size: 11px; color: #0f172a;">🟢 H2: ₹{nd_h2:.2f}</span>
                        </div>
                        <div style="flex: 1;">
                            <div style="font-size: 9px; font-weight: 800; color: #b91c1c; margin-bottom: 2px;">LOWS</div>
                            <span style="font-size: 11px; color: #0f172a;">🔴 L1: ₹{nd_l1:.2f}</span><br>
                            <span style="font-size: 11px; color: #0f172a;">🔴 L2: ₹{nd_l2:.2f}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with bot_c2:
            st.markdown(f"""
                <div class="card">
                    <div class="title">Next Week Targets</div>
                    <div style="display: flex; gap: 12px; margin-top: 6px;">
                        <div style="flex: 1;">
                            <div style="font-size: 9px; font-weight: 800; color: #047857; margin-bottom: 2px;">HIGHS</div>
                            <span style="font-size: 11px; color: #0f172a;">🟢 H1: ₹{nw_h1:.2f}</span><br>
                            <span style="font-size: 11px; color: #0f172a;">🟢 H2: ₹{nw_h2:.2f}</span>
                        </div>
                        <div style="flex: 1;">
                            <div style="font-size: 9px; font-weight: 800; color: #b91c1c; margin-bottom: 2px;">LOWS</div>
                            <span style="font-size: 11px; color: #0f172a;">🔴 L1: ₹{nw_l1:.2f}</span><br>
                            <span style="font-size: 11px; color: #0f172a;">🔴 L2: ₹{nw_l2:.2f}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # --- LIVE INTRADAY GRAPH SECTION ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>Today's Live Intraday Price Chart (5-Minute Ticks)</div>", unsafe_allow_html=True)
        if not idf.empty:
            intraday_chart_df = idf[['Close']].copy()
            intraday_chart_df.columns = ['Live Price']
            st.line_chart(intraday_chart_df, color="#047857" if bull else "#b91c1c", height=250)
        else:
            st.info("Intraday live data is settling. Showing last settled state.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("💡 Type an Indian stock ticker above (like TEJASNET, RELIANCE, or TCS) to execute the live multi-formula engine.")
