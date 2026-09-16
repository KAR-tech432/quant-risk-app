import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Configure Enterprise Mobile-Responsive Viewport & Layout
st.set_page_config(
    page_title="Enterprise Quantitative Market Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-End Enterprise CSS with Optimized Spacing & Gaps
st.markdown(
    """
    <style>
        .stApp { background-color: #090d16; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .block-container { padding: 1.5rem 1.5rem !important; max-width: 100%; }
        
        /* Enterprise Card Styling with Proper Padding */
        .enterprise-card {
            background: linear-gradient(135deg, #131b2e 0%, #0f1726 100%);
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        
        .card-title { font-size: 11px; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; }
        .card-main-val { font-size: 18px; font-weight: 900; margin: 4px 0; }
        .card-sub-text { font-size: 12px; font-weight: 600; line-height: 1.4; margin-top: 8px; padding: 8px; border-radius: 6px; background: rgba(0, 0, 0, 0.2); }
        
        /* Input & Controls Spacing Optimization */
        div.stButton > button {
            background: linear-gradient(135deg, #1e293b 0%, #0f1726 100%);
            color: #38bdf8;
            border: 1px solid #38bdf8;
            font-weight: 700;
            border-radius: 6px;
            width: 100%;
            padding: 10px;
        }
        div.stButton > button:hover { background-color: #38bdf8; color: #090d16; }
        
        .input-spacer { height: 6px; }
    </style>
""",
    unsafe_allow_html=True,
)

# Top Bar Input Controls with Clean Spacing Gaps
col_ex, col_input, col_btn = st.columns([1, 2.5, 1.2], gap="medium")
with col_ex:
    st.markdown("<div class='input-spacer'></div>", unsafe_allow_html=True)
    ex_label = st.selectbox("Market Exchange", ["NSE", "BSE"])
    ex = ".NS" if ex_label == "NSE" else ".BO"
with col_input:
    st.markdown("<div class='input-spacer'></div>", unsafe_allow_html=True)
    inp = st.text_input(
        "Symbol Ticker",
        value="",
        placeholder="e.g. RELIANCE, TCS, SBIN",
    ).upper()
with col_btn:
    st.markdown("<div style='height: 31px;'></div>", unsafe_allow_html=True)
    refresh_clicked = st.button("🔄 Execute Analysis", use_container_width=True)

ticker = None
if inp:
    clean_inp = inp.replace(".NS", "").replace(".BO", "").replace(".BS", "").strip()
    ticker = f"{clean_inp}{ex}"

# Header Section with Spacing Gap
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
head_c1, head_c2 = st.columns([2, 2])
with head_c1:
    st.markdown("<h3 style='color: #f8fafc; margin: 0; font-size: 20px;'>Enterprise Quant Terminal</h3>", unsafe_allow_html=True)

header_stock_placeholder = head_c2.empty()
if not ticker:
    header_stock_placeholder.markdown("<div style='text-align: right; color: #64748b; font-size: 12px; padding-top: 6px;'>📍 Status: Standby (Awaiting Ticker)</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 15px 0; border-color: #1e293b;'>", unsafe_allow_html=True)

if ticker:
    @st.fragment(run_every=30)
    def run_enterprise_engine():
        try:
            with st.spinner(f"Computing advanced mathematical matrix for {ticker}..."):
                stock = yf.Ticker(ticker)
                
                try:
                    co_name = stock.info.get('longName', ticker)
                except:
                    co_name = ticker

                df_raw = stock.history(period="5d", interval="5m")

                if df_raw.empty:
                    header_stock_placeholder.markdown(f"<div style='text-align: right; color: #f87171; font-size: 12px;'>📍 {co_name} (Data Offline)</div>", unsafe_allow_html=True)
                    st.error(f"❌ Real-time feed unavailable for '{ticker}'. Verify ticker spelling.")
                    return

                if df_raw.index.tz is not None:
                    df_raw.index = df_raw.index.tz_localize(None)

                latest_date = df_raw.index.date[-1]
                df_day = df_raw[df_raw.index.date == latest_date].between_time('09:15', '15:30')
                if df_day.empty:
                    df_day = df_raw.tail(75)

                current_price = float(df_day["Close"].iloc[-1])
                session_open = float(df_day["Open"].iloc[0])
                session_chg = ((current_price - session_open) / session_open) * 100
                chg_color = "#34d399" if session_chg >= 0 else "#f87171"

                header_stock_placeholder.markdown(
                    f"<div style='text-align: right; color: #f8fafc; font-size: 13px; font-weight: 700; padding-top: 4px;'>"
                    f"📍 {co_name} <span style='color: #34d399;'>| CMP: ₹{current_price:.2f}</span> "
                    f"<span style='color: {chg_color}; font-size: 11px;'>({session_chg:+.2f}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # --- 30-YEAR EXPERT MATHEMATICAL & QUANTITATIVE ENGINE ---
                H = float(df_day['High'].max())
                L = float(df_day['Low'].min())
                C = current_price
                daily_range = H - L
                
                mid_point = (H + L) / 2
                last_bar_bullish = df_day['Close'].iloc[-1] >= df_day['Open'].iloc[-1]
                
                # Market Bias & Expert Decision Logic
                if last_bar_bullish and C >= mid_point:
                    bias = "BULLISH ACCUMULATION 📈"
                    expert_decision = "ACCUMULATE / BUY"
                    card_bg_color = "linear-gradient(135deg, #064e3b 0%, #022c22 100%)"
                    border_color = "#059669"
                    simple_reason = "Price is holding firm in the upper half of the session boundary. Institutional bids are absorbing supply near highs."
                elif not last_bar_bullish and C < mid_point:
                    bias = "BEARISH DISTRIBUTION 📉"
                    expert_decision = "SELL / REDUCE"
                    card_bg_color = "linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%)"
                    border_color = "#dc2626"
                    simple_reason = "Price is trading suppressed in the lower half of the session range. Aggressive selling pressure dominates order flow."
                else:
                    bias = "EQUILIBRIUM / CHOP ⚖️"
                    expert_decision = "HOLD / WAIT"
                    card_bg_color = "linear-gradient(135deg, #78350f 100%, #451a03 100%)"
                    border_color = "#d97706"
                    simple_reason = "Price is rotating near the midpoint balance zone. Market participants are locked in a tight consolidation bracket."

                # Projections & Target Calculations (Dual Values)
                micro_step = daily_range / max(len(df_day), 10) * 0.75
                nc_high = C + micro_step
                nc_low = C - micro_step

                PP = (H + L + C) / 3
                nd_h1 = (2 * PP) - L
                nd_h2 = PP + daily_range
                nd_l1 = (2 * PP) - H
                nd_l2 = PP - daily_range

                nw_h1 = C + (daily_range * 1.1)
                nw_h2 = C + (daily_range * 2.0)
                nw_l1 = C - (daily_range * 1.1)
                nw_l2 = C - (daily_range * 2.0)

                s1 = (2 * PP) - H
                s2 = L - (H - PP)
                r1 = (2 * PP) - L
                r2 = H + (PP - L)

            # --- RENDER ENTERPRISE DASHBOARD GRID ---
            row1_c1, row1_c2 = st.columns(2, gap="medium")
            
            with row1_c1:
                st.markdown(
                    f"""<div class="enterprise-card" style="background: {card_bg_color}; border: 1px solid {border_color};">
                        <div class="card-title">Proprietary Market Bias & Decision</div>
                        <div class="card-main-val" style="color: #ffffff;">{bias}</div>
                        <div style="font-size: 13px; font-weight: 800; color: #38bdf8; margin-top: 4px;">Expert Recommendation: {expert_decision}</div>
                        <div class="card-sub-text" style="background: rgba(0,0,0,0.25); color: #f1f5f9;">{simple_reason}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            with row1_c2:
                st.markdown(
                    f"""<div class="enterprise-card">
                        <div class="card-title">Next Candle Projection (Micro)</div>
                        <div class="card-sub-text" style="background: rgba(255,255,255,0.03); margin-top: 0px;">
                            <span style="color: #34d399;">▲ Target High: ₹{nc_high:.2f}</span><br>
                            <span style="color: #f87171;">▼ Target Low: ₹{nc_low:.2f}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

            row2_c1, row2_c2, row2_c3 = st.columns(3, gap="medium")
            
            with row2_c1:
                st.markdown(
                    f"""<div class="enterprise-card">
                        <div class="card-title">Next Day Targets</div>
                        <div style="font-size: 11px; margin-top: 4px;">
                            <span style="color: #34d399; font-weight: 700;">H1: ₹{nd_h1:.2f}</span><br>
                            <span style="color: #34d399; font-weight: 700;">H2: ₹{nd_h2:.2f}</span>
                        </div>
                        <div style="font-size: 11px; margin-top: 6px;">
                            <span style="color: #f87171; font-weight: 700;">L1: ₹{nd_l1:.2f}</span><br>
                            <span style="color: #f87171; font-weight: 700;">L2: ₹{nd_l2:.2f}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

            with row2_c2:
                st.markdown(
                    f"""<div class="enterprise-card">
                        <div class="card-title">Next Week Targets</div>
                        <div style="font-size: 11px; margin-top: 4px;">
                            <span style="color: #34d399; font-weight: 700;">H1: ₹{nw_h1:.2f}</span><br>
                            <span style="color: #34d399; font-weight: 700;">H2: ₹{nw_h2:.2f}</span>
                        </div>
                        <div style="font-size: 11px; margin-top: 6px;">
                            <span style="color: #f87171; font-weight: 700;">L1: ₹{nw_l1:.2f}</span><br>
                            <span style="color: #f87171; font-weight: 700;">L2: ₹{nw_l2:.2f}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

            with row2_c3:
                st.markdown(
                    f"""<div class="enterprise-card">
                        <div class="card-title">Session S/R Matrix</div>
                        <div style="font-size: 11px; margin-top: 4px;">
                            <span style="color: #f87171; font-weight: 700;">R1: ₹{r1:.2f}</span><br>
                            <span style="color: #f87171; font-weight: 700;">R2: ₹{r2:.2f}</span>
                        </div>
                        <div style="font-size: 11px; margin-top: 6px;">
                            <span style="color: #34d399; font-weight: 700;">S1: ₹{s1:.2f}</span><br>
                            <span style="color: #34d399; font-weight: 700;">S2: ₹{s2:.2f}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"⚠️ Calculation error for ticker {ticker}: {e}")

    run_enterprise_engine()
else:
    st.info("💡 **Enterprise Terminal Ready:** Please choose your market exchange and enter a target stock ticker above to render structural projections.")
