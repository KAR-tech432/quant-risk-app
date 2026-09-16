import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Expert Market Analysis Terminal", page_icon="🏛️", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #0f141e; color: #f0f4f8; font-family: sans-serif; }
        .block-container { padding: 1rem 1.5rem; }
        .card { background: #1c212b; border: 1px solid #28303d; border-radius: 8px; padding: 14px; }
        div.stButton > button { background-color: #28303d; color: #00d09c; border: 1px solid #00d09c; font-weight: 600; border-radius: 6px; }
        div.stButton > button:hover { background-color: #00d09c; color: #0f141e; }
    </style>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    ex_label = st.selectbox("Market Exchange:", ["NSE", "BSE"])
    ex = ".NS" if ex_label == "NSE" else ".BO"
with c2:
    inp = st.text_input(
        "Enter Stock Ticker (e.g., RELIANCE, TCS, INFY, SBIN):",
        value="",
        placeholder="Type symbol...",
    ).upper()
with c3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.rerun()

ticker = None
if inp:
    clean_inp = inp.replace(".NS", "").replace(".BO", "").replace(".BS", "").strip()
    ticker = f"{clean_inp}{ex}"

hc_title, hc_stock = st.columns([2, 2])
with hc_title:
    st.markdown("<h2 style='color: white; margin: 0;'>Expert Market Analysis Terminal</h2>", unsafe_allow_html=True)

header_stock_placeholder = hc_stock.empty()
if not ticker:
    header_stock_placeholder.markdown("<div style='text-align: right; color: #8c96a5; font-size: 13px; padding-top: 10px;'>📍 No Asset Selected</div>", unsafe_allow_html=True)

st.markdown("---")

if ticker:
    @st.fragment(run_every=30)
    def render_expert_analysis():
        try:
            with st.spinner(f"Analyzing complete price action & tape behavior for {ticker}..."):
                stock = yf.Ticker(ticker)
                
                try:
                    co_name = stock.info.get('longName', ticker)
                except:
                    co_name = ticker

                # Fetching 5-day history to ensure we capture the most recent completed session data cleanly
                df_raw = stock.history(period="5d", interval="5m")

                if df_raw.empty:
                    header_stock_placeholder.markdown(f"<div style='text-align: right; color: #eb5b3c; font-size: 13px; padding-top: 6px;'>📍 {co_name} (No Data)</div>", unsafe_allow_html=True)
                    st.error(f"❌ Price telemetry unavailable for '{ticker}'.")
                    return

                if df_raw.index.tz is not None:
                    df_raw.index = df_raw.index.tz_localize(None)

                # Filter for the latest available trading day session
                latest_date = df_raw.index.date[-1]
                df_day = df_raw[df_raw.index.date == latest_date].between_time('09:15', '15:30')

                if df_day.empty:
                    df_day = df_raw.tail(75) # Fallback to latest records if session time boundaries mismatch

                df_candles = df_day.resample('10min', origin='09:15:00', closed='left', label='left').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                if df_candles.empty:
                    st.error("❌ Insufficient interval ticks to formulate market structure.")
                    return

                current_price = float(df_candles["Close"].iloc[-1])
                session_open = float(df_candles["Open"].iloc[0])
                session_chg = ((current_price - session_open) / session_open) * 100
                chg_color = "#00d09c" if session_chg >= 0 else "#eb5b3c"

                header_stock_placeholder.markdown(
                    f"<div style='text-align: right; color: #f0f4f8; font-size: 14px; font-weight: 700; padding-top: 4px;'>"
                    f"📍 {co_name} <span style='color: #00d09c;'>| CMP: ₹{current_price:.2f}</span> "
                    f"<span style='color: {chg_color}; font-size: 12px;'>({session_chg:+.2f}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # --- 30-YEAR VETERAN MATHEMATICAL & MARKET ANALYSIS ---
                recent_high = df_candles['High'].max()
                recent_low = df_candles['Low'].min()
                avg_vol = df_candles['Volume'].mean()
                latest_vol = df_candles['Volume'].iloc[-1]
                
                # Simple Language Rationale Generation
                mid_range = (recent_high + recent_low) / 2
                is_upper_half = current_price > mid_range
                volume_heavy = latest_vol > (1.1 * avg_vol)
                last_bar_green = df_candles['Close'].iloc[-1] >= df_candles['Open'].iloc[-1]

                if last_bar_green and is_upper_half:
                    bias = "BULLISH MOMENTUM 📈"
                    card_bg = "#00d09c"
                    simple_reason = "The stock is holding firmly in the upper half of its daily range with solid buyer backing. Sellers are currently failing to push prices lower."
                elif not last_bar_green and not is_upper_half:
                    bias = "BEARISH DISTRIBUTION 📉"
                    card_bg = "#eb5b3c"
                    simple_reason = "The stock is sliding near the lower end of its daily range. Sellers are active, and buyers are struggling to build upward momentum."
                else:
                    bias = "CONSOLIDATION / NEUTRAL ⚖️"
                    card_bg = "#f39c12"
                    simple_reason = "The stock is moving sideways within a tight zone. Both buyers and sellers are balanced, waiting for a breakout direction."

                # --- TARGET CALCULATIONS (Pure Mathematical Projections) ---
                # 1. Next Candle Projections
                candle_range = (recent_high - recent_low) / len(df_candles) if len(df_candles) > 0 else (current_price * 0.005)
                next_candle_high = current_price + (candle_range * 0.8)
                next_candle_low = current_price - (candle_range * 0.8)

                # 2. Next Day Projections (Professional Pivot Math)
                pivot = (recent_high + recent_low + current_price) / 3
                next_day_high = (2 * pivot) - recent_low
                next_day_low = (2 * pivot) - recent_high

                # 3. Next Week Projections (Multi-session swing range scale)
                week_range = recent_high - recent_low
                next_week_high = current_price + (week_range * 0.6)
                next_week_low = current_price - (week_range * 0.6)

            # --- RENDER DASHBOARD CARDS ---
            dc1, dc2, dc3 = st.columns([1.2, 1.2, 1.2])
            
            with dc1:
                st.markdown(
                    f"""<div class="card" style="background: {card_bg}; color: #0f141e; padding: 14px; border-radius: 8px;">
                    <div style="font-size: 10px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;">MARKET BIAS & ANALYSIS</div>
                    <div style="font-size: 15px; font-weight: 900; margin: 4px 0;">{bias}</div>
                    <div style="font-size: 11px; font-weight: 700; line-height: 1.35; margin-top: 6px; background: rgba(0,0,0,0.12); padding: 6px 8px; border-radius: 4px;">
                        {simple_reason}
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )
            
            with dc2:
                st.markdown(
                    f"""<div class="card" style="background: #1c212b; border: 1px solid #28303d; color: #f0f4f8; padding: 14px; border-radius: 8px;">
                    <div style="font-size: 10px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; color: #8c96a5;">TARGET PROJECTIONS</div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 6px; color: #f0f4f8;">
                        ⏳ <b>Next Candle:</b> <span style="color: #00d09c;">H: ₹{next_candle_high:.2f}</span> | <span style="color: #eb5b3c;">L: ₹{next_candle_low:.2f}</span>
                    </div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 6px; color: #f0f4f8;">
                        📅 <b>Next Day:</b> <span style="color: #00d09c;">H: ₹{next_day_high:.2f}</span> | <span style="color: #eb5b3c;">L: ₹{next_day_low:.2f}</span>
                    </div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 6px; color: #f0f4f8;">
                        📈 <b>Next Week:</b> <span style="color: #00d09c;">H: ₹{next_week_high:.2f}</span> | <span style="color: #eb5b3c;">L: ₹{next_week_low:.2f}</span>
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            with dc3:
                st.markdown(
                    f"""<div class="card" style="background: #1c212b; border: 1px solid #28303d; color: #f0f4f8; padding: 14px; border-radius: 8px;">
                    <div style="font-size: 10px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; color: #8c96a5;">KEY STRUCTURAL BOUNDARIES</div>
                    <div style="font-size: 12px; font-weight: 700; margin-top: 8px; color: #00d09c;">
                        🛡️ Session Support: ₹{recent_low:.2f}
                    </div>
                    <div style="font-size: 12px; font-weight: 700; margin-top: 6px; color: #eb5b3c;">
                        🚧 Session Resistance: ₹{recent_high:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 8px; color: #8c96a5;">
                        *Discipline: Base risk management on these exact high/low pivot boundaries.
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.subheader("⚡ Completed Session 10-Minute Market Execution Records")
            
            display_df = df_candles.tail(12).reset_index()
            time_col = "Datetime" if "Datetime" in display_df.columns else ("Date" if "Date" in display_df.columns else display_df.columns[0])
            
            formatted_rows = []
            for _, row in display_df.iterrows():
                dt_val = pd.to_datetime(row[time_col])
                t_stamp = dt_val.strftime('%H:%M')
                
                c_open = float(row["Open"])
                c_high = float(row["High"])
                c_low = float(row["Low"])
                c_close = float(row["Close"])
                c_vol = int(row["Volume"])
                
                bar_state = "Bullish Bar 🟢" if c_close >= c_open else "Bearish Bar 🔴"
                
                formatted_rows.append({
                    "Interval Time": t_stamp,
                    "Open": f"₹{c_open:.2f}",
                    "High": f"₹{c_high:.2f}",
                    "Low": f"₹{c_low:.2f}",
                    "Close": f"₹{c_close:.2f}",
                    "Volume": f"{c_vol:,}",
                    "Bar Context": bar_state
                })

            st.dataframe(pd.DataFrame(formatted_rows).iloc[::-1], use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error executing analysis for {ticker}. Details: {e}")

    render_expert_analysis()
else:
    st.info(
        "Please select your exchange and input a stock ticker to view full session market records and expert targets."
    )
