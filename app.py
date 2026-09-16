import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Discretionary Market Analysis Terminal", page_icon="🏛️", layout="wide")

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
    st.markdown("<h2 style='color: white; margin: 0;'>Discretionary Market Terminal</h2>", unsafe_allow_html=True)

header_stock_placeholder = hc_stock.empty()
if not ticker:
    header_stock_placeholder.markdown("<div style='text-align: right; color: #8c96a5; font-size: 13px; padding-top: 10px;'>📍 No Asset Selected</div>", unsafe_allow_html=True)

st.markdown("---")

if ticker:
    @st.fragment(run_every=15)
    def render_discretionary_analysis():
        try:
            with st.spinner(f"Executing veteran structural market analysis for {ticker}..."):
                stock = yf.Ticker(ticker)
                
                try:
                    co_name = stock.info.get('longName', ticker)
                except:
                    co_name = ticker

                df_raw = stock.history(period="1d", interval="5m")

                if df_raw.empty or len(df_raw) < 2:
                    header_stock_placeholder.markdown(f"<div style='text-align: right; color: #eb5b3c; font-size: 13px; padding-top: 6px;'>📍 {co_name} (Liquidity Inactive)</div>", unsafe_allow_html=True)
                    st.error(f"❌ Real-time tape data unavailable for '{ticker}'. Confirm active market hours.")
                    return

                if df_raw.index.tz is not None:
                    df_raw.index = df_raw.index.tz_localize(None)

                df_raw = df_raw.between_time('09:15', '15:30')

                if df_raw.empty:
                    st.error("❌ No price action recorded within standard session bounds (09:15 - 15:30).")
                    return

                df_candles = df_raw.resample('10min', origin='09:15:00', closed='left', label='left').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                if df_candles.empty:
                    st.error("❌ Insufficient interval bars to establish market structure.")
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

                # --- VETERAN DISCRETIONARY STRUCTURAL ASSESSMENT ---
                # Analyzing structural balance: supply/demand zones, volume expansion, and range behavior
                recent_high = df_candles['High'].max()
                recent_low = df_candles['Low'].min()
                
                # Volume profile evaluation
                avg_vol = df_candles['Volume'].mean()
                latest_vol = df_candles['Volume'].iloc[-1]
                volume_context = "High Volume Expansion" if latest_vol > (1.2 * avg_vol) else ("Low Volume Drift" if latest_vol < (0.8 * avg_vol) else "Normal Volume Participation")

                # Price structure alignment
                mid_range = (recent_high + recent_low) / 2
                position_in_range = "Upper Quartile (Supply Zone)" if current_price > mid_range else "Lower Quartile (Demand Zone)"

                # Qualitative Market Bias Decision
                # Evaluating momentum via candle structure rather than blind algorithmic outputs
                last_candle_bullish = df_candles['Close'].iloc[-1] >= df_candles['Open'].iloc[-1]
                trend_bias = "BULLISH ACCUMULATION 📈" if last_candle_bullish and current_price >= session_open else "BEARISH DISTRIBUTION 📉"
                card_background = "#00d09c" if "BULLISH" in trend_bias else "#eb5b3c"

                discretionary_rationale = (
                    f"Structural Context: Price is trading in the {position_in_range}. "
                    f"Tape activity shows {volume_context.lower()}. "
                    f"Recent price action reflects {'buying commitment near session lows' if last_candle_bullish else 'selling pressure or lack of aggressive sponsorship'}."
                )

                # Structural Support & Resistance boundaries
                support_level = recent_low
                resistance_level = recent_high

            # Render Discretionary Dashboard Cards
            dc1, dc2 = st.columns([1.5, 1.5])
            with dc1:
                st.markdown(
                    f"""<div class="card" style="background: {card_background}; color: #0f141e; padding: 16px; border-radius: 8px;">
                    <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;">DISCRETIONARY MARKET BIAS</div>
                    <div style="font-size: 16px; font-weight: 900; margin: 4px 0;">{trend_bias}</div>
                    <div style="font-size: 11px; font-weight: 700; line-height: 1.4; margin-top: 6px; background: rgba(0,0,0,0.12); padding: 6px 8px; border-radius: 4px;">
                        {discretionary_rationale}
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with dc2:
                st.markdown(
                    f"""<div class="card" style="background: #1c212b; border: 1px solid #28303d; color: #f0f4f8; padding: 16px; border-radius: 8px;">
                    <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; color: #8c96a5;">KEY STRUCTURAL LEVELS & RISK BOUNDARIES</div>
                    <div style="font-size: 12px; font-weight: 700; margin-top: 8px; color: #00d09c;">
                        🛡️ Immediate Structural Support: ₹{support_level:.2f}
                    </div>
                    <div style="font-size: 12px; font-weight: 700; margin-top: 6px; color: #eb5b3c;">
                        🚧 Immediate Overhead Supply: ₹{resistance_level:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 8px; color: #8c96a5;">
                        *Rule of Discipline: Manage risk relative to structural invalidation points rather than fixed targets.
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.subheader("⚡ Live 10-Minute Market Execution Records")
            
            display_df = df_candles.tail(10).reset_index()
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
            st.error(f"⚠️ Error parsing discretionary market context for {ticker}. Details: {e}")

    render_discretionary_analysis()
else:
    st.info(
        "Please select your exchange and input a stock ticker to initiate discretionary market structure analysis."
    )
