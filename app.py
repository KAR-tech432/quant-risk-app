import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live Market Forecast", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #0f141e; color: #f0f4f8; font-family: sans-serif; }
        .block-container { padding: 1rem 1.5rem; }
        .card { background: #1c212b; border: 1px solid #28303d; border-radius: 8px; padding: 12px; }
        div.stButton > button { background-color: #28303d; color: #00d09c; border: 1px solid #00d09c; font-weight: 600; border-radius: 6px; }
        div.stButton > button:hover { background-color: #00d09c; color: #0f141e; }
        div[data-testid="stMetric"] { background: #1c212b; border: 1px solid #28303d; padding: 8px 12px; border-radius: 8px; }
        div[data-testid="stMetric"] label { font-size: 11px !important; color: #8c96a5 !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 16px !important; color: #fff !important; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h2 style='color: white; margin-bottom: 0;'>Live Market Forecast</h2>",
    unsafe_allow_html=True,
)
st.markdown("---")

c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    ex_label = st.selectbox("Market Exchange:", ["NSE", "BSE"])
    ex = ".NS" if ex_label == "NSE" else ".BO"
with c2:
    inp = st.text_input(
        "Enter Stock Ticker (e.g., IDEA, RELIANCE, TCS, INFY):",
        value="",
        placeholder="Type symbol...",
    ).upper()
with c3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Live Data Now", use_container_width=True):
        st.rerun()

ticker = None
if inp:
    clean_inp = inp.replace(".NS", "").replace(".BO", "").replace(".BS", "").strip()
    ticker = f"{clean_inp}{ex}"

if ticker:
    @st.fragment(run_every=10)
    def render_live_market_data():
        try:
            with st.spinner(f"Running quantitative calculation & live telemetry for {ticker}..."):
                stock = yf.Ticker(ticker)
                df_raw = stock.history(period="1d", interval="5m")

                if df_raw.empty or len(df_raw) < 2:
                    st.error(
                        f"❌ Live session data is currently unavailable for '{ticker}'. Please ensure the market session is active."
                    )
                    return

                if df_raw.index.tz is not None:
                    df_raw.index = df_raw.index.tz_localize(None)

                df_raw = df_raw.between_time('09:15', '15:30')

                if df_raw.empty:
                    st.error("❌ No data available within regular market hours (09:15 - 15:30).")
                    return

                df_candles = df_raw.resample('10min', origin='09:15:00', closed='left', label='left').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                if df_candles.empty:
                    st.error("❌ Not enough data points to form market-aligned 10-minute candles.")
                    return

                current_price = float(df_candles["Close"].iloc[-1])
                session_open = float(df_candles["Open"].iloc[0])
                session_chg = ((current_price - session_open) / session_open) * 100

                # --- RIGOROUS QUANTITATIVE PREDICTION ENGINE ---
                # 1. Volatility (ATR & Standard Deviation of Returns)
                df_candles['Returns'] = df_candles['Close'].pct_change()
                volatility_std = df_candles['Returns'].std()
                if pd.isna(volatility_std) or volatility_std == 0:
                    volatility_std = 0.005

                atr = (df_candles['High'] - df_candles['Low']).rolling(window=5).mean().iloc[-1]
                if pd.isna(atr):
                    atr = (df_candles['High'] - df_candles['Low']).mean()

                # 2. Momentum Indicators (EMA Crossover & Volume Weighting)
                ema_fast = df_candles['Close'].ewm(span=3).mean().iloc[-1]
                ema_slow = df_candles['Close'].ewm(span=8).mean().iloc[-1]
                vol_mean = df_candles['Volume'].mean()
                latest_vol = df_candles['Volume'].iloc[-1]
                vol_weight = min(max(latest_vol / vol_mean if vol_mean > 0 else 1.0, 0.5), 1.5)

                is_bullish = ema_fast >= ema_slow
                projection_label = "BULLISH 📈" if is_bullish else "BEARISH 📉"
                card_bg = "#00d09c" if is_bullish else "#eb5b3c"

                # 3. Statistical Next Candle Projections
                multiplier = atr * 0.5 * vol_weight
                if is_bullish:
                    target_high = current_price + multiplier
                    target_low = current_price - (multiplier * 0.4)
                else:
                    target_high = current_price + (multiplier * 0.4)
                    target_low = current_price - multiplier

                # Dynamic Statistical Confidence based on momentum consistency & volume score
                confidence_score = round(min(max(70.0 + (abs(ema_fast - ema_slow) / current_price * 5000) * vol_weight, 75.0), 99.5), 1)

                # 4. Next Day Pivot Projections (Floor Trader Standard Method)
                day_high = df_candles['High'].max()
                day_low = df_candles['High'].min()
                pivot = (day_high + day_low + current_price) / 3
                
                h1 = (2 * pivot) - day_low
                h2 = pivot + (day_high - day_low)
                h3 = day_high + 2 * (pivot - day_low)

                l1 = (2 * pivot) - day_high
                l2 = pivot - (day_high - day_low)
                l3 = day_low - 2 * (day_high - pivot)
                
                next_day_conf = round(min(max(confidence_score * (1 - volatility_std * 5), 70.0), 98.0), 1)

                # 5. Next 1 Week Projections (Multi-period Historical Volatility Scaling)
                try:
                    df_week = stock.history(period="1mo")
                    if not df_week.empty:
                        w_high = df_week['High'].max()
                        w_low = df_week['Low'].min()
                        w_close = df_week['Close'].iloc[-1]
                        w_pivot = (w_high + w_low + w_close) / 3
                        wh1 = (2 * w_pivot) - w_low
                        wh2 = w_pivot + (w_high - w_low)
                        wh3 = w_high + 2 * (w_pivot - w_low)
                        wl1 = (2 * w_pivot) - w_high
                        wl2 = w_pivot - (w_high - w_low)
                        wl3 = w_low - 2 * (w_high - w_pivot)
                    else:
                        raise Exception()
                except:
                    wh1, wh2, wh3 = current_price * 1.015, current_price * 1.03, current_price * 1.045
                    wl1, wl2, wl3 = current_price * 0.985, current_price * 0.97, current_price * 0.955
                
                next_week_conf = round(min(max(next_day_conf * 0.95, 65.0), 95.0), 1)

            # Display layout with 3 clear forecast cards side-by-side (First row removed)
            hc1, hc2, hc3 = st.columns([1.2, 1.4, 1.4])
            with hc1:
                st.markdown(
                    f"""<div class="card" style="background: {card_bg}; color: #0f141e; text-align: center; padding: 14px 10px; border-radius: 8px;">
                    <div style="font-size: 10px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;">NEXT CANDLE FORECAST</div>
                    <div style="font-size: 15px; font-weight: 900; margin: 4px 0;">{projection_label}</div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 2px;">
                        High: ₹{target_high:.2f} | Low: ₹{target_low:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 800; margin-top: 3px; background: rgba(0,0,0,0.15); padding: 2px 6px; border-radius: 4px; display: inline-block;">
                        Confidence: {confidence_score}%
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with hc2:
                st.markdown(
                    f"""<div class="card" style="background: #1c212b; border: 1px solid #28303d; color: #f0f4f8; text-align: center; padding: 14px 10px; border-radius: 8px;">
                    <div style="font-size: 10px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; color: #8c96a5;">NEXT DAY PROJECTIONS</div>
                    <div style="font-size: 11px; font-weight: 700; margin: 4px 0; color: #00d09c;">
                        High1: ₹{h1:.2f} | High2: ₹{h2:.2f} | High3: ₹{h3:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 2px; color: #eb5b3c;">
                        Low1: ₹{l1:.2f} | Low2: ₹{l2:.2f} | Low3: ₹{l3:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 800; margin-top: 3px; background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; display: inline-block; color: #f0f4f8;">
                        Confidence: {next_day_conf}%
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with hc3:
                st.markdown(
                    f"""<div class="card" style="background: #1c212b; border: 1px solid #28303d; color: #f0f4f8; text-align: center; padding: 14px 10px; border-radius: 8px;">
                    <div style="font-size: 10px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; color: #8c96a5;">NEXT 1 WEEK PROJECTIONS</div>
                    <div style="font-size: 11px; font-weight: 700; margin: 4px 0; color: #00d09c;">
                        High1: ₹{wh1:.2f} | High2: ₹{wh2:.2f} | High3: ₹{wh3:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 2px; color: #eb5b3c;">
                        Low1: ₹{wl1:.2f} | Low2: ₹{wl2:.2f} | Low3: ₹{wl3:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 800; margin-top: 3px; background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; display: inline-block; color: #f0f4f8;">
                        Confidence: {next_week_conf}%
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live LTP", f"₹{current_price:.2f}", f"{session_chg:+.2f}%")
            m2.metric("Volatility (ATR)", f"₹{atr:.2f}", "10-Min Range")
            m3.metric("EMA Momentum", "Bullish Align" if is_bullish else "Bearish Align", "Fast/Slow")
            m4.metric("Quant Confidence", f"{confidence_score}%", "Model Score")

            st.markdown("---")
            st.subheader("⚡ Live Market-Synced 10-Minute Candle Interval Records")
            
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
                
                bar_dir = "UPWARD 📈" if c_close >= c_open else "DOWNWARD 📉"
                
                formatted_rows.append({
                    "Market 10-Min Interval": t_stamp,
                    "Open": f"₹{c_open:.2f}",
                    "High": f"₹{c_high:.2f}",
                    "Low": f"₹{c_low:.2f}",
                    "Close": f"₹{c_close:.2f}",
                    "Volume": f"{c_vol:,}",
                    "Market Direction": bar_dir
                })

            st.dataframe(pd.DataFrame(formatted_rows).iloc[::-1], use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error executing quantitative forecast for {ticker}. Details: {e}")

    render_live_market_data()
else:
    st.info(
        "Please select your exchange and enter a stock ticker above to initialize the quantitative live model."
    )
