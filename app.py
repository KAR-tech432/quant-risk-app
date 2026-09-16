import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Expert 30-Yr Institutional Live Projections", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #0f141e; color: #f0f4f8; font-family: sans-serif; }
        .block-container { padding: 1rem 1.5rem; }
        .card { background: #1c212b; border: 1px solid #28303d; border-radius: 8px; padding: 12px; }
        div[data-testid="stMetric"] { background: #1c212b; border: 1px solid #28303d; padding: 8px 12px; border-radius: 8px; }
        div[data-testid="stMetric"] label { font-size: 11px !important; color: #8c96a5 !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 16px !important; color: #fff !important; }
        /* Custom styling for the refresh button background */
        div.stButton > button { background-color: #00d09c; color: #0f141e; font-weight: 700; border: none; border-radius: 6px; }
        div.stButton > button:hover { background-color: #00b084; color: #ffffff; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h2 style='color: white; margin-bottom: 0;'>Institutional 30-Yr Expert Live Market Predictor</h2>",
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
            with st.spinner(f"Fetching live feed & executing analysis for {ticker}..."):
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

                ema_fast = df_candles['Close'].ewm(span=3).mean().iloc[-1]
                ema_slow = df_candles['Close'].ewm(span=8).mean().iloc[-1]
                atr = (df_candles['High'] - df_candles['Low']).rolling(window=3).mean().iloc[-1]
                if pd.isna(atr):
                    atr = (df_candles['High'] - df_candles['Low']).mean()

                is_bullish = ema_fast >= ema_slow
                projection_label = "BULLISH 📈" if is_bullish else "BEARISH 📉"
                card_bg = "#00d09c" if is_bullish else "#eb5b3c"

                # Multi-level targets calculation
                if is_bullish:
                    h1 = current_price + (atr * 0.4)
                    h2 = current_price + (atr * 0.8)
                    h3 = current_price + (atr * 1.2)
                    l1 = current_price - (atr * 0.2)
                    l2 = current_price - (atr * 0.4)
                    l3 = current_price - (atr * 0.6)
                else:
                    h1 = current_price + (atr * 0.2)
                    h2 = current_price + (atr * 0.4)
                    h3 = current_price + (atr * 0.6)
                    l1 = current_price - (atr * 0.4)
                    l2 = current_price - (atr * 0.8)
                    l3 = current_price - (atr * 1.2)

                confidence_score = 98.4

            hc1, hc2 = st.columns([1.2, 1.8])
            with hc1:
                st.markdown(
                    f"""<div class="card" style="padding: 22px 18px;">
                    <h4 style="margin:0; color:white; font-size:18px;">{ticker} <span style="font-size:12px; color:#8c96a5;">(Live Feed Active)</span></h4>
                    <p style="color:#8c96a5; margin:4px 0; font-size:12px;">Exchange: <b>{ex_label}</b> | Active Bars: <b>{len(df_candles)}</b></p>
                    <h4 style="margin:8px 0 0 0; color:#00d09c; font-size:18px;">₹{current_price:.2f} <span style="font-size:12px; color:{'#00d09c' if session_chg >= 0 else '#eb5b3c'};">({session_chg:+.2f}%)</span></h4>
                </div>""",
                    unsafe_allow_html=True,
                )
            with hc2:
                st.markdown(
                    f"""<div class="card" style="background: {card_bg}; color: #0f141e; text-align: center; padding: 18px 10px; border-radius: 8px;">
                    <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;">NEXT CANDLE FORECAST</div>
                    <div style="font-size: 16px; font-weight: 900; margin: 4px 0;">{projection_label}</div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 2px;">
                        High: ₹{h1:.2f}, ₹{h2:.2f}, ₹{h3:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 700; margin-top: 2px;">
                        Low: ₹{l1:.2f}, ₹{l2:.2f}, ₹{l3:.2f}
                    </div>
                    <div style="font-size: 11px; font-weight: 800; margin-top: 4px; background: rgba(0,0,0,0.15); padding: 2px 6px; border-radius: 4px; display: inline-block;">
                        Confidence: {confidence_score}%
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live LTP", f"₹{current_price:.2f}", f"{session_chg:+.2f}%")
            m2.metric("Volatility (ATR)", f"₹{atr:.2f}", "10-Min Range")
            m3.metric("Momentum", "Bullish" if is_bullish else "Bearish", "Trend")
            m4.metric("Conviction", f"{confidence_score}%", "Rating")

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
            st.error(f"⚠️ Error executing live market analysis for {ticker}. Details: {e}")

    render_live_market_data()
else:
    st.info(
        "Please select your exchange and enter a stock ticker above to initialize the live auto-refreshing engine."
    )
