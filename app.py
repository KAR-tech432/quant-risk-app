import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live 5-Min Candle Sync Engine", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #0f141e; color: #f0f4f8; font-family: sans-serif; }
        .block-container { padding: 1rem 1.5rem; }
        .card { background: #1c212b; border: 1px solid #28303d; border-radius: 8px; padding: 12px; }
        div[data-testid="stMetric"] { background: #1c212b; border: 1px solid #28303d; padding: 8px 12px; border-radius: 8px; }
        div[data-testid="stMetric"] label { font-size: 11px !important; color: #8c96a5 !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 16px !important; color: #fff !important; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h2 style='color: white; margin-bottom: 0;'>Live 5-Minute Candle Interval Synchronization Monitor</h2>",
    unsafe_allow_html=True,
)
st.markdown("---")

c1, c2 = st.columns([1, 2])
with c1:
    ex_label = st.selectbox("Market Exchange:", ["NSE", "BSE"])
    ex = ".NS" if ex_label == "NSE" else ".BO"
with c2:
    inp = st.text_input(
        "Enter Stock Ticker (e.g., TEJASNET, RELIANCE, TCS):",
        value="",
        placeholder="Type symbol...",
    ).upper()

ticker = None
if inp:
    clean_inp = inp.replace(".NS", "").replace(".BO", "").replace(".BS", "").strip()
    ticker = f"{clean_inp}{ex}"

if ticker:
    try:
        with st.spinner(f"Synchronizing live 5-minute session candles for {ticker}..."):
            stock = yf.Ticker(ticker)
            # Fetching accurate intraday 5-minute blocks
            df_candles = stock.history(period="1d", interval="5m")

            if df_candles.empty or len(df_candles) < 2:
                st.error(
                    f"❌ Live 5-minute candle data is currently unavailable for '{ticker}'. Please ensure the market session is active."
                )
                st.stop()

            current_price = float(df_candles["Close"].iloc[-1])
            session_open = float(df_candles["Open"].iloc[0])
            session_chg = ((current_price - session_open) / session_open) * 100

            # Live active candle direction status based on latest 5m bar close vs open
            latest_open = float(df_candles["Open"].iloc[-1])
            latest_close = float(df_candles["Close"].iloc[-1])
            is_upward = latest_close >= latest_open
            status_text = "UPWARD TICK (BULLISH BAR)" if is_upward else "DOWNWARD TICK (BEARISH BAR)"
            status_color = "#00d09c" if is_upward else "#eb5b3c"

        hc1, hc2 = st.columns([1.5, 1])
        with hc1:
            st.markdown(
                f"""<div class="card" style="padding: 22px 18px;">
                <h4 style="margin:0; color:white; font-size:18px;">{ticker} <span style="font-size:12px; color:#8c96a5;">(Live Candle Feed)</span></h4>
                <p style="color:#8c96a5; margin:4px 0; font-size:12px;">Exchange: <b>{ex_label}</b> | Total 5m Bars: <b>{len(df_candles)}</b></p>
                <h4 style="margin:8px 0 0 0; color:#00d09c; font-size:18px;">₹{current_price:.2f} <span style="font-size:12px; color:{'#00d09c' if session_chg >= 0 else '#eb5b3c'};">({session_chg:+.2f}%)</span></h4>
            </div>""",
                unsafe_allow_html=True,
            )
        with hc2:
            st.markdown(
                f"""<div class="card" style="background: {status_color}; color: #0f141e; text-align: center; padding: 22px 18px;">
                <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.5px;">LIVE 5-MIN BAR STATUS</div>
                <div style="font-size: 20px; font-weight: 900; margin: 6px 0;">{status_text}</div>
                <div style="font-size: 12px; font-weight: 600;">Synced directly to active candle intervals.</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live LTP", f"₹{current_price:.2f}", f"{session_chg:+.2f}%")
        m2.metric("Active Bar Volume", f"{int(df_candles['Volume'].iloc[-1]):,}", "Latest 5m Block")
        m3.metric("Session High", f"₹{df_candles['High'].max():.2f}", "Peak")
        m4.metric("Session Low", f"₹{df_candles['Low'].min():.2f}", "Floor")

        st.markdown("---")

        st.subheader("⚡ Live-Synced 5-Minute Candle Intervals Table")
        
        # Format the dataframe cleanly to display exact candle timestamps and parameters without estimations
        display_df = df_candles.tail(10).reset_index()
        
        # Normalize timestamp column name cleanly across different pandas/yfinance builds
        time_col = "Datetime" if "Datetime" in display_df.columns else ("Date" if "Date" in display_df.columns else display_df.columns[0])
        
        formatted_rows = []
        for _, row in display_df.iterrows():
            t_stamp = str(row[time_col])
            c_open = float(row["Open"])
            c_high = float(row["High"])
            c_low = float(row["Low"])
            c_close = float(row["Close"])
            c_vol = int(row["Volume"])
            
            bar_dir = "UPWARD 📈" if c_close >= c_open else "DOWNWARD 📉"
            
            formatted_rows.append({
                "5-Min Candle Timestamp": t_stamp,
                "Open": f"₹{c_open:.2f}",
                "High": f"₹{c_high:.2f}",
                "Low": f"₹{c_low:.2f}",
                "Close": f"₹{c_close:.2f}",
                "Volume": f"{c_vol:,}",
                "Actual Direction": bar_dir
            })

        st.dataframe(pd.DataFrame(formatted_rows).iloc[::-1], use_container_width=True)

    except Exception as e:
        st.error(
            f"⚠️ Error fetching live 5-minute candle timestamps for {ticker}. Details: {e}"
        )
else:
    st.info(
        "Please select your exchange and enter a stock ticker above to view live-synchronized 5-minute candle intervals."
    )
