import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live Tick-Sync Order Flow & Direction", page_icon="📈", layout="wide")

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
    "<h2 style='color: white; margin-bottom: 0;'>Live Tick-Sync Order Flow & Direction Monitor</h2>",
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
        with st.spinner(f"Syncing live session ticks and matching market intervals for {ticker}..."):
            stock = yf.Ticker(ticker)
            # Pulling exact live 5-minute session candlesticks
            df_ticks = stock.history(period="1d", interval="5m")

            if df_ticks.empty or len(df_ticks) < 2:
                st.error(
                    f"❌ Live session tick data unavailable for '{ticker}'. Please verify market hours or ticker symbol."
                )
                st.stop()

            current_price = float(df_ticks["Close"].iloc[-1])
            open_price = float(df_ticks["Open"].iloc[0])
            session_chg = ((current_price - open_price) / open_price) * 100

            # Match actual live momentum using direct candle comparison (latest close vs open of current session)
            is_upward_tick = current_price >= df_ticks["Open"].iloc[-1]
            main_direction = "UPWARD (BULLISH)" if is_upward_tick else "DOWNWARD (BEARISH)"
            card_color = "#00d09c" if is_upward_tick else "#eb5b3c"

        hc1, hc2 = st.columns([1.5, 1])
        with hc1:
            st.markdown(
                f"""<div class="card" style="padding: 22px 18px;">
                <h4 style="margin:0; color:white; font-size:18px;">{ticker} <span style="font-size:12px; color:#8c96a5;">(Live Session Feed)</span></h4>
                <p style="color:#8c96a5; margin:4px 0; font-size:12px;">Exchange: <b>{ex_label}</b> | Total Candles Synced: <b>{len(df_ticks)}</b></p>
                <h4 style="margin:8px 0 0 0; color:#00d09c; font-size:18px;">₹{current_price:.2f} <span style="font-size:12px; color:{'#00d09c' if session_chg >= 0 else '#eb5b3c'};">({session_chg:+.2f}%)</span></h4>
            </div>""",
                unsafe_allow_html=True,
            )
        with hc2:
            st.markdown(
                f"""<div class="card" style="background: {card_color}; color: #0f141e; text-align: center; padding: 22px 18px;">
                <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.5px;">LIVE TICK DIRECTION STATUS</div>
                <div style="font-size: 20px; font-weight: 900; margin: 6px 0;">{main_direction}</div>
                <div style="font-size: 12px; font-weight: 600;">Directly matched to live session price action.</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live LTP", f"₹{current_price:.2f}", f"{session_chg:+.2f}%")
        m2.metric("Session High", f"₹{df_ticks['High'].max():.2f}", "Peak Tick")
        m3.metric("Session Low", f"₹{df_ticks['Low'].min():.2f}", "Trough Tick")
        m4.metric("Session Volume", f"{int(df_ticks['Volume'].sum()):,}", "Total Traded")

        st.markdown("---")

        st.subheader("⚡ Live-Synced Interval Recording Matrix")
        
        # Build matrix mapping actual recent session highs/lows to intervals to ensure 100% data alignment
        sync_matrix = []
        recent_window = df_ticks.tail(6).reset_index()
        for idx, row in recent_window.iterrows():
            c_close = float(row["Close"])
            c_open = float(row["Open"])
            c_high = float(row["High"])
            c_low = float(row["Low"])
            
            candle_direction = "UPWARD 📈" if c_close >= c_open else "DOWNWARD 📉"
            sync_matrix.append({
                "Session Timestamp": str(row["Datetime"]) if "Datetime" in row else f"Interval {idx + 1}",
                "Synced Candle Open": f"₹{c_open:.2f}",
                "Synced Candle Close": f"₹{c_close:.2f}",
                "Session High/Low Range": f"H: ₹{c_high:.2f} / L: ₹{c_low:.2f}",
                "Live Direction Match": candle_direction
            })
            
        st.dataframe(pd.DataFrame(sync_matrix), use_container_width=True)

    except Exception as e:
        st.error(
            f"⚠️ Error syncing live data for {ticker}. Details: {e}"
        )
else:
    st.info(
        "Please select your exchange and enter a stock ticker above to initialize live tick-sync recording."
    )
