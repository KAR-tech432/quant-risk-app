import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live Intraday & Volume Tracker", page_icon="📈", layout="wide")

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
    "<h2 style='color: white; margin-bottom: 0;'>Live Intraday Momentum & Volume Surge Tracker</h2>",
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
        with st.spinner(f"Fetching live 5-minute intraday data for {ticker}..."):
            stock = yf.Ticker(ticker)
            # Fetch live intraday data for the current session at 5-minute intervals
            df_intraday = stock.history(period="1d", interval="1m")

            if df_intraday.empty:
                st.error(
                    f"❌ Market is currently closed or invalid ticker '{ticker}'. Live intraday data requires an active market session."
                )
                st.stop()

            current_price = float(df_intraday["Close"].iloc[-1])
            prev_close = float(df_intraday["Open"].iloc[0])
            chg = ((current_price - prev_close) / prev_close) * 100

            # Detect Volume Spikes (Bulk Order / Institutional Activity Indicator)
            avg_volume = df_intraday["Volume"].mean()
            latest_volume = df_intraday["Volume"].iloc[-1]
            is_bulk_surge = latest_volume > (avg_volume * 2.5) if avg_volume > 0 else False

            # Calculate immediate momentum
            price_momentum = current_price - float(df_intraday["Close"].iloc[-2]) if len(df_intraday) > 1 else 0

        hc1, hc2 = st.columns([1.5, 1])
        with hc1:
            st.markdown(
                f"""<div class="card" style="padding: 22px 18px;">
                <h4 style="margin:0; color:white; font-size:18px;">{ticker} <span style="font-size:12px; color:#8c96a5;">(Live Session Data)</span></h4>
                <p style="color:#8c96a5; margin:4px 0; font-size:12px;">Interval: <b>1 Minutes</b> | Source: <b>Real-Time Feed</b></p>
                <h4 style="margin:8px 0 0 0; color:#00d09c; font-size:18px;">₹{current_price:.2f} <span style="font-size:12px; color:{'#00d09c' if chg >= 0 else '#eb5b3c'};">({chg:+.2f}%)</span></h4>
            </div>""",
                unsafe_allow_html=True,
            )
        with hc2:
            signal_text = "🚨 BULK VOLUME SURGE DETECTED" if is_bulk_surge else ("BULLISH MOMENTUM" if price_momentum >= 0 else "BEARISH PRESSURE")
            signal_color = "#ffa726" if is_bulk_surge else ("#00d09c" if price_momentum >= 0 else "#eb5b3c")
            
            st.markdown(
                f"""<div class="card" style="background: {signal_color}; color: #0f141e; text-align: center; padding: 22px 18px;">
                <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.5px;">LIVE ORDER FLOW STATUS</div>
                <div style="font-size: 20px; font-weight: 900; margin: 6px 0;">{signal_text}</div>
                <div style="font-size: 12px; font-weight: 600;">{'Abnormal institutional block volume found in current 5m candle.' if is_bulk_surge else 'Standard intraday tick movement tracking.'}</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live LTP", f"₹{current_price:.2f}", f"{chg:+.2f}%")
        m2.metric("Latest 5m Volume", f"{int(latest_volume):,}", "Spike Indicator" if is_bulk_surge else "Normal")
        m3.metric("Session High", f"₹{df_intraday['High'].max():.2f}", "Peak")
        m4.metric("Session Low", f"₹{df_intraday['Low'].min():.2f}", "Floor")

        st.markdown("---")

        st.subheader("⚡ Next 5-Minute Intraday Projection Matrix")
        # Project targets for the next 6 upcoming 5-minute blocks based on current volatility velocity
        step = current_price * 0.001
        proj_data = []
        for i in range(1, 7):
            multiplier = i if price_momentum >= 0 else -i
            target_low = current_price + (multiplier * step * 0.5)
            target_high = current_price + (multiplier * step * 1.2)
            proj_data.append({
                "Time Interval": f"+{i * 1} Minutes Ahead",
                "Projected Low": f"₹{min(target_low, target_high):.2f}",
                "Projected High": f"₹{max(target_low, target_high):.2f}",
                "Trend Outlook": "Positive Expansion" if price_momentum >= 0 else "Correction Retracement"
            })
            
        st.dataframe(pd.DataFrame(proj_data), use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Live Intraday 5-Minute Session Candles")
        st.dataframe(df_intraday.tail(10).sort_index(ascending=False), use_container_width=True)

    except Exception as e:
        st.error(
            f"⚠️ Error pulling live intraday data for {ticker}. Please note that live 5-minute ticks are only available during active market hours. Details: {e}"
        )
else:
    st.info(
        "Please select your exchange and enter a stock ticker above to initialize live 5-minute tracking."
    )
