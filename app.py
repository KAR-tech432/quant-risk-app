import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

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
    "<h2 style='color: white; margin-bottom: 0;'>Real-Time Stock Analysis & Prediction Portal</h2>",
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
    clean_inp = (
        inp.replace(".NS", "")
        .replace(".BO", "")
        .replace(".BS", "")
        .strip()
    )
    ticker = f"{clean_inp}{ex}"

if ticker:
    try:
        with st.spinner(f"Fetching real market data & computing pivot levels for {ticker}..."):
            stock = yf.Ticker(ticker)
            df_hist = stock.history(period="15d")

            if df_hist.empty:
                st.error(
                    f"❌ Could not retrieve data for '{ticker}'. Please verify the ticker symbol."
                )
                st.stop()

            current_price = float(df_hist["Close"].iloc[-1])
            prev_close = float(
                df_hist["Close"].iloc[-2]
                if len(df_hist) > 1
                else current_price
            )
            chg = ((current_price - prev_close) / prev_close) * 100

            # Calculate RSI (14)
            delta = df_hist["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (
                float(100 - (100 / (1 + rs)).iloc[-1])
                if not rs.empty and not pd.isna(rs.iloc[-1])
                else 50.0
            )

            # Calculate Classic Pivot Points & Support Levels from previous day's data
            high_p = float(df_hist["High"].iloc[-2])
            low_p = float(df_hist["Low"].iloc[-2])
            close_p = float(df_hist["Close"].iloc[-2])
            
            pivot = (high_p + low_p + close_p) / 3
            s1 = (2 * pivot) - high_p
            s2 = pivot - (high_p - low_p)
            s3 = low_p - 2 * (high_p - pivot)

        hc1, hc2 = st.columns([1.5, 1])
        with hc1:
            st.markdown(
                f"""<div class="card" style="padding: 22px 18px;">
                <h4 style="margin:0; color:white; font-size:18px;">{ticker} <span style="font-size:12px; color:#8c96a5;">(Live Market Data)</span></h4>
                <p style="color:#8c96a5; margin:4px 0; font-size:12px;">Exchange: <b>{ex_label}</b> | Source: <b>Yahoo Finance</b></p>
                <h4 style="margin:8px 0 0 0; color:#00d09c; font-size:18px;">₹{current_price:.2f} <span style="font-size:12px; color:{'#00d09c' if chg >= 0 else '#eb5b3c'};">({chg:+.2f}%)</span></h4>
            </div>""",
                unsafe_allow_html=True,
            )
        with hc2:
            # Custom signal logic checking support boundary breaches
            is_breakdown = current_price <= s1
            action = "BEARISH / DOWN-TEST" if is_breakdown else ("BUY" if rsi < 40 else ("ACCUMULATE" if 40 <= rsi <= 60 else "SELL"))
            color = "#eb5b3c" if is_breakdown else ("#00d09c" if action == "BUY" else ("#ffa726" if action == "ACCUMULATE" else "#eb5b3c"))
            
            st.markdown(
                f"""<div class="card" style="background: {color}; color: #0f141e; text-align: center; padding: 22px 18px;">
                <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.5px;">RISK & SIGNAL ALERT</div>
                <div style="font-size: 22px; font-weight: 900; margin: 6px 0;">{action}</div>
                <div style="font-size: 12px; font-weight: 600;">{'Price breaching critical support floors.' if is_breakdown else 'Standard technical momentum tracking.'}</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last Traded Price", f"₹{current_price:.2f}", f"{chg:+.2f}%")
        m2.metric("Support 1 (S1)", f"₹{s1:.2f}", "Immediate Floor")
        m3.metric("Support 2 (S2)", f"₹{s2:.2f}", "Deep Drop Target")
        m4.metric("RSI (14)", f"{rsi:.1f}", "Momentum")

        st.markdown("---")

        st.subheader("⚠️ Technical Support & Breakdown Target Matrix")
        support_data = [
            {"Level Type": "Pivot Point (PP)", "Price Boundary": f"₹{pivot:.2f}", "Status": "Neutral Centerline"},
            {"Level Type": "Support 1 (S1)", "Price Boundary": f"₹{s1:.2f}", "Status": "Critical Test Zone / Minor Breakdown"},
            {"Level Type": "Support 2 (S2)", "Price Boundary": f"₹{s2:.2f}", "Status": "Major Correction / Target Floor Zone"},
            {"Level Type": "Support 3 (S3)", "Price Boundary": f"₹{s3:.2f}", "Status": "Extreme Crash Lower Band"}
        ]
        st.dataframe(pd.DataFrame(support_data), use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Recent Historical Data (15 Days)")
        st.dataframe(df_hist.sort_index(ascending=False), use_container_width=True)

    except Exception as e:
        st.error(
            f"⚠️ Error processing pivot calculations for {ticker}. Details: {e}"
        )
else:
    st.info(
        "Please select your exchange and enter a stock ticker above to view deep support floors and breakdown matrices."
    )
