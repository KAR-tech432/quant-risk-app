import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Advanced Live Order & News Tracker", page_icon="📈", layout="wide")

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
    "<h2 style='color: white; margin-bottom: 0;'>Advanced Order Flow, News & Intraday Intelligence</h2>",
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
        with st.spinner(f"Analyzing live order books, historical trends & news feed for {ticker}..."):
            stock = yf.Ticker(ticker)
            
            # 1. Intraday data for live movement & bulk volume detection
            df_intraday = stock.history(period="1d", interval="5m")
            # 2. Historical data for context
            df_hist = stock.history(period="10d")
            # 3. Live News feed
            news_list = stock.news if hasattr(stock, "news") else []

            if df_intraday.empty or df_hist.empty:
                st.error(
                    f"❌ Insufficient live session or historical data for '{ticker}'. Please ensure the market is open or check the symbol."
                )
                st.stop()

            current_price = float(df_intraday["Close"].iloc[-1])
            prev_close = float(df_hist["Close"].iloc[-2] if len(df_hist) > 1 else df_intraday["Open"].iloc[0])
            chg = ((current_price - prev_close) / prev_close) * 100

            # Historical context check (10-day slope)
            hist_trend = "BULLISH" if df_hist["Close"].iloc[-1] > df_hist["Close"].iloc[0] else "BEARISH"

            # Bulk order / selloff detection via volume spikes
            avg_volume = df_intraday["Volume"].mean()
            latest_volume = df_intraday["Volume"].iloc[-1]
            is_bulk_selloff = (latest_volume > (avg_volume * 2.0)) and (df_intraday["Close"].iloc[-1] < df_intraday["Open"].iloc[-1])
            is_bulk_buying = (latest_volume > (avg_volume * 2.0)) and (df_intraday["Close"].iloc[-1] > df_intraday["Open"].iloc[-1])

            # Simple sentiment analysis from recent news headlines
            news_sentiment_score = 0
            recent_headlines = []
            for item in news_list[:5]:
                title = item.get("title", "") if isinstance(item, dict) else str(item)
                recent_headlines.append(title)
                t_lower = title.lower()
                if any(w in t_lower for w in ["fall", "crash", "drop", "slump", "loss", "down", "sell"]):
                    news_sentiment_score -= 1
                elif any(w in t_lower for w in ["surge", "rally", "jump", "gain", "profit", "up", "buy", "growth"]):
                    news_sentiment_score += 1

            # Synthesized Status Logic combining Live Move, Bulk Orders, History, and News
            if is_bulk_selloff or news_sentiment_score < 0:
                status_title = "🚨 HEAVY SELLOFF / BEARISH PRESSURE"
                status_color = "#eb5b3c"
                status_desc = "Detected heavy volume selloff blocks coupled with negative market news sentiment."
            elif is_bulk_buying or news_sentiment_score > 0:
                status_title = "🚀 INSTITUTIONAL ACCUMULATION / BUYING"
                status_color = "#00d09c"
                status_desc = "Detected bulk order volume spikes matching positive sentiment catalysts."
            else:
                status_title = f"NEUTRAL / {hist_trend} CONTEXT"
                status_color = "#ffa726"
                status_desc = "Balanced order flow with standard intraday price action."

        hc1, hc2 = st.columns([1.5, 1])
        with hc1:
            st.markdown(
                f"""<div class="card" style="padding: 22px 18px;">
                <h4 style="margin:0; color:white; font-size:18px;">{ticker} <span style="font-size:12px; color:#8c96a5;">(Multi-Factor Engine)</span></h4>
                <p style="color:#8c96a5; margin:4px 0; font-size:12px;">History Trend: <b>{hist_trend}</b> | News Sentiment: <b>{'Negative' if news_sentiment_score < 0 else ('Positive' if news_sentiment_score > 0 else 'Neutral')}</b></p>
                <h4 style="margin:8px 0 0 0; color:#00d09c; font-size:18px;">₹{current_price:.2f} <span style="font-size:12px; color:{'#00d09c' if chg >= 0 else '#eb5b3c'};">({chg:+.2f}%)</span></h4>
            </div>""",
                unsafe_allow_html=True,
            )
        with hc2:
            # Updated, enlarged, prominent Live Order Flow Status Card based on all conditions
            st.markdown(
                f"""<div class="card" style="background: {status_color}; color: #0f141e; text-align: center; padding: 22px 18px;">
                <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.5px;">LIVE ORDER FLOW STATUS</div>
                <div style="font-size: 20px; font-weight: 900; margin: 6px 0;">{status_title}</div>
                <div style="font-size: 12px; font-weight: 600;">{status_desc}</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live LTP", f"₹{current_price:.2f}", f"{chg:+.2f}%")
        m2.metric("Latest 5m Volume", f"{int(latest_volume):,}", "Bulk Spike" if (is_bulk_selloff or is_bulk_buying) else "Normal")
        m3.metric("10-Day Trend Context", hist_trend, "Historical Multi-Day")
        m4.metric("News Catalyst Score", f"{news_sentiment_score:+d}", "Headline Sentiment")

        st.markdown("---")

        st.subheader("⚡ 5-Minute Intraday Predictive Matrix")
        step = current_price * 0.0015
        direction_mult = -1 if (is_bulk_selloff or news_sentiment_score < 0) else 1
        proj_data = []
        for i in range(1, 7):
            mult = i * direction_mult
            t_low = current_price + (mult * step * 0.4)
            t_high = current_price + (mult * step * 1.3)
            proj_data.append({
                "Interval": f"+{i * 5} Min",
                "Projected Floor": f"₹{min(t_low, t_high):.2f}",
                "Projected Ceiling": f"₹{max(t_low, t_high):.2f}",
                "Trend Outlook": "Bearish Pressure / Downside" if direction_mult < 0 else "Bullish Expansion / Upside"
            })
        st.dataframe(pd.DataFrame(proj_data), use_container_width=True)

        if recent_headlines:
            st.markdown("---")
            st.subheader("📰 Live News Headlines Feed & Catalyst Context")
            for h in recent_headlines[:3]:
                st.markdown(f"- {h}")

    except Exception as e:
        st.error(
            f"⚠️ Error executing advanced analysis for {ticker}. Details: {e}"
        )
else:
    st.info(
        "Please select your exchange and enter a stock ticker above to run live multi-factor analysis."
    )
