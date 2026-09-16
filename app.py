import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Live Market-Aligned 10-Min Projections", page_icon="📈", layout="wide")

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
    "<h2 style='color: white; margin-bottom: 0;'>Live Market-Aligned 10-Minute Analytics & Next Candle Predictor</h2>",
    unsafe_allow_html=True,
)
st.markdown("---")

c1, c2 = st.columns([1, 2])
with c1:
    ex_label = st.selectbox("Market Exchange:", ["NSE", "BSE"])
    ex = ".NS" if ex_label == "NSE" else ".BO"
with c2:
    inp = st.text_input(
        "Enter Stock Ticker (e.g., IDEA, RELIANCE, TCS, INFY):",
        value="",
        placeholder="Type symbol...",
    ).upper()

ticker = None
if inp:
    clean_inp = inp.replace(".NS", "").replace(".BO", "").replace(".BS", "").strip()
    ticker = f"{clean_inp}{ex}"

if ticker:
    try:
        with st.spinner(f"Running full market data analysis and projecting next candle for {ticker}..."):
            stock = yf.Ticker(ticker)
            df_raw = stock.history(period="1d", interval="5m")

            if df_raw.empty or len(df_raw) < 2:
                st.error(
                    f"❌ Live session data is currently unavailable for '{ticker}'. Ensure the market session is active."
                )
                st.stop()

            # Clean timezone for flawless resampling
            if df_raw.index.tz is not None:
                df_raw.index = df_raw.index.tz_localize(None)

            # Filter regular exchange hours (09:15 to 15:30)
            df_raw = df_raw.between_time('09:15', '15:30')

            if df_raw.empty:
                st.error("❌ No data available within regular market hours (09:15 - 15:30).")
                st.stop()

            # Resample strictly into 10-minute market blocks starting from 09:15:00
            df_candles = df_raw.resample('10min', origin='09:15:00', closed='left', label='left').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            if df_candles.empty:
                st.error("❌ Not enough data points to form market-aligned 10-minute candles.")
                st.stop()

            current_price = float(df_candles["Close"].iloc[-1])
            session_open = float(df_candles["Open"].iloc[0])
            session_chg = ((current_price - session_open) / session_open) * 100

            # --- EXPERT MARKET ANALYSIS & NEXT CANDLE PROJECTION ENGINE ---
            # Evaluating recent momentum, volume expansion, and close-open delta of the last 3 bars
            recent_bars = df_candles.tail(3)
            momentum_score = 0
            for _, r in recent_bars.iterrows():
                if r['Close'] >= r['Open']:
                    momentum_score += 1
                else:
                    momentum_score -= 1

            avg_range = (df_candles['High'] - df_candles['Low']).mean()
            last_vol = df_candles['Volume'].iloc[-1]
            avg_vol = df_candles['Volume'].mean()
            vol_surge = last_vol > avg_vol

            # Projection Logic
            is_bullish_projection = (momentum_score > 0) or (vol_surge and df_candles['Close'].iloc[-1] >= df_candles['Open'].iloc[-1])
            projection_label = "UPWARD (BULLISH PROJECTION 📈)" if is_bullish_projection else "DOWNWARD (BEARISH PROJECTION 📉)"
            projection_color = "#00d09c" if is_bullish_projection else "#eb5b3c"
            
            # Projected target price range for next 10m bar
            projected_delta = avg_range * 0.6 if is_bullish_projection else -(avg_range * 0.6)
            projected_target = current_price + projected_delta
            confidence_pct = min(88, max(55, 60 + abs(momentum_score) * 10 + (10 if vol_surge else 0)))

        hc1, hc2 = st.columns([1.5, 1])
        with hc1:
            st.markdown(
                f"""<div class="card" style="padding: 22px 18px;">
                <h4 style="margin:0; color:white; font-size:18px;">{ticker} <span style="font-size:12px; color:#8c96a5;">(Market Expertise Engine)</span></h4>
                <p style="color:#8c96a5; margin:4px 0; font-size:12px;">Exchange: <b>{ex_label}</b> | Total 10m Bars: <b>{len(df_candles)}</b></p>
                <h4 style="margin:8px 0 0 0; color:#00d09c; font-size:18px;">₹{current_price:.2f} <span style="font-size:12px; color:{'#00d09c' if session_chg >= 0 else '#eb5b3c'};">({session_chg:+.2f}%)</span></h4>
            </div>""",
                unsafe_allow_html=True,
            )
        with hc2:
            st.markdown(
                f"""<div class="card" style="background: {projection_color}; color: #0f141e; text-align: center; padding: 22px 18px;">
                <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.5px;">NEXT 10-MIN CANDLE FORECAST</div>
                <div style="font-size: 18px; font-weight: 900; margin: 6px 0;">{projection_label}</div>
                <div style="font-size: 12px; font-weight: 600;">Est. Target: ₹{projected_target:.2f} | Confidence: {confidence_pct}%</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live LTP", f"₹{current_price:.2f}", f"{session_chg:+.2f}%")
        m2.metric("Volume Surge Status", "High Activity" if vol_surge else "Normal", "vs Session Avg")
        m3.metric("Avg 10m Range", f"₹{avg_range:.2f}", "Volatility Index")
        m4.metric("Projection Confidence", f"{confidence_pct}%", "Statistical Weight")

        st.markdown("---")

        # --- INTERACTIVE LIVE 10-MINUTE CANDLESTICK CHART ---
        st.subheader("📊 Interactive Live Market-Aligned 10-Minute Candlestick Graph")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_candles.index,
            open=df_candles['Open'],
            high=df_candles['High'],
            low=df_candles['Low'],
            close=df_candles['Close'],
            increasing_line_color='#00d09c',
            decreasing_line_color='#eb5b3c',
            name='10-Min Market Candles'
        )])
        
        fig.update_layout(
            paper_bgcolor='#0f141e',
            plot_bgcolor='#1c212b',
            font=dict(color='#f0f4f8'),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title='Market Session Timestamps (10 Min Interval: 09:15, 09:25, 09:35...)', gridcolor='#28303d', rangeslider=dict(visible=False)),
            yaxis=dict(title='Price (₹)', gridcolor='#28303d'),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("⚡ Market-Synced 10-Minute Candle Interval Records")
        
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
        st.error(
            f"⚠️ Error analyzing market data and projecting next candle for {ticker}. Details: {e}"
        )
else:
    st.info(
        "Please select your exchange and enter a stock ticker above to load the live analysis and next candle projection engine."
    )
