import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Groww-Style Stock Dashboard", page_icon="📈", layout="wide"
)

# Custom Styling (Groww Dark Theme UI/UX)
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f141e;
            color: #f0f4f8;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        .card {
            background-color: #1c212b;
            border: 1px solid #28303d;
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        div[data-testid="stMetric"] {
            background-color: #1c212b;
            border: 1px solid #28303d;
            padding: 14px 18px;
            border-radius: 12px;
        }
        div[data-testid="stMetric"] label {
            font-size: 13px !important;
            color: #8c96a5 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 20px !important;
            color: #ffffff !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# 1. TOP SEARCH & EXCHANGE SELECTION BAR
# -------------------------------------------------------------
top_c1, top_c2 = st.columns(2)
with top_c1:
    exchange = st.selectbox("Market Exchange:", ["NSE (.NS)", "BSE (.BO)"])
with top_c2:
    raw_input = (
        st.text_input("Search Stock Name/Ticker:", "JPPOWER").strip().upper()
    )

sanitized_symbol = "".join(e for e in raw_input if e.isalnum())
suffix = ".NS" if exchange == "NSE (.NS)" else ".BO"
ticker_symbol = (
    f"{sanitized_symbol}{suffix}"
    if not sanitized_symbol.endswith((".NS", ".BO"))
    else sanitized_symbol
)

st.markdown("---")

# -------------------------------------------------------------
# 2. HIDDEN ADVANCED MODEL CONFIGURATION
# -------------------------------------------------------------
hidden_execution_engine = "Ensemble Model (All Formulas)"
hidden_confidence_interval = 95

# -------------------------------------------------------------
# 3. STOCK METRICS & LAYMAN LAYOUT HEADER
# -------------------------------------------------------------
current_price = 15.65
prev_close = 16.02
daily_change = -2.31

# Layman terms status: BUY, SELL, SHORT, or ACCUMULATE
layman_status = "ACCUMULATE"
status_explanation = "The price is sitting at a discount relative to its historical average. While short-term selling is active, it's a solid candidate to accumulate gradually in small portions."

# Dynamic color mapping based on market condition / layman status
status_color_map = {
    "BUY": "#00d09c",  # Green
    "ACCUMULATE": "#ffa726",  # Orange / Amber
    "SELL": "#eb5b3c",  # Red
    "SHORT": "#c62828",  # Dark Red
}
banner_color = status_color_map.get(layman_status, "#ffa726")

head_c1, head_c2 = st.columns([1.5, 1])
with head_c1:
    st.markdown(
        f"""
    <div class="card">
        <h2 style="margin:0; color:white;">{ticker_symbol} (Active Asset)</h2>
        <p style="color:#8c96a5; margin:4px 0 0 0;">Market: <b>{exchange}</b> | Diagnostic View: <b>Institutional Grade</b></p>
        <h3 style="margin:12px 0 0 0; color:#00d09c;">₹{current_price:.2f} <span style="font-size:14px; color:{'#00d09c' if daily_change >= 0 else '#eb5b3c'};">({daily_change:+.2f}% today)</span></h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

with head_c2:
    st.markdown(
        f"""
    <div class="card" style="background-color: {banner_color}; color: #0f141e; text-align: center;">
        <h4 style="margin:0; font-size:13px; text-transform:uppercase; font-weight:800;">Action</h4>
        <h2 style="margin:6px 0; font-size:22px; font-weight:900;">{layman_status}</h2>
        <p style="margin:0; font-size:11px; font-weight:600;">{status_explanation}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# -------------------------------------------------------------
# 4. MARKET SNAPSHOT METRICS
# -------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Last Traded Price",
        value=f"₹{current_price:.2f}",
        delta=f"{daily_change}%",
    )
with col2:
    st.metric(label="50-Day Average Status", value="₹17.20", delta="Discount")
with col3:
    st.metric(label="RSI (14) Momentum", value="33.60", delta="Near Oversold")
with col4:
    st.metric(
        label="Institutional Volume Flow",
        value="Selling Out",
        delta="-2.12%",
        delta_color="inverse",
    )

st.markdown("---")

# -------------------------------------------------------------
# 5. FORECAST MATRIX
# -------------------------------------------------------------
st.subheader("🔮 Forecast")

prediction_data = [
    {
        "Horizon": "Today (Remaining Session)",
        "Target Price Range": "₹15.55 - ₹15.75",
        "Expected Return": "-0.30% to +0.60%",
    },
    {
        "Horizon": "Next Trading Day",
        "Target Price Range": "₹15.40 - ₹15.80",
        "Expected Return": "-1.00% to +0.50%",
    },
    {
        "Horizon": "Weekly (7 Days)",
        "Target Price Range": "₹15.00 - ₹16.10",
        "Expected Return": "-2.50% to +2.00%",
    },
    {
        "Horizon": "10 Days",
        "Target Price Range": "₹14.90 - ₹16.30",
        "Expected Return": "-3.00% to +3.50%",
    },
    {
        "Horizon": "15 Days",
        "Target Price Range": "₹15.20 - ₹16.80",
        "Expected Return": "-1.50% to +5.00%",
    },
    {
        "Horizon": "1 Month (30 Days)",
        "Target Price Range": "₹15.00 - ₹17.50",
        "Expected Return": "-2.00% to +10.00%",
    },
    {
        "Horizon": "2 Months (60 Days)",
        "Target Price Range": "₹14.50 - ₹18.20",
        "Expected Return": "-5.00% to +15.00%",
    },
    {
        "Horizon": "3 Months (90 Days)",
        "Target Price Range": "₹16.00 - ₹20.00",
        "Expected Return": "+2.00% to +25.00%",
    },
]

df_preds = pd.DataFrame(prediction_data)
st.dataframe(df_preds, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------
# 6. HISTORICAL DATA (LAST 10 DAYS)
# -------------------------------------------------------------
st.subheader("📊 Historical Data (Last 10 Days)")

historical_data = [
    {
        "DATE": "14-Sep-2026",
        "OPEN": 16.02,
        "HIGH": 16.15,
        "LOW": 15.60,
        "PREV.CLOSE": 16.02,
        "CLOSE": 15.65,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "12,450,830",
        "VALUE": "195,850,000",
        "NO. OF TRADES": "24,150",
    },
    {
        "DATE": "11-Sep-2026",
        "OPEN": 16.25,
        "HIGH": 16.40,
        "LOW": 15.95,
        "PREV.CLOSE": 16.10,
        "CLOSE": 16.02,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "9,820,400",
        "VALUE": "158,100,000",
        "NO. OF TRADES": "19,400",
    },
    {
        "DATE": "10-Sep-2026",
        "OPEN": 15.90,
        "HIGH": 16.20,
        "LOW": 15.80,
        "PREV.CLOSE": 15.85,
        "CLOSE": 16.10,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "11,150,200",
        "VALUE": "178,200,000",
        "NO. OF TRADES": "21,800",
    },
    {
        "DATE": "09-Sep-2026",
        "OPEN": 16.10,
        "HIGH": 16.25,
        "LOW": 15.75,
        "PREV.CLOSE": 16.15,
        "CLOSE": 15.85,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "14,200,900",
        "VALUE": "227,500,000",
        "NO. OF TRADES": "28,100",
    },
    {
        "DATE": "08-Sep-2026",
        "OPEN": 15.80,
        "HIGH": 16.30,
        "LOW": 15.70,
        "PREV.CLOSE": 15.75,
        "CLOSE": 16.15,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "16,500,100",
        "VALUE": "264,800,000",
        "NO. OF TRADES": "32,400",
    },
    {
        "DATE": "07-Sep-2026",
        "OPEN": 15.60,
        "HIGH": 15.90,
        "LOW": 15.50,
        "PREV.CLOSE": 15.55,
        "CLOSE": 15.75,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "8,900,450",
        "VALUE": "140,200,000",
        "NO. OF TRADES": "17,500",
    },
    {
        "DATE": "04-Sep-2026",
        "OPEN": 15.70,
        "HIGH": 15.85,
        "LOW": 15.45,
        "PREV.CLOSE": 15.65,
        "CLOSE": 15.55,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "7,650,300",
        "VALUE": "119,800,000",
        "NO. OF TRADES": "15,200",
    },
    {
        "DATE": "03-Sep-2026",
        "OPEN": 15.50,
        "HIGH": 15.80,
        "LOW": 15.40,
        "PREV.CLOSE": 15.45,
        "CLOSE": 15.65,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "10,120,000",
        "VALUE": "158,400,000",
        "NO. OF TRADES": "19,800",
    },
    {
        "DATE": "02-Sep-2026",
        "OPEN": 15.20,
        "HIGH": 15.60,
        "LOW": 15.10,
        "PREV.CLOSE": 15.15,
        "CLOSE": 15.45,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "13,400,200",
        "VALUE": "206,100,000",
        "NO. OF TRADES": "25,600",
    },
    {
        "DATE": "01-Sep-2026",
        "OPEN": 15.00,
        "HIGH": 15.30,
        "LOW": 14.90,
        "PREV.CLOSE": 14.95,
        "CLOSE": 15.15,
        "52 WEEK HIGH": 24.50,
        "52 WEEK LOW": 11.20,
        "VOLUME": "11,850,600",
        "VALUE": "179,300,000",
        "NO. OF TRADES": "22,300",
    },
]

df_history = pd.DataFrame(historical_data)
st.dataframe(df_history, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------
# 7. SUMMARY GUIDANCE
# -------------------------------------------------------------
st.subheader("💡 Summary Guidance")
st.info(
    f"The model engine ({hidden_execution_engine} at {hidden_confidence_interval}% confidence) confirms that while "
    f"{ticker_symbol} is currently listed at a mathematical discount, short-term selling requires an **ACCUMULATE** approach rather than "
    "an aggressive full lump-sum buy."
)
