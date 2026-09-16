import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone, timedelta

# ============================================================
# RULE-BASED MANUAL QUANT TERMINAL
# No AI / No Machine Learning / No Predictive Algorithm
#
# All levels are derived from transparent technical formulas.
# ============================================================

st.set_page_config(
    page_title="Rule-Based Quant Terminal",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.stApp {
    background:#f1f5f9;
    color:#0f172a;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}

.card {
    background:#ffffff;
    border:1px solid #cbd5e1;
    border-radius:10px;
    padding:15px;
    margin-bottom:12px;
    box-shadow:0 2px 5px rgba(0,0,0,.05);
}

.title {
    font-size:11px;
    font-weight:900;
    text-transform:uppercase;
    letter-spacing:.8px;
    color:#475569;
}

.metric {
    font-size:21px;
    font-weight:900;
}

.small {
    font-size:12px;
    color:#475569;
}

.bull {
    color:#047857;
}

.bear {
    color:#b91c1c;
}

.neutral {
    color:#92400e;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONTROL BAR
# ============================================================

c1, c2, c3 = st.columns([1, 2.5, 1.2])

exchange = c1.selectbox("Exchange", ["NSE", "BSE"])

suffix = ".NS" if exchange == "NSE" else ".BO"

symbol = c2.text_input(
    "Asset Ticker",
    value="TEJASNET"
).upper().strip()

run = c3.button(
    "Run Analysis",
    use_container_width=True
)

ticker = f"{symbol}{suffix}" if symbol else None


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=60)
def load_data(ticker):

    stock = yf.Ticker(ticker)

    daily = stock.history(
        period="2y",
        interval="1d",
        auto_adjust=False
    )

    intraday = stock.history(
        period="1d",
        interval="5m",
        auto_adjust=False
    )

    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)

    if not intraday.empty and intraday.index.tz is not None:
        intraday.index = intraday.index.tz_localize(None)

    return daily, intraday


# ============================================================
# TECHNICAL FUNCTIONS
# ============================================================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def true_range(df):

    previous_close = df["Close"].shift(1)

    a = df["High"] - df["Low"]
    b = (df["High"] - previous_close).abs()
    c = (df["Low"] - previous_close).abs()

    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr_simple(df, period=14):

    tr = true_range(df)

    return tr.rolling(period).mean()


def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def macd(series):

    ema12 = ema(series, 12)
    ema26 = ema(series, 26)

    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)

    histogram = macd_line - signal

    return macd_line, signal, histogram


def classic_pivots(prev_high, prev_low, prev_close):

    pp = (prev_high + prev_low + prev_close) / 3

    r1 = (2 * pp) - prev_low
    s1 = (2 * pp) - prev_high

    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)

    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)

    return pp, r1, r2, r3, s1, s2, s3


def fibonacci_levels(high, low):

    difference = high - low

    return {
        "23.6%": high - difference * 0.236,
        "38.2%": high - difference * 0.382,
        "50.0%": high - difference * 0.500,
        "61.8%": high - difference * 0.618,
        "78.6%": high - difference * 0.786,
    }


# ============================================================
# MAIN ENGINE
# ============================================================

if ticker:

    df, intraday = load_data(ticker)

    if df.empty or len(df) < 200:

        st.error(
            "Insufficient historical data. At least 200 daily candles are required."
        )

        st.stop()

    # --------------------------------------------------------
    # BASIC PRICE DATA
    # --------------------------------------------------------

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    price = float(close.iloc[-1])

    previous_close = float(close.iloc[-2])

    day_change = (
        (price - previous_close)
        / previous_close
    ) * 100

    current_high = float(high.iloc[-1])
    current_low = float(low.iloc[-1])

    previous_high = float(high.iloc[-2])
    previous_low = float(low.iloc[-2])

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    sma5 = float(close.rolling(5).mean().iloc[-1])
    sma10 = float(close.rolling(10).mean().iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma100 = float(close.rolling(100).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])

    ema20 = float(ema(close, 20).iloc[-1])
    ema50 = float(ema(close, 50).iloc[-1])

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    atr14 = float(atr_simple(df, 14).iloc[-1])

    atr_percent = (
        atr14 / price
    ) * 100

    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

    std20 = float(
        close.rolling(20).std().iloc[-1]
    )

    bb_middle = sma20
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi14 = float(
        rsi(close, 14).iloc[-1]
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_line, macd_signal, macd_hist = macd(close)

    macd_now = float(macd_line.iloc[-1])
    macd_signal_now = float(macd_signal.iloc[-1])
    macd_hist_now = float(macd_hist.iloc[-1])

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    roc5 = (
        (price / float(close.iloc[-6])) - 1
    ) * 100

    roc10 = (
        (price / float(close.iloc[-11])) - 1
    ) * 100

    roc20 = (
        (price / float(close.iloc[-21])) - 1
    ) * 100

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    avg_volume20 = float(
        volume.rolling(20).mean().iloc[-1]
    )

    volume_ratio = (
        float(volume.iloc[-1])
        / avg_volume20
        if avg_volume20 > 0 else 1
    )

    # --------------------------------------------------------
    # RECENT RANGE
    # --------------------------------------------------------

    high10 = float(high.tail(10).max())
    low10 = float(low.tail(10).min())

    high20 = float(high.tail(20).max())
    low20 = float(low.tail(20).min())

    high52 = float(high.tail(252).max())
    low52 = float(low.tail(252).min())

    range10 = high10 - low10
    range20 = high20 - low20

    # --------------------------------------------------------
    # PIVOTS
    # --------------------------------------------------------

    (
        pivot,
        r1,
        r2,
        r3,
        s1,
        s2,
        s3
    ) = classic_pivots(
        previous_high,
        previous_low,
        previous_close
    )

    # --------------------------------------------------------
    # FIBONACCI - 20 DAY SWING
    # --------------------------------------------------------

    fib = fibonacci_levels(
        high20,
        low20
    )

    # ========================================================
    # MANUAL MARKET SCORE
    # ========================================================

    bullish_points = []
    bearish_points = []

    # Price vs moving averages
    if price > sma20:
        bullish_points.append("Price above SMA20")
    else:
        bearish_points.append("Price below SMA20")

    if price > sma50:
        bullish_points.append("Price above SMA50")
    else:
        bearish_points.append("Price below SMA50")

    if price > sma200:
        bullish_points.append("Price above SMA200")
    else:
        bearish_points.append("Price below SMA200")

    # Trend alignment
    if sma20 > sma50:
        bullish_points.append("SMA20 above SMA50")
    else:
        bearish_points.append("SMA20 below SMA50")

    if sma50 > sma200:
        bullish_points.append("SMA50 above SMA200")
    else:
        bearish_points.append("SMA50 below SMA200")

    # EMA
    if price > ema20:
        bullish_points.append("Price above EMA20")
    else:
        bearish_points.append("Price below EMA20")

    # Pivot
    if price > pivot:
        bullish_points.append("Price above pivot")
    else:
        bearish_points.append("Price below pivot")

    # RSI
    if 50 <= rsi14 <= 70:
        bullish_points.append("RSI positive")
    elif rsi14 < 35:
        bullish_points.append("RSI oversold-zone")
    elif rsi14 > 70:
        bearish_points.append("RSI overbought")

    # MACD
    if macd_line.iloc[-1] > macd_signal.iloc[-1]:
        bullish_points.append("MACD above signal")
    else:
        bearish_points.append("MACD below signal")

    # Momentum
    if roc10 > 0:
        bullish_points.append("10-day momentum positive")
    else:
        bearish_points.append("10-day momentum negative")

    # Volume
    if volume_ratio >= 1.20 and day_change > 0:
        bullish_points.append("High volume + positive price")
    elif volume_ratio >= 1.20 and day_change < 0:
        bearish_points.append("High volume + negative price")

    bull_count = len(bullish_points)
    bear_count = len(bearish_points)

    total_points = bull_count + bear_count

    bullish_pct = (
        bull_count / total_points * 100
        if total_points else 50
    )

    bearish_pct = (
        bear_count / total_points * 100
        if total_points else 50
    )

    # ========================================================
    # MARKET STATE
    # ========================================================

    if (
        bull_count >= 8
        and price > sma50
        and price > sma200
    ):

        market_state = "BULLISH TREND"
        state_color = "#047857"

    elif (
        bear_count >= 8
        and price < sma50
        and price < sma200
    ):

        market_state = "BEARISH TREND"
        state_color = "#b91c1c"

    else:

        market_state = "MIXED / RANGE"
        state_color = "#92400e"

    # ========================================================
    # MANUAL PRICE SCENARIOS
    # ========================================================

    # --------------------------------------------------------
    # BASE RANGE
    # Uses:
    # Pivot + ATR
    # --------------------------------------------------------

    base_low = max(
        s1,
        price - atr14
    )

    base_high = min(
        r1,
        price + atr14
    )

    # --------------------------------------------------------
    # BULLISH TARGETS
    # --------------------------------------------------------

    bullish_target1 = r1
    bullish_target2 = r2

    # Do not allow targets to be absurdly far from price.
    bullish_target2 = min(
        bullish_target2,
        price + atr14 * 2.0
    )

    # --------------------------------------------------------
    # BEARISH TARGETS
    # --------------------------------------------------------

    bearish_target1 = s1
    bearish_target2 = s2

    bearish_target2 = max(
        bearish_target2,
        price - atr14 * 2.0
    )

    # --------------------------------------------------------
    # EXTREME TECHNICAL LEVELS
    # --------------------------------------------------------

    extreme_support = max(
        low20,
        price - 2 * atr14
    )

    extreme_resistance = min(
        high20,
        price + 2 * atr14
    )

    # ========================================================
    # BREAKOUT / BREAKDOWN CONDITIONS
    # ========================================================

    bullish_breakout = (
        price > r1
        and volume_ratio >= 1.20
    )

    bearish_breakdown = (
        price < s1
        and volume_ratio >= 1.20
    )

    # ========================================================
    # MANUAL DECISION FRAMEWORK
    # ========================================================

    if bullish_breakout:

        decision = "BULLISH BREAKOUT CONFIRMATION"

        decision_color = "#047857"

        explanation = (
            "Price is above the first resistance and volume "
            "is materially above its 20-day average. "
            "A breakout is therefore technically confirmed."
        )

    elif bearish_breakdown:

        decision = "BEARISH BREAKDOWN CONFIRMATION"

        decision_color = "#b91c1c"

        explanation = (
            "Price is below the first support with elevated "
            "volume. The downside move has technical confirmation."
        )

    elif market_state == "BULLISH TREND":

        decision = "BULLISH TREND — WAIT FOR ENTRY"

        decision_color = "#047857"

        explanation = (
            "The broader trend is bullish, but the program "
            "requires price confirmation before treating a move "
            "as a breakout."
        )

    elif market_state == "BEARISH TREND":

        decision = "BEARISH TREND — WAIT FOR REVERSAL"

        decision_color = "#b91c1c"

        explanation = (
            "The broader trend is bearish. A reversal should "
            "be confirmed before treating the stock as a "
            "new bullish setup."
        )

    else:

        decision = "MIXED / RANGE — WAIT"

        decision_color = "#92400e"

        explanation = (
            "The indicators are not sufficiently aligned. "
            "Price should be observed around support and resistance "
            "rather than forcing a directional conclusion."
        )

    # ========================================================
    # MARKET STATUS
    # ========================================================

    ist_now = datetime.now(
        timezone(timedelta(hours=5, minutes=30))
    )

    current_time = ist_now.time()

    market_open = time(9, 15)
    market_close = time(15, 30)

    if (
        ist_now.weekday() < 5
        and market_open <= current_time <= market_close
    ):

        market_status = "🟢 MARKET OPEN"

    elif (
        ist_now.weekday() < 5
        and current_time < market_open
    ):

        market_status = "🟡 PRE-MARKET"

    else:

        market_status = "🔴 MARKET CLOSED"

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        f"""
        <h2 style="margin-bottom:0">
        📊 {ticker}
        <span style="color:{decision_color}">
        ₹{price:.2f} ({day_change:+.2f}%)
        </span>
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"{market_status} | "
        f"IST {ist_now.strftime('%H:%M:%S')}"
    )

    # ========================================================
    # DECISION CARD
    # ========================================================

    st.markdown(
        f"""
        <div class="card"
             style="border:2px solid {decision_color};">

            <div class="title">
                RULE-BASED MARKET ASSESSMENT
            </div>

            <div class="metric"
                 style="color:{decision_color}">
                {decision}
            </div>

            <div class="small"
                 style="margin-top:8px">
                {explanation}
            </div>

            <hr>

            <b>Bullish conditions:</b> {bull_count}<br>
            <b>Bearish conditions:</b> {bear_count}<br>
            <b>Technical state:</b> {market_state}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # PRICE SCENARIOS
    # ========================================================

    st.subheader("Manual Price Scenarios")

    p1, p2, p3 = st.columns(3)

    with p1:

        st.markdown(
            f"""
            <div class="card">

            <div class="title">
                BEARISH SCENARIO
            </div>

            <div class="metric bear">
                ₹{bearish_target1:.2f}
            </div>

            <div class="small">
                First downside support
            </div>

            <hr>

            <b>S2:</b> ₹{bearish_target2:.2f}<br>
            <b>20D Low:</b> ₹{low20:.2f}<br>
            <b>52W Low:</b> ₹{low52:.2f}

            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:

        st.markdown(
            f"""
            <div class="card">

            <div class="title">
                BASE / RANGE SCENARIO
            </div>

            <div class="metric neutral">
                ₹{base_low:.2f} — ₹{base_high:.2f}
            </div>

            <div class="small">
                Pivot / ATR-based operating range
            </div>

            <hr>

            <b>Pivot:</b> ₹{pivot:.2f}<br>
            <b>ATR14:</b> ₹{atr14:.2f}<br>
            <b>ATR %:</b> {atr_percent:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )

    with p3:

        st.markdown(
            f"""
            <div class="card">

            <div class="title">
                BULLISH SCENARIO
            </div>

            <div class="metric bull">
                ₹{bullish_target1:.2f}
            </div>

            <div class="small">
                First upside resistance
            </div>

            <hr>

            <b>R2:</b> ₹{bullish_target2:.2f}<br>
            <b>20D High:</b> ₹{high20:.2f}<br>
            <b>52W High:</b> ₹{high52:.2f}

            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    st.subheader("Moving Average Structure")

    ma_df = pd.DataFrame({
        "Indicator": [
            "SMA 5",
            "SMA 10",
            "SMA 20",
            "SMA 50",
            "SMA 100",
            "SMA 200",
            "EMA 20",
            "EMA 50"
        ],
        "Value": [
            sma5,
            sma10,
            sma20,
            sma50,
            sma100,
            sma200,
            ema20,
            ema50
        ]
    })

    ma_df["Position"] = ma_df["Value"].apply(
        lambda x: "ABOVE" if price > x else "BELOW"
    )

    st.dataframe(
        ma_df.style.format({
            "Value": "₹{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    st.subheader("Momentum & Volatility")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "RSI 14",
        f"{rsi14:.1f}"
    )

    m2.metric(
        "MACD",
        f"{macd_now:.2f}"
    )

    m3.metric(
        "ATR 14",
        f"₹{atr14:.2f}"
    )

    m4.metric(
        "Volume Ratio",
        f"{volume_ratio:.2f}x"
    )

    # ========================================================
    # PIVOTS
    # ========================================================

    st.subheader("Classic Pivot Structure")

    pivot_df = pd.DataFrame({
        "Level": [
            "R3",
            "R2",
            "R1",
            "Pivot",
            "S1",
            "S2",
            "S3"
        ],
        "Price": [
            r3,
            r2,
            r1,
            pivot,
            s1,
            s2,
            s3
        ]
    })

    st.dataframe(
        pivot_df.style.format({
            "Price": "₹{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # BOLLINGER
    # ========================================================

    st.subheader("Bollinger Bands")

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Upper",
        f"₹{bb_upper:.2f}"
    )

    b2.metric(
        "Middle",
        f"₹{bb_middle:.2f}"
    )

    b3.metric(
        "Lower",
        f"₹{bb_lower:.2f}"
    )

    # ========================================================
    # FIBONACCI
    # ========================================================

    st.subheader("20-Day Fibonacci Levels")

    fib_df = pd.DataFrame({
        "Retracement": list(fib.keys()),
        "Price": list(fib.values())
    })

    st.dataframe(
        fib_df.style.format({
            "Price": "₹{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # TECHNICAL LEVELS
    # ========================================================

    st.subheader("Important Price Map")

    levels = pd.DataFrame({
        "Level": [
            "52W High",
            "20D High",
            "Bullish R2",
            "Bullish R1",
            "Current Price",
            "Pivot",
            "Bearish S1",
            "Bearish S2",
            "20D Low",
            "52W Low"
        ],
        "Price": [
            high52,
            high20,
            bullish_target2,
            bullish_target1,
            price,
            pivot,
            bearish_target1,
            bearish_target2,
            low20,
            low52
        ]
    })

    levels["Distance %"] = (
        (levels["Price"] / price) - 1
    ) * 100

    st.dataframe(
        levels.style.format({
            "Price": "₹{:.2f}",
            "Distance %": "{:+.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # WHAT MUST HAPPEN NEXT
    # ========================================================

    st.subheader("Confirmation Rules")

    confirmation = []

    confirmation.append(
        f"Break above ₹{r1:.2f} = first bullish confirmation."
    )

    confirmation.append(
        f"Break above ₹{r2:.2f} = stronger upside confirmation."
    )

    confirmation.append(
        f"Break below ₹{s1:.2f} = first bearish confirmation."
    )

    confirmation.append(
        f"Break below ₹{s2:.2f} = stronger downside confirmation."
    )

    confirmation.append(
        f"Price above SMA20 ({sma20:.2f}) = short-term trend improvement."
    )

    confirmation.append(
        f"Price above SMA50 ({sma50:.2f}) = medium-term trend improvement."
    )

    confirmation.append(
        f"Volume > 1.2x average is required for strong breakout confirmation."
    )

    for item in confirmation:
        st.markdown(f"- {item}")

    # ========================================================
    # INTRADAY CHART
    # ========================================================

    st.subheader("5-Minute Intraday Price")

    if not intraday.empty:

        chart_df = intraday[["Close"]].copy()
        chart_df.columns = ["Price"]

        st.line_chart(
            chart_df,
            height=300
        )

    else:

        st.info(
            "Intraday data unavailable."
        )

    # ========================================================
    # DISCLAIMER / METHODOLOGY
    # ========================================================

    st.markdown("---")

    st.caption(
        """
        Methodology: This terminal uses deterministic technical-analysis
        calculations only. It does not use AI, machine learning, neural
        networks, predictive algorithms, or external price predictions.

        No technical formula can guarantee future prices. Price scenarios
        are calculated from historical market data and should be treated
        as technical levels rather than guaranteed forecasts.
        """
    )
