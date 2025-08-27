# app.py
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import date, timedelta

# ---- Streamlit page setup ----
st.set_page_config(
    page_title="Stock Market Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Analytics Dashboard")
st.caption("Real-time analytics powered by Yahoo Finance (`yfinance`).")

# ---- Sidebar controls ----
st.sidebar.header("Controls")

default_tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]
tickers = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=",".join(default_tickers)
).upper().replace(" ", "").split(",")

today = date.today()
one_year_ago = today - timedelta(days=365)

start_date = st.sidebar.date_input("Start date", one_year_ago)
end_date = st.sidebar.date_input("End date", today)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# Moving average windows
ma_short = st.sidebar.number_input("Short MA window", min_value=5, max_value=100, value=20, step=1)
ma_long  = st.sidebar.number_input("Long MA window",  min_value=20, max_value=400, value=50, step=5)

# ---- Currency toggle ----
currency_option = st.sidebar.radio("Currency", ["USD ($)", "INR (₹)"])
usd_inr_rate = None
if currency_option == "INR (₹)":
    forex = yf.download("USDINR=X", period="5d", interval="1d")
    if not forex.empty:
        usd_inr_rate = float(forex["Close"].iloc[-1])  # ✅ ensure float

# ---- Helper: trading periods for volatility annualization ----
annualization_map = {"1d": 252, "1wk": 52, "1mo": 12}
ann_factor = annualization_map.get(interval, 252)

# ---- Fetch data ----
@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_prices(tickers, start, end, interval):
    df = yf.download(
        tickers,
        start=str(start),
        end=str(end + timedelta(days=1)),  # inclusive end
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )
    return df

if st.sidebar.button("Fetch / Refresh"):
    st.cache_data.clear()

with st.spinner("Fetching data from Yahoo Finance..."):
    data = fetch_prices(tickers, start_date, end_date, interval)

if data is None or len(data) == 0:
    st.warning("No data returned. Check tickers/date range.")
    st.stop()

# ---- Normalize data into {ticker: DataFrame} ----
def split_by_ticker(df, tickers):
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t in df.columns.get_level_values(0):
                out[t] = df[t].dropna(how="all")
    else:
        out[tickers[0]] = df.dropna(how="all")
    return out

by_ticker = split_by_ticker(data, tickers)
present_tickers = list(by_ticker.keys())
missing = sorted(set(tickers) - set(present_tickers))
if missing:
    st.info(f"Skipped missing/invalid tickers: {', '.join(missing)}")

# ---- KPIs ----
def compute_kpis(df_close: pd.Series):
    last_price = float(df_close.dropna().iloc[-1])
    period_ret = float(df_close.dropna().iloc[-1] / df_close.dropna().iloc[0] - 1.0)
    daily_ret = df_close.pct_change().dropna()
    ann_vol = float(daily_ret.std() * math.sqrt(ann_factor))
    return last_price, period_ret, ann_vol

st.subheader("📊 Key Metrics")
k_cols = st.columns(min(4, len(present_tickers)))
for i, t in enumerate(present_tickers[:4]):
    close = by_ticker[t]["Close"].dropna()
    if len(close) < 2:
        continue
    last_price, period_ret, ann_vol = compute_kpis(close)

    with k_cols[i]:
        if currency_option == "INR (₹)" and usd_inr_rate is not None:
            st.metric(
                label=f"{t} — Last Price",
                value=f"₹{last_price * usd_inr_rate:,.2f}",
                delta=f"{period_ret*100:.2f}%"
            )
        else:
            st.metric(
                label=f"{t} — Last Price",
                value=f"${last_price:,.2f}",
                delta=f"{period_ret*100:.2f}%"
            )
        st.caption(f"Annualized Volatility: {ann_vol*100:.2f}%  |  Bars: {len(close)}")

st.divider()

# ---- Price + Moving Averages ----
st.subheader("📉 Price & Moving Averages")
for t in present_tickers:
    df_t = by_ticker[t].copy()
    if "Close" not in df_t or df_t["Close"].dropna().empty:
        continue

    df_t["MA_short"] = df_t["Close"].rolling(ma_short).mean()
    df_t["MA_long"]  = df_t["Close"].rolling(ma_long).mean()

    if currency_option == "INR (₹)" and usd_inr_rate is not None:
        df_t["Close"] = df_t["Close"] * usd_inr_rate
        df_t["MA_short"] = df_t["MA_short"] * usd_inr_rate
        df_t["MA_long"] = df_t["MA_long"] * usd_inr_rate

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_t.index, df_t["Close"], label=f"{t} Close")
    ax.plot(df_t.index, df_t["MA_short"], label=f"{ma_short}-bar MA")
    ax.plot(df_t.index, df_t["MA_long"],  label=f"{ma_long}-bar MA")
    ax.set_title(f"{t} Close with Moving Averages ({interval})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)" if currency_option == "INR (₹)" else "Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.divider()

# ---- Returns distribution ----
st.subheader("📦 Daily Returns Distribution")
cols = st.columns(2)
for idx, t in enumerate(present_tickers[:4]):
    df_t = by_ticker[t].copy()
    if "Close" not in df_t or len(df_t["Close"]) < 2:
        continue
    returns = df_t["Close"].pct_change().dropna()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(returns, bins=40, edgecolor="black")
    ax.set_title(f"{t} — Daily Returns Histogram")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.2)
    cols[idx % 2].pyplot(fig)

st.divider()

# ---- Correlation heatmap ----
st.subheader("🔗 Correlation Between Tickers (Daily Returns)")
retdict = {}
for t in present_tickers:
    close = by_ticker[t].get("Close", pd.Series(dtype=float)).pct_change().rename(t)
    retdict[t] = close
returns_df = pd.concat(retdict.values(), axis=1).dropna(how="any")
if not returns_df.empty and len(returns_df.columns) > 1:
    corr = returns_df.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation of Daily Returns")
    st.pyplot(fig)
else:
    st.info("Need at least 2 valid tickers with overlapping dates to compute correlation.")

st.divider()

# ---- Portfolio ----
st.subheader("💼 Equal-Weight Portfolio (Cumulative Return)")
if not returns_df.empty:
    eq_ret = returns_df.mean(axis=1)
    cum_ret = (1 + eq_ret).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum_ret.index, cum_ret, label="Equal-Weight Portfolio")
    ax.set_title("Equal-Weight Portfolio Cumulative Growth")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of ₹1" if currency_option == "INR (₹)" else "Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    last_vals = {}
    for t in present_tickers:
        c = by_ticker[t]["Close"].dropna()
        if len(c) > 1:
            last_vals[t] = float(c.iloc[-1] / c.iloc[0] - 1)
    if last_vals:
        top_df = pd.Series(last_vals).sort_values(ascending=False).to_frame("Period Return")
        st.write("**Top movers (period % return):**")
        st.dataframe((top_df * 100).round(2))
else:
    st.info("Not enough data to compute portfolio metrics.")

st.divider()

st.caption(
    "Data source: Yahoo Finance via the `yfinance` library. "
    "This dashboard is for educational/analytic purposes only."
)
