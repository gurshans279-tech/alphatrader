import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from datetime import datetime
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="AlphaTrader Pro", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #141b2d 100%); }
[data-testid="stSidebar"] { background: rgba(20, 27, 45, 0.9); border-right: 1px solid #2a3548; }
.glass-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px; margin-bottom: 16px; }
.metric-value { font-size: 28px; font-weight: 700; color: #00ff7f; }
.metric-label { font-size: 12px; color: #8b98a8; text-transform: uppercase; }
.badge { padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px; display: inline-block; }
.badge-buy { background: rgba(0,217,126,0.15); color: #00d97e; border: 1px solid #00d97e; }
.badge-sell { background: rgba(255,56,96,0.15); color: #ff3860; border: 1px solid #ff3860; }
.badge-hold { background: rgba(255,184,0,0.15); color: #ffb800; border: 1px solid #ffb800; }
</style>
""", unsafe_allow_html=True)

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - (100 / (1 + g / l))

def atr(df, n=14):
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vwap(df):
    tp = (df["High"]+df["Low"]+df["Close"])/3
    return (tp*df["Volume"]).cumsum() / df["Volume"].cumsum()

@st.cache_data(ttl=30)
def get_universe():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500 = tables[0]["Symbol"].tolist()
        return sp500
    except:
        return ["AAPL","NVDA","TSLA","AMD","MSFT","AMZN","META","GOOGL","NFLX","AVGO",
                "COIN","PLTR","SOFI","MARA","HOOD","SNAP","UBER","SHOP","INTC","MU"]

@st.cache_data(ttl=15)
def load_intraday(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_daily(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def scan_for_day_setup(ticker):
    df = load_intraday(ticker)
    if df.empty or len(df) < 20:
        return None
    df["RSI"] = rsi(df["Close"])
    df["VWAP"] = vwap(df)
    df["ATR"] = atr(df)
    last = df.iloc[-1]
    p, a, r, v = float(last["Close"]), float(last["ATR"]), float(last["RSI"]), float(last["VWAP"])
    score = 0
    if 50 < r < 70 and p > v: score += 3
    if r < 35 and p > v: score += 2
    avg_vol = df["Volume"].tail(20).mean()
    if last["Volume"] > avg_vol * 1.5: score += 2
    if p > df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]: score += 1
    if score < 3:
        return None
    strat = "Momentum" if 50 < r < 70 else "Oversold Bounce"
    stop = p - 1.5*a
    target = p + 2.5*a
    return {"symbol": ticker, "price": p, "score": score, "strategy": strat,
            "stop": stop, "target": target, "rsi": r}

def run_full_scan():
    universe = get_universe()
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(scan_for_day_setup, t): t for t in universe}
        for future in as_completed(future_to_ticker):
            try:
                result = future.result(timeout=10)
                if result:
                    results.append(result)
            except:
                pass
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:30]

def monte_carlo(ticker, n_sims=10000):
    df = load_intraday(ticker)
    if df.empty or len(df) < 100:
        return None
    rets = df["Close"].pct_change().dropna().values
    last = float(df["Close"].iloc[-1])
    a = float(atr(df).iloc[-1])
    target = last + 2.5 * a
    stop = last - 1.5 * a
    rng = np.random.default_rng(42)
    hit_t, hit_s, neither = 0, 0, 0
    for _ in range(n_sims):
        path = last * np.cumprod(1 + rng.choice(rets, 78))
        for p in path:
            if p >= target:
                hit_t += 1; break
            if p <= stop:
                hit_s += 1; break
        else:
            neither += 1
    return {"win_rate": hit_t / n_sims, "loss_rate": hit_s / n_sims,
            "flat_rate": neither / n_sims, "n_sims": n_sims,
            "target": target, "stop": stop}

@st.cache_data(ttl=300)
def fetch_news(t, n=10):
    out = []
    try:
        for i in (yf.Ticker(t).news or [])[:n]:
            out.append({"source": i.get("publisher", "Yahoo"), "title": i.get("title", "")})
    except:
        pass
    return out

def generate_rationale(price, rsi_val, vwap_val, atr_val, macd_up, sma20, sma50):
    reasons = []
    if rsi_val < 30:
        reasons.append(f"RSI ({rsi_val:.0f}) is deeply oversold, signaling a potential bounce.")
    elif rsi_val > 70:
        reasons.append(f"RSI ({rsi_val:.0f}) is overbought, suggesting a pullback risk.")
    if price > vwap_val:
        reasons.append(f"Price is above VWAP (${vwap_val:.2f}), showing intraday buying pressure.")
    else:
        reasons.append(f"Price is below VWAP (${vwap_val:.2f}), showing intraday selling pressure.")
    if macd_up:
        reasons.append("MACD is above its signal line, confirming bullish momentum.")
    else:
        reasons.append("MACD is below its signal line, indicating fading momentum.")
    if price > sma20 and sma20 > sma50:
        reasons.append("Moving averages are stacked up, a bullish alignment.")
    elif price < sma20 and sma20 < sma50:
        reasons.append("Moving averages are stacked down, a bearish alignment.")
    return reasons

st.sidebar.markdown("## ⚡ AlphaTrader Pro")
st.sidebar.caption("Real-time market intelligence. Not financial advice.")
tab1, tab2 = st.tabs(["⚡ Day Trading", "📈 Swing Trading"])

with tab1:
    st.header("Day Trading · Live Scanner")
    st.caption("Scanning S&P 500 for high-probability setups. Updates every 15s.")
    @st.fragment(run_every="15s")
    def live_scanner():
        results = run_full_scan()
        if not results:
            st.warning("No high-conviction setups found. Check back shortly.")
            return
        st.success(f"Found {len(results)} high-conviction setups across the market.")
        df_res = pd.DataFrame(results)
        df_res["R:R"] = (df_res["target"] - df_res["price"]) / (df_res["price"] - df_res["stop"])
        st.dataframe(df_res[["symbol","strategy","price","score","R:R","rsi","stop","target"]],
                     use_container_width=True, hide_index=True)
    live_scanner()
    st.divider()
    st.subheader("🔍 Analyze a Specific Stock")
    ticker = st.text_input("Enter ticker", "TSLA", key="day_ticker").upper()
    if ticker:
        df_intraday = load_intraday(ticker)
        if not df_intraday.empty:
            df_intraday["RSI"] = rsi(df_intraday["Close"])
            df_intraday["VWAP"] = vwap(df_intraday)
            df_intraday["ATR"] = atr(df_intraday)
            macd_line = df_intraday["Close"].ewm(span=12, adjust=False).mean() - df_intraday["Close"].ewm(span=26, adjust=False).mean()
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_up = macd_line.iloc[-1] > macd_signal.iloc[-1]
            last = df_intraday.iloc[-1]
            price = float(last["Close"]); atr_val = float(last["ATR"])
            rsi_val = float(last["RSI"]); vwap_val = float(last["VWAP"])
            score = 0
            if rsi_val < 30: score += 2
            if rsi_val > 70: score -= 2
            score += 2 if price > vwap_val else -2
            score += 2 if macd_up else -2
            verdict = "BUY" if score >= 3 else "SELL" if score <= -3 else "HOLD"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${price:.2f}")
            c2.metric("RSI", f"{rsi_val:.0f}")
            c3.metric("VWAP", f"${vwap_val:.2f}")
            c4.metric("ATR", f"${atr_val:.2f}")
            st.markdown("### 🧠 Trade Rationale")
            sma20 = float(df_intraday["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(df_intraday["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
            reasons = generate_rationale(price, rsi_val, vwap_val, atr_val, macd_up, sma20, sma50)
            badge_class = "badge-buy" if verdict == "BUY" else "badge-sell" if verdict == "SELL" else "badge-hold"
            st.markdown(f'<div class="glass-card"><span class="badge {badge_class}">{verdict}</span>', unsafe_allow_html=True)
            for reason in reasons:
                st.markdown(f"- {reason}")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("### 🎲 Probability of Profit (10,000 Simulations)")
            tm = monte_carlo(ticker)
            if tm:
                col1, col2, col3 = st.columns(3)
                col1.metric("🎯 Win Rate", f"{tm['win_rate']*100:.1f}%")
                col2.metric("🛑 Loss Rate", f"{tm['loss_rate']*100:.1f}%")
                col3.metric("⏰ Flat", f"{tm['flat_rate']*100:.1f}%")
                st.caption(f"Target: ${tm['target']:.2f} | Stop: ${tm['stop']:.2f}")

with tab2:
    st.header("Swing Trading · 2-4 Day Holds")
    ticker_swing = st.text_input("Enter ticker", "AAPL", key="swing_ticker").upper()
    if ticker_swing:
        df_daily = load_daily(ticker_swing)
        if not df_daily.empty:
            df_daily["RSI"] = rsi(df_daily["Close"])
            df_daily["SMA20"] = df_daily["Close"].rolling(20).mean()
            df_daily["SMA50"] = df_daily["Close"].rolling(50).mean()
            df_daily["ATR"] = atr(df_daily)
            last = df_daily.iloc[-1]
            price = float(last["Close"]); rsi_val = float(last["RSI"]); atr_val = float(last["ATR"])
            sma20 = float(last["SMA20"]); sma50 = float(last["SMA50"])
            score = 0
            if price > sma50: score += 2
            if price > sma20 and sma20 > sma50: score += 2
            if rsi_val < 40: score += 1
            verdict = "BUY" if score >= 4 else "SELL" if score <= -2 else "HOLD"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${price:.2f}")
            c2.metric("RSI", f"{rsi_val:.0f}")
            c3.metric("20/50 SMA", f"${sma20:.0f}/${sma50:.0f}")
            c4.metric("ATR", f"${atr_val:.2f}")
            stop = price - 2.0 * atr_val
            target = price + 4.0 * atr_val
            st.markdown(f"### Verdict: **{verdict}**")
            st.write(f"Entry ${price:.2f} · Stop ${stop:.2f} · Target ${target:.2f}")
