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
[data-testid="stAppViewContainer"], .stApp { background: radial-gradient(ellipse at top, #0f172a 0%, #050810 100%); }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0e1a 0%, #141b2d 100%); border-right: 1px solid #1e293b; }
h1, h2, h3, h4 { color: #f1f5f9 !important; font-weight: 800 !important; letter-spacing: -0.8px !important; }
[data-testid="stMetricValue"] { color: #10b981 !important; font-family: 'SF Mono', Menlo, monospace !important; font-weight: 800 !important; font-size: 26px !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; text-transform: uppercase; font-size: 10px; letter-spacing: 1.5px; font-weight: 600; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: #000 !important; font-weight: 800; border: none; border-radius: 10px; padding: 14px 28px; box-shadow: 0 4px 20px rgba(16,185,129,0.4); }
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(16,185,129,0.6); }
.stButton > button { background: #1e293b; color: #f1f5f9 !important; border: 1px solid #334155; border-radius: 10px; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; border-bottom: 1px solid #1e293b; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #94a3b8; font-weight: 700; padding: 12px 24px; font-size: 16px; border: none; }
.stTabs [aria-selected="true"] { color: #10b981 !important; border-bottom: 2px solid #10b981 !important; }
div[data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; }
.stTextInput input { background: #0f172a !important; color: #f1f5f9 !important; border: 1px solid #334155 !important; border-radius: 8px; }
.glass { background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px; margin: 12px 0; }
.badge { padding: 10px 24px; border-radius: 30px; font-weight: 900; font-size: 22px; display: inline-block; letter-spacing: 2px; font-family: 'SF Mono', Menlo, monospace; }
.badge-buy { background: #10b981; color: #000; box-shadow: 0 0 40px rgba(16,185,129,0.5); }
.badge-sell { background: #ef4444; color: #fff; box-shadow: 0 0 40px rgba(239,68,68,0.5); }
.badge-hold { background: #f59e0b; color: #000; box-shadow: 0 0 40px rgba(245,158,11,0.4); }
.news-row { padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06); line-height: 1.4; }
.news-src { color: #10b981; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-right: 8px; }
.news-title { color: #cbd5e1; font-size: 14px; }
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

@st.cache_data(ttl=60)
def get_universe():
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return t[0]["Symbol"].tolist()
    except:
        return ["AAPL","NVDA","TSLA","AMD","MSFT","AMZN","META","GOOGL","NFLX","AVGO",
                "COIN","PLTR","SOFI","MARA","HOOD","SNAP","UBER","SHOP","INTC","MU"]

@st.cache_data(ttl=15)
def load_intraday(t):
    try:
        df = yf.download(t, period="2d", interval="5m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_daily(t):
    try:
        df = yf.download(t, period="1y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_news(t, n=40):
    out, seen = [], set()
    for q in [f"{t} stock", f"{t} earnings", f"{t} analyst", f"{t} upgrade", f"{t} downgrade"]:
        try:
            url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url, request_headers={"User-Agent":"Mozilla/5.0"})
            for e in feed.entries[:8]:
                title = e.title
                pub = title.rsplit(" - ", 1)[-1] if " - " in title else "Google News"
                key = title.lower()[:45]
                if key and key not in seen:
                    seen.add(key); out.append({"source": pub, "title": title})
        except:
            pass
    try:
        for i in (yf.Ticker(t).news or [])[:20]:
            title = i.get("title", "")
            key = title.lower()[:45]
            if key and key not in seen:
                seen.add(key); out.append({"source": i.get("publisher","Yahoo"), "title": title})
    except:
        pass
    return out[:n]

def monte_carlo(t, n_sims=10000, horizon="day"):
    df = load_intraday(t) if horizon == "day" else load_daily(t)
    if df.empty or len(df) < 100:
        return None
    rets = df["Close"].pct_change().dropna().values
    last = float(df["Close"].iloc[-1]); a = float(atr(df).iloc[-1])
    if horizon == "day":
        target, stop, bars = last + 2.5*a, last - 1.5*a, 78
    else:
        target, stop, bars = last + 4*a, last - 2*a, 4
    rng = np.random.default_rng(42); ht = hs = hn = 0
    for _ in range(n_sims):
        path = last * np.cumprod(1 + rng.choice(rets, bars))
        for p in path:
            if p >= target: ht += 1; break
            if p <= stop: hs += 1; break
        else:
            hn += 1
    return {"win_rate": ht/n_sims, "loss_rate": hs/n_sims, "flat_rate": hn/n_sims,
            "n": n_sims, "target": target, "stop": stop}

def scan_day(t):
    df = load_intraday(t)
    if df.empty or len(df) < 20: return None
    df["RSI"] = rsi(df["Close"]); df["VWAP"] = vwap(df); df["ATR"] = atr(df)
    last = df.iloc[-1]
    p, a, r, v = float(last["Close"]), float(last["ATR"]), float(last["RSI"]), float(last["VWAP"])
    score = 0
    if 50 < r < 70 and p > v: score += 3
    if r < 35 and p > v: score += 2
    if last["Volume"] > df["Volume"].tail(20).mean() * 1.5: score += 2
    if p > df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]: score += 1
    if score < 3: return None
    strat = "Momentum" if 50 < r < 70 else "Oversold Bounce"
    return {"symbol": t, "strategy": strat, "price": p, "score": score,
            "rsi": r, "stop": p - 1.5*a, "target": p + 2.5*a}

def scan_swing(t):
    df = load_daily(t)
    if df.empty or len(df) < 60: return None
    df["RSI"] = rsi(df["Close"])
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["ATR"] = atr(df)
    last = df.iloc[-1]
    p, r, a = float(last["Close"]), float(last["RSI"]), float(last["ATR"])
    s20, s50 = float(last["SMA20"]), float(last["SMA50"])
    s200 = float(last["SMA200"]) if pd.notna(last["SMA200"]) else 0
    avgvol = df["Volume"].tail(20).mean()
    score = 0
    if p > s50: score += 2
    if p > s20 and s20 > s50: score += 2
    if r < 40: score += 1
    if s200 and p > s200: score += 1
    if p > df["High"].iloc[-21:-1].max() and last["Volume"] > avgvol * 1.3: score += 2
    if score < 3: return None
    return {"symbol": t, "price": p, "rsi": r, "score": score,
            "sma20": s20, "sma50": s50, "stop": p - 2*a, "target": p + 4*a}

def run_scan(scan_fn):
    universe = get_universe()
    results = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(scan_fn, t): t for t in universe}
        for f in as_completed(futs):
            try:
                r = f.result(timeout=10)
                if r: results.append(r)
            except: pass
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:30]

def rationale_day(p, r, v, macd_up, s20, s50):
    out = []
    if r < 30: out.append(f"RSI ({r:.0f}) oversold — bounce setup")
    elif r > 70: out.append(f"RSI ({r:.0f}) overbought — pullback risk")
    else: out.append(f"RSI ({r:.0f}) neutral — no extreme")
    out.append(f"Price {'above' if p>v else 'below'} VWAP (${v:.2f}) — {'buyers' if p>v else 'sellers'} in control")
    out.append("MACD bullish — momentum confirms" if macd_up else "MACD bearish — momentum fading")
    if p > s20 and s20 > s50: out.append("Moving averages stacked up — bullish alignment")
    elif p < s20 and s20 < s50: out.append("Moving averages stacked down — bearish alignment")
    return out

def rationale_swing(p, r, s20, s50, s200):
    out = []
    if p > s50: out.append(f"Price above 50-day SMA (${s50:.2f}) — medium-term uptrend")
    else: out.append(f"Price below 50-day SMA (${s50:.2f}) — medium-term downtrend")
    if s200 and p > s200: out.append(f"Price above 200-day SMA (${s200:.2f}) — long-term uptrend")
    elif s200: out.append(f"Price below 200-day SMA (${s200:.2f}) — long-term weakness")
    if r < 40: out.append(f"RSI ({r:.0f}) not overbought — room to run")
    elif r > 70: out.append(f"RSI ({r:.0f}) overbought — pullback risk")
    return out

st.sidebar.markdown("## ⚡ AlphaTrader Pro")
st.sidebar.caption("Not financial advice. Scans S&P 500. 500 stocks, not 10,000.")
tab1, tab2 = st.tabs(["⚡ Day Trading", "📈 Swing Trading"])

with tab1:
    st.header("⚡ Day Trading")
    st.caption("Scanning the S&P 500. Auto-refresh 15s.")
    @st.fragment(run_every="15s")
    def day_scanner():
        results = run_scan(scan_day)
        if not results:
            st.warning("No high-conviction setups right now. Market conditions not aligned.")
            return
        st.success(f"✅ Found {len(results)} setups")
        df = pd.DataFrame(results)
        df["R:R"] = (df["target"] - df["price"]) / (df["price"] - df["stop"])
        st.dataframe(df[["symbol","strategy","price","score","R:R","rsi","stop","target"]],
                     use_container_width=True, hide_index=True)
    day_scanner()
    st.divider()
    st.subheader("🔍 Deep Analysis")
    d_sym = st.text_input("Ticker", "TSLA", key="d").upper()
    if st.button("Analyze", key="da", type="primary"):
        df = load_intraday(d_sym)
        if df.empty:
            st.error(f"No data for {d_sym}")
        else:
            df["RSI"] = rsi(df["Close"]); df["VWAP"] = vwap(df); df["ATR"] = atr(df)
            ml = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
            ms = ml.ewm(span=9, adjust=False).mean()
            macd_up = ml.iloc[-1] > ms.iloc[-1]
            last = df.iloc[-1]
            p, a, r, v = float(last["Close"]), float(last["ATR"]), float(last["RSI"]), float(last["VWAP"])
            sc = (2 if r<30 else -2 if r>70 else 0) + (2 if p>v else -2) + (2 if macd_up else -2)
            verdict = "BUY" if sc>=3 else "SELL" if sc<=-3 else "HOLD"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${p:.2f}"); c2.metric("RSI", f"{r:.0f}")
            c3.metric("VWAP", f"${v:.2f}"); c4.metric("ATR", f"${a:.2f}")
            bc = "badge-buy" if verdict=="BUY" else "badge-sell" if verdict=="SELL" else "badge-hold"
            st.markdown(f'<div class="glass"><span class="badge {bc}">{verdict}</span></div>', unsafe_allow_html=True)
            st.markdown("### 🧠 Why")
            s20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
            s50 = float(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
            for line in rationale_day(p, r, v, macd_up, s20, s50):
                st.markdown(f"- {line}")
            st.markdown("### 💰 Price Levels")
            st.write(f"**Buy** ${p-1.5*a:.2f} · **Stop** ${p-3.5*a:.2f} · **Target** ${p+3*a:.2f}")
            st.markdown("### 🎲 10,000 Simulations")
            tm = monte_carlo(d_sym, 10000, "day")
            if tm:
                m1, m2, m3 = st.columns(3)
                m1.metric("🎯 Win Rate", f"{tm['win_rate']*100:.1f}%")
                m2.metric("🛑 Loss Rate", f"{tm['loss_rate']*100:.1f}%")
                m3.metric("⏰ Flat", f"{tm['flat_rate']*100:.1f}%")
            st.markdown("### 📰 News")
            for n in fetch_news(d_sym):
                st.markdown(f'<div class="news-row"><span class="news-src">{n["source"]}</span><span class="news-title">{n["title"]}</span></div>', unsafe_allow_html=True)

with tab2:
    st.header("📈 Swing Trading · 2-4 Day Holds")
    st.caption("Scanning S&P 500 for multi-day setups.")
    @st.fragment(run_every="60s")
    def swing_scanner():
        results = run_scan(scan_swing)
        if not results:
            st.warning("No swing setups right now.")
            return
        st.success(f"✅ Found {len(results)} swing setups")
        df = pd.DataFrame(results)
        df["R:R"] = (df["target"] - df["price"]) / (df["price"] - df["stop"])
        st.dataframe(df[["symbol","price","score","R:R","rsi","stop","target"]],
                     use_container_width=True, hide_index=True)
    swing_scanner()
    st.divider()
    st.subheader("🔍 Deep Analysis")
    s_sym = st.text_input("Ticker", "AAPL", key="s").upper()
    if st.button("Analyze", key="sa", type="primary"):
        df = load_daily(s_sym)
        if df.empty:
            st.error(f"No data for {s_sym}")
        else:
            df["RSI"] = rsi(df["Close"])
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()
            df["SMA200"] = df["Close"].rolling(200).mean()
            df["ATR"] = atr(df)
            last = df.iloc[-1]
            p, r, a = float(last["Close"]), float(last["RSI"]), float(last["ATR"])
            s20, s50 = float(last["SMA20"]), float(last["SMA50"])
            s200 = float(last["SMA200"]) if pd.notna(last["SMA200"]) else 0
            sc = 0
            if p > s50: sc += 2
            if p > s20 and s20 > s50: sc += 2
            if r < 40: sc += 1
            if s200 and p > s200: sc += 1
            verdict = "BUY" if sc>=4 else "SELL" if sc<=-2 else "HOLD"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${p:.2f}"); c2.metric("RSI", f"{r:.0f}")
            c3.metric("20/50 SMA", f"${s20:.0f}/${s50:.0f}"); c4.metric("ATR", f"${a:.2f}")
            bc = "badge-buy" if verdict=="BUY" else "badge-sell" if verdict=="SELL" else "badge-hold"
            st.markdown(f'<div class="glass"><span class="badge {bc}">{verdict}</span></div>', unsafe_allow_html=True)
            st.markdown("### 🧠 Why")
            for line in rationale_swing(p, r, s20, s50, s200):
                st.markdown(f"- {line}")
            st.markdown("### 💰 Price Levels")
            st.write(f"**Buy** ${p-a:.2f} · **Stop** ${p-2.5*a:.2f} · **Target** ${p+4*a:.2f}")
            st.markdown("### 🎲 10,000 Simulations (4-day)")
            tm = monte_carlo(s_sym, 10000, "swing")
            if tm:
                m1, m2, m3 = st.columns(3)
                m1.metric("🎯 Win Rate", f"{tm['win_rate']*100:.1f}%")
                m2.metric("🛑 Loss Rate", f"{tm['loss_rate']*100:.1f}%")
                m3.metric("⏰ Flat", f"{tm['flat_rate']*100:.1f}%")
            st.markdown("### 📰 News")
            for n in fetch_news(s_sym):
                st.markdown(f'<div class="news-row"><span class="news-src">{n["source"]}</span><span class="news-title">{n["title"]}</span></div>', unsafe_allow_html=True)
