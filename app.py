import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── MODERN THEME ──────────────────────────────────────────────────────────────
try:
    from neonui import apply_theme
    apply_theme("midnight")  # A clean, modern dark theme
except ImportError:
    st.warning("NeonUI not installed. Run: pip install neonui")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AlphaTrader Pro", page_icon="⚡", layout="wide")

# ─── DATA & CORE FUNCTIONS ───────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_sp500_tickers():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return table[0]["Symbol"].tolist()
    except Exception:
        # Fallback list if Wikipedia is unavailable
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ"]

@st.cache_data(ttl=15)
def load_intraday(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_daily(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))

def atr(df, n=14):
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vwap(df):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()

def monte_carlo(symbol, horizon="day", n_sims=10000):
    df = load_intraday(symbol) if horizon == "day" else load_daily(symbol)
    if df.empty or len(df) < 100:
        return None
    returns = df["Close"].pct_change().dropna().values
    last_price = float(df["Close"].iloc[-1])
    vol = float(atr(df).iloc[-1])

    if horizon == "day":
        target, stop, bars = last_price + 3.0 * vol, last_price - 1.5 * vol, 78
    else:
        target, stop, bars = last_price + 4 * vol, last_price - 2 * vol, 4

    rng = np.random.default_rng(seed=42)
    hits_target, hits_stop, neither = 0, 0, 0

    for _ in range(n_sims):
        path = last_price * np.cumprod(1 + rng.choice(returns, bars))
        for price in path:
            if price >= target:
                hits_target += 1
                break
            if price <= stop:
                hits_stop += 1
                break
        else:
            neither += 1

    win_rate = hits_target / n_sims
    loss_rate = hits_stop / n_sims
    expected_value = (win_rate * (target - last_price)) - (loss_rate * (last_price - stop))

    return {
        "win_rate": win_rate, "loss_rate": loss_rate, "flat_rate": neither / n_sims,
        "expected_value": expected_value, "target": target, "stop": stop, "n_sims": n_sims
    }

# ─── APP LAYOUT ──────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚡ AlphaTrader Pro")
st.sidebar.caption("Real-time market intelligence. Not financial advice.")
st.sidebar.caption("Scans S&P 500. Free tier limit — not 10,000 stocks.")

tab_day, tab_swing, tab_fund = st.tabs(["⚡ Day Trading", "📈 Swing Trading", "💰 Fundamentals"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: DAY TRADING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_day:
    st.header("Day Trading · Live Scanner")
    st.caption("Scanning S&P 500. Auto-refreshes every 15 seconds.")

    @st.fragment(run_every="15s")
    def live_day_scanner():
        tickers = get_sp500_tickers()
        results = []
        progress_bar = st.progress(0, text="Scanning for intraday setups...")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(load_intraday, ticker): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures)):
                ticker = futures[future]
                df = future.result()
                if not df.empty and len(df) >= 20:
                    df["RSI"] = rsi(df["Close"])
                    df["VWAP"] = vwap(df)
                    df["ATR"] = atr(df)
                    last = df.iloc[-1]
                    price, vol, rsi_val, vwap_val = float(last["Close"]), float(last["ATR"]), float(last["RSI"]), float(last["VWAP"])

                    score = 0
                    if 50 < rsi_val < 70 and price > vwap_val: score += 3
                    if rsi_val < 35 and price > vwap_val: score += 2
                    if last["Volume"] > df["Volume"].tail(20).mean() * 1.5: score += 2
                    if price > df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]: score += 1

                    if score >= 3:
                        results.append({
                            "Symbol": ticker, "Strategy": "Momentum" if 50 < rsi_val < 70 else "Oversold Bounce",
                            "Price": price, "Score": score, "RSI": rsi_val,
                            "Stop": price - 1.5 * vol, "Target": price + 3.0 * vol
                        })
                progress_bar.progress((i + 1) / len(tickers), text=f"Scanned {i + 1}/{len(tickers)}")
        progress_bar.empty()

        if not results:
            st.warning("No high-conviction intraday setups found right now.")
        else:
            results.sort(key=lambda x: x["Score"], reverse=True)
            top_results = results[:30]
            st.success(f"✅ Found {len(results)} setups. Showing top {len(top_results)}.")
            df_results = pd.DataFrame(top_results)
            df_results["R:R"] = ((df_results["Target"] - df_results["Price"]) / (df_results["Price"] - df_results["Stop"])).round(1)
            st.dataframe(df_results[["Symbol", "Strategy", "Price", "Score", "R:R", "RSI", "Stop", "Target"]], use_container_width=True, hide_index=True)

    live_day_scanner()

    st.divider()
    st.subheader("🔍 Deep Analysis & Expected Value")
    day_symbol = st.text_input("Enter ticker for day trade analysis", "TSLA", key="day_input").upper()

    if st.button("Analyze Day Trade", key="day_btn", type="primary"):
        df = load_intraday(day_symbol)
        if df.empty:
            st.error(f"No intraday data for {day_symbol}")
        else:
            df["RSI"] = rsi(df["Close"])
            df["VWAP"] = vwap(df)
            df["ATR"] = atr(df)
            last = df.iloc[-1]
            price, vol, rsi_val, vwap_val = float(last["Close"]), float(last["ATR"]), float(last["RSI"]), float(last["VWAP"])
            macd_line = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_up = macd_line.iloc[-1] > macd_signal.iloc[-1]

            score = (2 if rsi_val < 30 else -2 if rsi_val > 70 else 0) + (2 if price > vwap_val else -2) + (2 if macd_up else -2)
            verdict = "BUY" if score >= 3 else "SELL" if score <= -3 else "HOLD"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"${price:.2f}")
            col2.metric("RSI", f"{rsi_val:.0f}")
            col3.metric("VWAP", f"${vwap_val:.2f}")
            col4.metric("ATR", f"${vol:.2f}")

            badge_class = "badge-buy" if verdict == "BUY" else "badge-sell" if verdict == "SELL" else "badge-hold"
            st.markdown(f'<div style="text-align:center; margin: 20px 0;"><span class="badge {badge_class}">{verdict}</span></div>', unsafe_allow_html=True)

            st.markdown("### 🧠 Trade Rationale")
            reasons = []
            if rsi_val < 30: reasons.append(f"RSI ({rsi_val:.0f}) is oversold, suggesting a potential bounce.")
            elif rsi_val > 70: reasons.append(f"RSI ({rsi_val:.0f}) is overbought, suggesting a pullback risk.")
            else: reasons.append(f"RSI ({rsi_val:.0f}) is neutral.")
            reasons.append(f"Price is {'above' if price > vwap_val else 'below'} VWAP (${vwap_val:.2f}), indicating {'buying' if price > vwap_val else 'selling'} pressure.")
            reasons.append("MACD is above its signal line, confirming bullish momentum." if macd_up else "MACD is below its signal line, indicating fading momentum.")
            for reason in reasons:
                st.markdown(f"- {reason}")

            st.markdown("### 🎲 Probability & Expected Value (10,000 Simulations)")
            mc_result = monte_carlo(day_symbol, "day")
            if mc_result:
                c1, c2, c3 = st.columns(3)
                c1.metric("🎯 Win Rate", f"{mc_result['win_rate']*100:.1f}%")
                c2.metric("🛑 Loss Rate", f"{mc_result['loss_rate']*100:.1f}%")
                ev = mc_result['expected_value']
                c3.metric("📊 Expected Value", f"${ev:+.2f}")

                if ev > 0:
                    st.success(f"✅ POSITIVE EDGE: On average, this trade would have made **+${ev:.2f}** per share. The math is on your side.")
                elif ev > -0.05:
                    st.warning(f"⚖️ BREAK-EVEN: Expected value is **${ev:+.2f}**. The trade is essentially a coin flip. Skipping is the smarter play.")
                else:
                    st.error(f"❌ NO EDGE: On average, this trade would have lost **-${abs(ev):.2f}** per share. The math is against you. Do not take it.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SWING TRADING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_swing:
    st.header("Swing Trading · 2-4 Day Holds")
    st.caption("Scanning S&P 500. Auto-refreshes every 60 seconds.")

    @st.fragment(run_every="60s")
    def live_swing_scanner():
        tickers = get_sp500_tickers()
        results = []
        progress_bar = st.progress(0, text="Scanning for swing setups...")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(load_daily, ticker): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures)):
                ticker = futures[future]
                df = future.result()
                if not df.empty and len(df) >= 60:
                    df["RSI"] = rsi(df["Close"])
                    df["SMA20"] = df["Close"].rolling(20).mean()
                    df["SMA50"] = df["Close"].rolling(50).mean()
                    df["SMA200"] = df["Close"].rolling(200).mean()
                    df["ATR"] = atr(df)
                    last = df.iloc[-1]
                    price, rsi_val, vol = float(last["Close"]), float(last["RSI"]), float(last["ATR"])
                    sma20, sma50 = float(last["SMA20"]), float(last["SMA50"])
                    sma200 = float(last["SMA200"]) if pd.notna(last["SMA200"]) else 0

                    score = 0
                    if price > sma50: score += 2
                    if price > sma20 and sma20 > sma50: score += 2
                    if rsi_val < 40: score += 1
                    if sma200 and price > sma200: score += 1
                    if price > df["High"].iloc[-21:-1].max(): score += 2

                    if score >= 4:
                        results.append({
                            "Symbol": ticker, "Price": price, "Score": score, "RSI": rsi_val,
                            "20d SMA": sma20, "50d SMA": sma50,
                            "Stop": price - 2 * vol, "Target": price + 4 * vol
                        })
                progress_bar.progress((i + 1) / len(tickers), text=f"Scanned {i + 1}/{len(tickers)}")
        progress_bar.empty()

        if not results:
            st.warning("No swing setups found right now.")
        else:
            results.sort(key=lambda x: x["Score"], reverse=True)
            top_results = results[:30]
            st.success(f"✅ Found {len(results)} swing setups. Showing top {len(top_results)}.")
            df_results = pd.DataFrame(top_results)
            df_results["R:R"] = ((df_results["Target"] - df_results["Price"]) / (df_results["Price"] - df_results["Stop"])).round(1)
            st.dataframe(df_results[["Symbol", "Price", "Score", "R:R", "RSI", "20d SMA", "50d SMA", "Stop", "Target"]], use_container_width=True, hide_index=True)

    live_swing_scanner()

    st.divider()
    st.subheader("🔍 Deep Analysis & Expected Value")
    swing_symbol = st.text_input("Enter ticker for swing analysis", "AAPL", key="swing_input").upper()

    if st.button("Analyze Swing Trade", key="swing_btn", type="primary"):
        df = load_daily(swing_symbol)
        if df.empty:
            st.error(f"No daily data for {swing_symbol}")
        else:
            df["RSI"] = rsi(df["Close"])
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()
            df["SMA200"] = df["Close"].rolling(200).mean()
            df["ATR"] = atr(df)
            last = df.iloc[-1]
            price, rsi_val, vol = float(last["Close"]), float(last["RSI"]), float(last["ATR"])
            sma20, sma50 = float(last["SMA20"]), float(last["SMA50"])
            sma200 = float(last["SMA200"]) if pd.notna(last["SMA200"]) else 0

            score = 0
            if price > sma50: score += 2
            if price > sma20 and sma20 > sma50: score += 2
            if rsi_val < 40: score += 1
            if sma200 and price > sma200: score += 1
            verdict = "BUY" if score >= 4 else "SELL" if score <= -2 else "HOLD"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"${price:.2f}")
            col2.metric("RSI", f"{rsi_val:.0f}")
            col3.metric("20d / 50d SMA", f"${sma20:.0f} / ${sma50:.0f}")
            col4.metric("ATR", f"${vol:.2f}")

            badge_class = "badge-buy" if verdict == "BUY" else "badge-sell" if verdict == "SELL" else "badge-hold"
            st.markdown(f'<div style="text-align:center; margin: 20px 0;"><span class="badge {badge_class}">{verdict}</span></div>', unsafe_allow_html=True)

            st.markdown("### 🧠 Trade Rationale")
            reasons = []
            if price > sma50: reasons.append(f"Price is above the 50-day SMA (${sma50:.2f}), indicating a medium-term uptrend.")
            else: reasons.append(f"Price is below the 50-day SMA (${sma50:.2f}), indicating a medium-term downtrend.")
            if sma200 and price > sma200: reasons.append(f"Price is above the 200-day SMA (${sma200:.2f}), confirming a long-term uptrend.")
            elif sma200: reasons.append(f"Price is below the 200-day SMA (${sma200:.2f}), signaling long-term weakness.")
            if rsi_val < 40: reasons.append(f"RSI ({rsi_val:.0f}) is not overbought, leaving room for upside.")
            for reason in reasons:
                st.markdown(f"- {reason}")

            st.markdown("### 🎲 Probability & Expected Value (10,000 Simulations, 4-Day Hold)")
            mc_result = monte_carlo(swing_symbol, "swing")
            if mc_result:
                c1, c2, c3 = st.columns(3)
                c1.metric("🎯 Win Rate", f"{mc_result['win_rate']*100:.1f}%")
                c2.metric("🛑 Loss Rate", f"{mc_result['loss_rate']*100:.1f}%")
                ev = mc_result['expected_value']
                c3.metric("📊 Expected Value", f"${ev:+.2f}")

                if ev > 0:
                    st.success(f"✅ POSITIVE EDGE: On average, this trade would have made **+${ev:.2f}** per share. The math is on your side.")
                elif ev > -0.05:
                    st.warning(f"⚖️ BREAK-EVEN: Expected value is **${ev:+.2f}**. Skipping is the smarter play.")
                else:
                    st.error(f"❌ NO EDGE: On average, this trade would have lost **-${abs(ev):.2f}** per share. The math is against you.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fund:
    st.header("Fundamentals · Long-Term Quality")
    st.caption("Scans S&P 500 for revenue growth, earnings growth, and analyst targets.")

    if st.button("🔍 Run Fundamental Scan", type="primary", key="fund_scan"):
        with st.spinner("Pulling fundamentals for 500 stocks — this may take 60 seconds..."):
            tickers = get_sp500_tickers()
            results = []

            def get_fundamentals_data(ticker):
                try:
                    info = yf.Ticker(ticker).info
                    return {
                        "Symbol": ticker,
                        "Name": (info.get("shortName") or ticker)[:18],
                        "Price": info.get("currentPrice") or 0,
                        "P/E": info.get("trailingPE"),
                        "Rev Growth": info.get("revenueGrowth"),
                        "Earn Growth": info.get("earningsGrowth"),
                        "Margin": info.get("profitMargins"),
                        "Analyst Target": info.get("targetMeanPrice"),
                        "Recommendation": info.get("recommendationKey", "N/A"),
                        "Market Cap": info.get("marketCap") or 0
                    }
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(get_fundamentals_data, ticker): ticker for ticker in tickers}
                for future in as_completed(futures):
                    data = future.result()
                    if data and data["Price"] > 0 and data["Market Cap"] > 1e9:
                        score = 0
                        if (data["Rev Growth"] or 0) > 0.15: score += 2
                        elif (data["Rev Growth"] or 0) > 0.05: score += 1
                        if (data["Earn Growth"] or 0) > 0.15: score += 2
                        elif (data["Earn Growth"] or 0) > 0.05: score += 1
                        if (data["Margin"] or 0) > 0.15: score += 1
                        if data["Analyst Target"] and data["Price"] and data["Analyst Target"] > data["Price"] * 1.15: score += 2
                        if data["Recommendation"] in ("buy", "strong_buy"): score += 1
                        if data["P/E"] and 5 < data["P/E"] < 40: score += 1
                        data["Score"] = score
                        if score >= 3:
                            results.append(data)

            if not results:
                st.warning("No stocks passed the fundamental screen.")
            else:
                results.sort(key=lambda x: x["Score"], reverse=True)
                top_results = results[:30]
                st.success(f"✅ Found {len(results)} fundamentally strong stocks. Showing top {len(top_results)}.")
                df_results = pd.DataFrame(top_results)
                df_results["Upside %"] = ((df_results["Analyst Target"].fillna(0) - df_results["Price"]) / df_results["Price"].replace(0, 1) * 100).round(1)
                df_results["Rev Growth %"] = (df_results["Rev Growth"].fillna(0) * 100).round(1)
                df_results["Earn Growth %"] = (df_results["Earn Growth"].fillna(0) * 100).round(1)
                st.dataframe(df_results[["Symbol", "Name", "Price", "Score", "Rev Growth %", "Earn Growth %", "P/E", "Upside %", "Recommendation"]], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔍 Deep Dive on a Single Ticker")
    fund_symbol = st.text_input("Enter ticker", "NVDA", key="fund_input").upper()

    if st.button("Analyze Fundamentals", key="fund_btn", type="primary"):
        info = yf.Ticker(fund_symbol).info
        if not info:
            st.error(f"No fundamental data for {fund_symbol}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${info.get('currentPrice') or 0:.2f}")
            c2.metric("P/E Ratio", f"{info.get('trailingPE') or 0:.1f}")
            c3.metric("Revenue Growth YoY", f"{(info.get('revenueGrowth') or 0)*100:.1f}%")
            c4.metric("Earnings Growth YoY", f"{(info.get('earningsGrowth') or 0)*100:.1f}%")

            if info.get("targetMeanPrice") and info.get("currentPrice"):
                upside = (info["targetMeanPrice"] - info["currentPrice"]) / info["currentPrice"] * 100
                st.metric("Analyst Consensus Target", f"${info['targetMeanPrice']:.2f}", f"{upside:+.1f}% Upside")

            st.markdown("### 🧠 What the Numbers Say")
            reasons = []
            rev_growth = info.get("revenueGrowth") or 0
            earn_growth = info.get("earningsGrowth") or 0
            margin = info.get("profitMargins") or 0
            target = info.get("targetMeanPrice") or 0
            price = info.get("currentPrice") or 0

            if rev_growth > 0.15: reasons.append(f"Revenue is up **{rev_growth*100:.0f}%** year-over-year — the business is growing fast.")
            elif rev_growth < 0: reasons.append(f"Revenue is **down {abs(rev_growth)*100:.0f}%** year-over-year — the business is shrinking.")
            if earn_growth > 0.15: reasons.append(f"Earnings are up **{earn_growth*100:.0f}%** year-over-year — profits are expanding.")
            elif earn_growth < -0.05: reasons.append(f"Earnings are **down {abs(earn_growth)*100:.0f}%** year-over-year — profitability is falling.")
            if margin > 0.20: reasons.append(f"Net profit margin is **{margin*100:.0f}%** — strong pricing power.")
            if target and price:
                if upside > 15: reasons.append(f"Analysts see **{upside:.0f}% upside** (average target ${target:.2f}).")
                elif upside < -10: reasons.append(f"Analysts see **{abs(upside):.0f}% downside** (average target ${target:.2f}).")
            for reason in reasons:
                st.markdown(f"- {reason}")

            st.caption("⚠️ These are facts about the business today. They do not predict tomorrow's stock price.")
