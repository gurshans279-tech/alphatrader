import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from datetime import datetime
from urllib.parse import quote

st.set_page_config(page_title="AlphaTrader", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#0a0e1a 0%,#141b2d 100%); }
h1,h2,h3 { font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',sans-serif; letter-spacing:-0.5px; }
[data-testid="stMetricValue"] { font-family:'SF Mono',Menlo,monospace; font-weight:700; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0a0e1a 0%,#1a2332 100%); border-right:1px solid #2a3548; }
.card { background:#141b2d; border:1px solid #2a3548; border-radius:12px; padding:20px; margin:8px 0; }
.card-buy { border-left:4px solid #00d97e; box-shadow:0 0 20px rgba(0,217,126,0.15); }
.card-sell { border-left:4px solid #ff3860; box-shadow:0 0 20px rgba(255,56,96,0.15); }
.card-wait { border-left:4px solid #ffb800; box-shadow:0 0 20px rgba(255,184,0,0.15); }
.action-badge { display:inline-block; padding:8px 20px; border-radius:24px; font-weight:800; font-size:20px; letter-spacing:1px; }
.badge-buy { background:#00d97e; color:#0a0e1a; }
.badge-sell { background:#ff3860; color:white; }
.badge-wait { background:#ffb800; color:#0a0e1a; }
.badge-hold { background:#4a5568; color:white; }
.tooltip { border-bottom:1px dotted #4a5568; cursor:help; }
.num { font-family:'SF Mono',Menlo,monospace; }
</style>
""", unsafe_allow_html=True)

DAY_UNIV = ["AAPL","NVDA","TSLA","AMD","MSFT","AMZN","META","GOOGL","NFLX","AVGO",
            "COIN","PLTR","SOFI","MARA","HOOD","SNAP","UBER","SHOP","INTC","MU",
            "SPY","QQQ","IWM","JPM","BAC","XOM","CVX","BA","DIS","LLY"]
SWING_UNIV = ["AAPL","NVDA","TSLA","MSFT","AMZN","META","GOOGL","JPM","V","UNH",
              "XOM","JNJ","WMT","PG","HD","KO","PEP","COST","MCD","DIS","BA",
              "CAT","DE","LMT","RTX","T","VZ","MRK","ABBV","LLY"]

def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-(100/(1+g/l))

def atr(df,n=14):
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift()).abs(),
                  (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def vwap(df):
    tp=(df["High"]+df["Low"]+df["Close"])/3
    return (tp*df["Volume"]).cumsum()/df["Volume"].cumsum()

@st.cache_data(ttl=60)
def load(t,p,i):
    try:
        df=yf.download(t,period=p,interval=i,progress=False,auto_adjust=True)
        if isinstance(df.columns,pd.MultiIndex):
            df.columns=df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_news(t,n=60):
    out,seen=[],set()
    queries=[f"{t} stock",f"{t} earnings",f"{t} analyst",f"{t} upgrade",
             f"{t} downgrade",f"{t} price target",f"{t} news",f"{t} forecast",
             f"{t} shares",f"{t} valuation",f"{t} Q4",f"{t} results"]
    for q in queries:
        try:
            url=f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
            feed=feedparser.parse(url,request_headers={"User-Agent":"Mozilla/5.0"})
            for e in feed.entries[:10]:
                title=e.title
                pub=title.rsplit(" - ",1)[-1] if " - " in title else "Google News"
                key=title.lower()[:45]
                if key and key not in seen:
                    seen.add(key); out.append({"source":pub,"title":title})
        except Exception: pass
    try:
        for i in (yf.Ticker(t).news or [])[:30]:
            title=i.get("title","")
            key=title.lower()[:45]
            if key and key not in seen:
                seen.add(key); out.append({"source":i.get("publisher","Yahoo"),"title":title})
    except Exception: pass
    return out[:n]

def monte_carlo(t,n_sims=10000,horizon="day"):
    df=load(t,"60d","5m")
    if df.empty or len(df)<100:
        df=load(t,"30d","1h")
    if df.empty or len(df)<100:
        df=load(t,"5d","5m")
    if df.empty or len(df)<50:
        return None
    rets=df["Close"].pct_change().dropna().values
    last=float(df["Close"].iloc[-1]); a=float(atr(df).iloc[-1])
    if horizon=="day":
        target,stop=last+3*a,last-1.5*a
        bars=78
    else:
        target,stop=last+4*a,last-2*a
        bars=4
    rng=np.random.default_rng(42); ht=hs=hn=0
    for _ in range(n_sims):
        path=last*np.cumprod(1+rng.choice(rets,bars))
        hit=False
        for p in path:
            if p>=target: ht+=1;hit=True;break
            if p<=stop: hs+=1;hit=True;break
        if not hit: hn+=1
    return {"p_target":ht/n_sims,"p_stop":hs/n_sims,"p_neither":hn/n_sims,
            "target":target,"stop":stop,"n":n_sims,"bars":len(df)}

def day_setup(t):
    df=load(t,"5d","5m")
    if df.empty or len(df)<15: return None
    df["RSI"]=rsi(df["Close"]); df["VWAP"]=vwap(df); df["ATR"]=atr(df)
    today=df[df.index.date==df.index.date.max()]
    if len(today)<3: today=df.tail(20)
    orb=today.head(3); oh,ol=orb["High"].max(),orb["Low"].min(); om=(oh+ol)/2
    last=today.iloc[-1]
    p,a,r,v=float(last["Close"]),float(last["ATR"]),float(last["RSI"]),float(last["VWAP"])
    strat,stop,target,sc=None,0,0,0
    if p>oh and today.iloc[-2]["Close"]<=oh:
        strat,stop,target,sc="ORB Breakout",om,p+(p-om)*2,3.0
    elif df.iloc[-2]["Close"]<df.iloc[-2]["VWAP"] and p>v:
        strat,stop,target,sc="VWAP Reclaim",v*0.995,p+(p-v)*2,2.5
    elif 50<r<70 and p>v:
        strat,stop,target,sc="Momentum",p-1.5*a,p+3*a,1.5
    elif p>oh*0.995 and p<oh:
        strat,stop,target,sc="Near ORB Break",om,p+(p-om)*2,1.0
    elif r<35 and p>v*0.998:
        strat,stop,target,sc="Oversold Bounce",p-1.5*a,p+2.5*a,1.0
    if not strat: return None
    return {"sym":t,"strat":strat,"price":p,"stop":stop,"target":target,"score":sc}

def swing_setup(t):
    df=load(t,"1y","1d")
    if df.empty or len(df)<60: return None
    df["RSI"]=rsi(df["Close"])
    df["SMA20"]=df["Close"].rolling(20).mean()
    df["SMA50"]=df["Close"].rolling(50).mean()
    df["SMA200"]=df["Close"].rolling(200).mean()
    df["ATR"]=atr(df)
    last=df.iloc[-1]
    p,r,a=float(last["Close"]),float(last["RSI"]),float(last["ATR"])
    s20,s50=float(last["SMA20"]),float(last["SMA50"])
    s200=float(last["SMA200"]) if pd.notna(last["SMA200"]) else 0
    avgvol=df["Volume"].tail(20).mean()
    strat,stop,target,sc=None,0,0,0
    if s200>0 and p>s200 and abs(p-s20)/s20<0.02:
        strat,stop,target,sc="Pullback to 20-day",s20*0.96,p+(p-s20*0.96)*3,3.0
    elif p>df["High"].iloc[-21:-1].max() and df["Volume"].iloc[-1]>avgvol*1.3:
        strat,stop,target,sc="20-Day Breakout",p-2*a,p+4*a,2.5
    elif r<35:
        strat,stop,target,sc="Oversold Bounce",p-2.5*a,p+4*a,1.5
    elif p>s20 and s20>s50 and r<60:
        strat,stop,target,sc="Uptrend Continuation",s20*0.97,p+3*a,1.0
    if not strat: return None
    return {"sym":t,"strat":strat,"price":p,"stop":stop,"target":target,"score":sc}

def get_action(verdict,price,buy_zone,rsi,above_vwap):
    if verdict=="BUY":
        if above_vwap and rsi<40: return ("BUY NOW","badge-buy","card-buy","Signals aligned — enter at market")
        if price>buy_zone*1.02: return (f"WAIT FOR ${buy_zone:.2f}","badge-wait","card-wait",f"Wait for pullback to ${buy_zone:.2f} before entering")
        return ("BUY NOW","badge-buy","card-buy","Setup valid at current price")
    if verdict=="SELL":
        return ("SELL NOW","badge-sell","card-sell","Signals bearish — exit or avoid")
    return ("HOLD / WAIT","badge-hold","card-wait","Mixed signals — no edge, wait for clarity")

st.sidebar.markdown("## ⚡ AlphaTrader")
st.sidebar.caption("Not financial advice")
account=st.sidebar.number_input("Account ($)",1000,10_000_000,25_000,step=1000)
risk_pct=st.sidebar.slider("Risk per trade (%)",0.25,3.0,1.0,0.25)

with st.sidebar.expander("📖 Glossary"):
    st.markdown("""
**RSI** — Momentum 0-100. <30 oversold, >70 overbought.

**SMA 20** — Average closing price over 20 days. Shows 1-month trend.

**SMA 50** — Average over 50 days. Shows 2-3 month trend.

**SMA 200** — Average over 200 days. Shows 1-year trend.

**ATR** — Average True Range: how much price typically moves in a day. Used for sizing stops.

**VWAP** — Volume-weighted average price. The average price traders actually paid today.
""")

tab1,tab2=st.tabs(["⚡ Day Trading","📈 Swing (2-4 Days)"])

with tab1:
    st.markdown("## ⚡ Day Trading")
    st.caption(f"Intraday · must exit by 15:55 ET · {datetime.now():%H:%M}")
    if st.button("🔍 SCAN 30 STOCKS NOW",type="primary",use_container_width=True,key="dscan"):
        bar=st.progress(0,text="Scanning…")
        results=[]
        for i,s in enumerate(DAY_UNIV):
            r=day_setup(s)
            if r: results.append(r)
            bar.progress((i+1)/len(DAY_UNIV),text=f"Scanned {i+1}/{len(DAY_UNIV)} — found {len(results)}")
        results.sort(key=lambda x:x["score"],reverse=True)
        st.session_state["dres"]=results
        bar.empty()
    if "dres" in st.session_state:
        res=st.session_state["dres"]
        if not res:
            st.warning("Scanned 30 stocks — 0 setups right now. Conditions (breakout/VWAP reclaim) not met. Try Swing tab or check back during market hours.")
        else:
            st.success(f"✅ {len(res)} setups found")
            rows=[]
            for r in res:
                rr=(r["target"]-r["price"])/(r["price"]-r["stop"]) if r["price"]>r["stop"] else 0
                rows.append({"Symbol":r["sym"],"Strategy":r["strat"],"Price":f"${r['price']:.2f}",
                             "Stop":f"${r['stop']:.2f}","Target":f"${r['target']:.2f}",
                             "R:R":f"{rr:.1f}","Score":r["score"]})
            st.dataframe(rows,use_container_width=True,hide_index=True)
    st.divider()
    d_sym=st.text_input("Analyze ticker","TSLA",key="dsym").upper()
    if st.button("Analyze",key="dan",use_container_width=True):
        df=load(d_sym,"5d","5m")
        if df.empty:
            st.error(f"No intraday data for {d_sym}")
        else:
            df["RSI"]=rsi(df["Close"]); df["VWAP"]=vwap(df); df["ATR"]=atr(df)
            last=df.iloc[-1]
            p,a,r,v=float(last["Close"]),float(last["ATR"]),float(last["RSI"]),float(last["VWAP"])
            sc=(2 if r<30 else -2 if r>70 else 0)+(2 if p>v else -2)
            verdict="BUY" if sc>=3 else "SELL" if sc<=-3 else "HOLD"
            buy_zone,stop,target,trail=p-1.5*a,p-3.5*a,p+3*a,p-2*a
            res=float(df["High"].tail(20).max())
            act,badge,card,reason=get_action(verdict,p,buy_zone,r,p>v)
            st.markdown(f'<div class="card {card}"><span class="action-badge {badge}">{act}</span><div style="margin-top:12px;font-size:16px;color:#b8c5d6">{reason}</div></div>',unsafe_allow_html=True)
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Price",f"${p:.2f}")
            c2.metric("RSI",f"{r:.1f}")
            c3.metric("VWAP",f"${v:.2f}")
            c4.metric("ATR",f"${a:.2f}")
            lc,rc=st.columns([2,3])
            with lc:
                st.markdown("### 🧠 Why")
                if r<30: st.write(f"• RSI {r:.0f} = **oversold** (below 30). Bounce likely.")
                elif r>70: st.write(f"• RSI {r:.0f} = **overbought** (above 70). Pullback risk.")
                else: st.write(f"• RSI {r:.0f} = neutral. No momentum extreme.")
                st.write(f"• Price {'**above**' if p>v else '**below**'} VWAP (${v:.2f}) — {'buyers in control' if p>v else 'sellers in control'}")
                st.markdown("### 💰 Price Levels")
                st.write(f"**Buy at** ${buy_zone:.2f}")
                st.write(f"**Stop** ${stop:.2f}")
                st.write(f"**Target** ${target:.2f}")
                st.write(f"**Trail** ${trail:.2f}")
                st.write(f"**Resistance** ${res:.2f}")
                rps=p-stop
                if rps>0:
                    q=int((account*risk_pct/100)/rps)
                    st.write(f"**{q} shares** · risk ${q*rps:.2f}")
            with rc:
                st.line_chart(df[["Close","VWAP"]].tail(78),height=280)
            st.divider()
            st.markdown("### 🎲 10,000-Simulation Odds")
            if st.button("Run Simulation",key="dtm",type="primary"):
                with st.spinner("Running 10,000 simulations…"):
                    tm=monte_carlo(d_sym,10000,"day")
                if tm:
                    st.caption(f"Based on {tm['bars']} historical bars")
                    m1,m2,m3=st.columns(3)
                    m1.metric("🎯 Hit Target",f"{tm['p_target']*100:.1f}%")
                    m2.metric("🛑 Hit Stop",f"{tm['p_stop']*100:.1f}%")
                    m3.metric("⏰ Neither",f"{tm['p_neither']*100:.1f}%")
                    edge=tm['p_target']-tm['p_stop']
                    if edge>0.05: st.success(f"✅ EDGE +{edge*100:.1f}% for target")
                    elif edge<-0.05: st.error(f"❌ NO EDGE — stop hits {abs(edge)*100:.1f}% more")
                    else: st.warning(f"⚖️ COIN FLIP ({edge*100:+.1f}%)")
                else: st.error("Not enough historical data")
            st.divider()
            news=fetch_news(d_sym)
            st.markdown(f"### 📰 {len(news)} Headlines")
            for n in news:
                st.write(f"**{n['source']}** — {n['title']}")

with tab2:
    st.markdown("## 📈 Swing Trading (2-4 Day Hold)")
    st.caption(f"Daily bars · exit by day 4 · {datetime.now():%H:%M}")
    if st.button("🔍 SCAN 30 STOCKS NOW",type="primary",use_container_width=True,key="sscan"):
        bar=st.progress(0,text="Scanning…")
        results=[]
        for i,s in enumerate(SWING_UNIV):
            r=swing_setup(s)
            if r: results.append(r)
            bar.progress((i+1)/len(SWING_UNIV),text=f"Scanned {i+1}/{len(SWING_UNIV)} — found {len(results)}")
        results.sort(key=lambda x:x["score"],reverse=True)
        st.session_state["sres"]=results
        bar.empty()
    if "sres" in st.session_state:
        res=st.session_state["sres"]
        if not res:
            st.warning("Scanned 30 stocks — 0 swing setups now. Check back or use a different universe.")
        else:
            st.success(f"✅ {len(res)} setups found")
            rows=[]
            for r in res:
                rr=(r["target"]-r["price"])/(r["price"]-r["stop"]) if r["price"]>r["stop"] else 0
                rows.append({"Symbol":r["sym"],"Strategy":r["strat"],"Price":f"${r['price']:.2f}",
                             "Stop":f"${r['stop']:.2f}","Target":f"${r['target']:.2f}",
                             "R:R":f"{rr:.1f}","Score":r["score"],"Hold":"2-4d"})
            st.dataframe(rows,use_container_width=True,hide_index=True)
    st.divider()
    s_sym=st.text_input("Analyze ticker","AAPL",key="ssym").upper()
    if st.button("Analyze",key="san",use_container_width=True):
        df=load(s_sym,"1y","1d")
        if df.empty:
            st.error(f"No daily data for {s_sym}")
        else:
            df["RSI"]=rsi(df["Close"])
            df["SMA20"]=df["Close"].rolling(20).mean()
            df["SMA50"]=df["Close"].rolling(50).mean()
            df["SMA200"]=df["Close"].rolling(200).mean()
            df["ATR"]=atr(df)
            last=df.iloc[-1]
            p,r,a=float(last["Close"]),float(last["RSI"]),float(last["ATR"])
            s20,s50=float(last["SMA20"]),float(last["SMA50"])
            s200=float(last["SMA200"]) if pd.notna(last["SMA200"]) else 0
            sc=(2 if r<40 else -2 if r>70 else 0)+(2 if p>s50 else -2)
            if s200 and p>s200: sc+=1
            if p>s20 and s20>s50: sc+=1
            verdict="BUY" if sc>=3 else "SELL" if sc<=-3 else "HOLD"
            buy_zone,stop,target,trail=p-1*a,p-2.5*a,p+4*a,p-2*a
            res=float(df["High"].tail(20).max())
            act,badge,card,reason=get_action(verdict,p,buy_zone,r,p>s50)
            st.markdown(f'<div class="card {card}"><span class="action-badge {badge}">{act}</span><div style="margin-top:12px;font-size:16px;color:#b8c5d6">{reason}</div></div>',unsafe_allow_html=True)
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Price",f"${p:.2f}")
            c2.metric("RSI",f"{r:.1f}")
            c3.metric("20/50 SMA",f"${s20:.0f}/${s50:.0f}")
            c4.metric("ATR",f"${a:.2f}")
            lc,rc=st.columns([2,3])
            with lc:
                st.markdown("### 🧠 Why")
                if r<40: st.write(f"• RSI {r:.0f} = not overbought, room to run")
                elif r>70: st.write(f"• RSI {r:.0f} = overbought, pullback risk")
                else: st.write(f"• RSI {r:.0f} = neutral")
                st.write(f"• Price {'**above**' if p>s50 else '**below**'} 50-day avg (${s50:.2f}) — {'medium-term uptrend' if p>s50 else 'medium-term downtrend'}")
                if s200: st.write(f"• Price {'**above**' if p>s200 else '**below**'} 200-day avg (${s200:.2f}) — {'long-term uptrend' if p>s200 else 'long-term downtrend'}")
                if p>s20 and s20>s50: st.write("• Moving averages **stacked up** (price > 20-day > 50-day) — bullish alignment")
                st.markdown("### 💰 Price Levels")
                st.write(f"**Buy at** ${buy_zone:.2f}")
                st.write(f"**Stop** ${stop:.2f}")
                st.write(f"**Target** ${target:.2f}")
                st.write(f"**Trail** ${trail:.2f}")
                st.write(f"**Resistance** ${res:.2f}")
                rps=p-stop
                if rps>0:
                    q=int((account*risk_pct/100)/rps)
                    st.write(f"**{q} shares** · risk ${q*rps:.2f}")
            with rc:
                st.line_chart(df[["Close","SMA20","SMA50"]].tail(120),height=320)
            st.divider()
            st.markdown("### 🎲 10,000-Simulation Odds (4-day hold)")
            if st.button("Run Simulation",key="stm",type="primary"):
                with st.spinner("Running 10,000 simulations…"):
                    tm=monte_carlo(s_sym,10000,"swing")
                if tm:
                    st.caption(f"Based on {tm['bars']} historical bars")
                    m1,m2,m3=st.columns(3)
                    m1.metric("🎯 Hit Target",f"{tm['p_target']*100:.1f}%")
                    m2.metric("🛑 Hit Stop",f"{tm['p_stop']*100:.1f}%")
                    m3.metric("⏰ Neither",f"{tm['p_neither']*100:.1f}%")
                    edge=tm['p_target']-tm['p_stop']
                    if edge>0.05: st.success(f"✅ EDGE +{edge*100:.1f}% for target")
                    elif edge<-0.05: st.error(f"❌ NO EDGE — stop hits {abs(edge)*100:.1f}% more")
                    else: st.warning(f"⚖️ COIN FLIP ({edge*100:+.1f}%)")
                else: st.error("Not enough historical data")
            st.divider()
            news=fetch_news(s_sym)
            st.markdown(f"### 📰 {len(news)} Headlines")
            for n in news:
                st.write(f"**{n['source']}** — {n['title']}")

st.divider()
st.caption("⚠️ Not financial advice. Paper trade first.")
