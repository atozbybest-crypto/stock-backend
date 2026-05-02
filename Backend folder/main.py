from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime, time
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from io import StringIO
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IST = ZoneInfo("Asia/Kolkata")
NIFTY500_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
TECHNIQUES = ['Price above 5 EMA', 'Price above 10 EMA', 'Price above 20 SMA', 'Price above 50 SMA', 'Price above 200 SMA', '5 EMA vs 10 EMA', '10 EMA vs 20 EMA', '20 SMA vs 50 SMA', '50 SMA vs 200 SMA', 'Slope of 20 SMA', 'Slope of 50 SMA', 'Higher highs', 'Higher lows', 'Lower highs', 'Lower lows', 'Day return positive', '3-day return positive', '5-day return positive', '10-day return positive', '20-day return positive', 'RSI 14', 'RSI 9', 'MACD line above signal', 'MACD above zero', 'Stochastic %K above %D', 'Bollinger upper touch', 'Bollinger lower touch', 'Bollinger squeeze', 'ATR rising', 'ATR falling', 'Volatility below average', 'Volatility above average', 'Volume above 20-day avg', 'Volume spike', 'On-balance volume rising', 'OBV falling', 'Price-volume confirmation', 'VWAP above price', 'VWAP below price', 'Support bounce', 'Resistance break', 'Gap up', 'Gap down', 'Close near high', 'Close near low', 'Candlestick bullish', 'Candlestick bearish', 'Doji presence', 'Hammer pattern', 'Shooting star', 'ADX strong trend', 'ADX weak trend', 'Directional +DI above -DI', 'Directional -DI above +DI', 'Momentum 3', 'Momentum 7', 'Momentum 14', 'ROC positive', 'ROC negative', '52-week high proximity', '52-week low proximity']

def load_nifty500_symbols():
    try:
        r = requests.get(NIFTY500_CSV, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        symbol_col = None
        for c in df.columns:
            if c.strip().lower() == "symbol":
                symbol_col = c
                break
        if symbol_col is None:
            return ["SBIN.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        syms = df[symbol_col].astype(str).str.strip().tolist()
        syms = [s for s in syms if s and s.lower() != "nan"]
        syms = [s + ".NS" if not s.endswith(".NS") else s for s in syms]
        return sorted(list(dict.fromkeys(syms)))
    except Exception:
        return ["SBIN.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

def get_history(symbol, interval, period):
    t = yf.Ticker(symbol)
    hist = t.history(interval=interval, period=period, auto_adjust=False, prepost=False)
    if hist is None or hist.empty:
        hist = t.history(interval="1d", period="1mo", auto_adjust=False, prepost=False)
    return hist.dropna()

def safe(series):
    return pd.Series(series).dropna()

def compute_techniques(df):
    close = safe(df["Close"])
    high = safe(df["High"]) if "High" in df else close
    low = safe(df["Low"]) if "Low" in df else close
    vol = safe(df["Volume"]) if "Volume" in df else pd.Series(dtype=float)
    n = len(close)
    out = []

    def add(name, signal, score, note):
        out.append({"name": name, "signal": signal, "score": round(float(score), 1), "note": note})

    if n < 5:
        for t in TECHNIQUES:
            add(t, "Neutral", 50, "Not enough data")
        return out

    ema5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
    ema10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1] if n >= 20 else close.mean()
    sma50 = close.rolling(50).mean().iloc[-1] if n >= 50 else close.mean()
    sma200 = close.rolling(200).mean().iloc[-1] if n >= 200 else close.mean()
    r = close.pct_change().dropna()
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1] if n >= 10 else close.mean()
    ma20 = sma20
    ma50 = sma50
    ma200 = sma200

    rsi14 = None
    if n >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss and loss > 0 else 999
        rsi14 = 100 - (100 / (1 + rs))

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_v = macd.iloc[-1]
    macd_s = macd_sig.iloc[-1]

    std20 = close.rolling(20).std().iloc[-1] if n >= 20 else r.std()
    upper = ma20 + 2 * std20 if pd.notna(std20) else close.max()
    lower = ma20 - 2 * std20 if pd.notna(std20) else close.min()
    atr = (high - low).rolling(14).mean().iloc[-1] if len(high) >= 14 and len(low) >= 14 else (high - low).mean()
    vol20 = vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else (vol.mean() if len(vol) else 0)

    last = close.iloc[-1]
    prev = close.iloc[-2]
    prev3 = close.iloc[-4] if n >= 4 else prev
    prev5 = close.iloc[-6] if n >= 6 else prev
    prev10 = close.iloc[-11] if n >= 11 else prev
    prev20 = close.iloc[-21] if n >= 21 else prev

    roc = (last / prev10 - 1) * 100 if prev10 else 0
    momentum3 = last - prev3
    momentum7 = last - (close.iloc[-8] if n >= 8 else prev)
    momentum14 = last - (close.iloc[-15] if n >= 15 else prev)
    supports = close.rolling(20).min().iloc[-1] if n >= 20 else close.min()
    resist = close.rolling(20).max().iloc[-1] if n >= 20 else close.max()
    near_high = last >= close.rolling(52).max().iloc[-1] * 0.95 if n >= 52 else last >= close.max() * 0.95
    near_low = last <= close.rolling(52).min().iloc[-1] * 1.05 if n >= 52 else last <= close.min() * 1.05
    bullish_candle = last > prev
    bearish_candle = last < prev
    doji = abs(last - prev) / prev < 0.002 if prev else False
    hammer = (last > prev) and ((last - low.iloc[-1]) > 2 * abs(last - prev)) if len(low) else False
    shooting = (last < prev) and ((high.iloc[-1] - last) > 2 * abs(last - prev)) if len(high) else False
    adx = abs((high.diff().abs().rolling(14).mean().iloc[-1] if len(high) >= 14 else 0) - (low.diff().abs().rolling(14).mean().iloc[-1] if len(low) >= 14 else 0))
    plus_di = (high.diff().clip(lower=0).rolling(14).mean().iloc[-1] if len(high) >= 14 else 0)
    minus_di = (-low.diff().clip(upper=0).rolling(14).mean().iloc[-1] if len(low) >= 14 else 0)

    for t in TECHNIQUES:
        if t == "Price above 5 EMA":
            add(t, "Bullish" if last > ema5 else "Bearish", 70 if last > ema5 else 30, "Latest close vs EMA5")
        elif t == "Price above 10 EMA":
            add(t, "Bullish" if last > ema10 else "Bearish", 68 if last > ema10 else 32, "Latest close vs EMA10")
        elif t == "Price above 20 SMA":
            add(t, "Bullish" if last > ma20 else "Bearish", 66 if last > ma20 else 34, "Latest close vs SMA20")
        elif t == "Price above 50 SMA":
            add(t, "Bullish" if last > ma50 else "Bearish", 64 if last > ma50 else 36, "Latest close vs SMA50")
        elif t == "Price above 200 SMA":
            add(t, "Bullish" if last > ma200 else "Bearish", 62 if last > ma200 else 38, "Latest close vs SMA200")
        elif t == "5 EMA vs 10 EMA":
            add(t, "Bullish" if ema5 > ema10 else "Bearish", 68 if ema5 > ema10 else 32, "EMA5 and EMA10 crossover state")
        elif t == "10 EMA vs 20 EMA":
            add(t, "Bullish" if ema10 > ema20 else "Bearish", 66 if ema10 > ema20 else 34, "EMA10 vs EMA20")
        elif t == "20 SMA vs 50 SMA":
            add(t, "Bullish" if ma20 > ma50 else "Bearish", 65 if ma20 > ma50 else 35, "SMA20 vs SMA50")
        elif t == "50 SMA vs 200 SMA":
            add(t, "Bullish" if ma50 > ma200 else "Bearish", 63 if ma50 > ma200 else 37, "SMA50 vs SMA200")
        elif t == "Slope of 20 SMA":
            v = ma20 > close.rolling(20).mean().shift(5).iloc[-1] if n >= 25 else False
            add(t, "Bullish" if v else "Bearish", 62 if v else 50, "20-SMA recent slope")
        elif t == "Slope of 50 SMA":
            v = ma50 > close.rolling(50).mean().shift(5).iloc[-1] if n >= 55 else False
            add(t, "Bullish" if v else "Bearish", 61 if v else 50, "50-SMA recent slope")
        elif t == "Higher highs":
            v = high.iloc[-1] >= high.tail(5).max()
            add(t, "Bullish" if v else "Neutral", 60 if v else 40, "Recent highs pattern")
        elif t == "Higher lows":
            v = low.iloc[-1] >= low.tail(5).min()
            add(t, "Bullish" if v else "Neutral", 60 if v else 40, "Recent lows pattern")
        elif t == "Lower highs":
            v = high.iloc[-1] < high.tail(5).max()
            add(t, "Bearish" if v else "Neutral", 40 if v else 60, "Recent highs pattern")
        elif t == "Lower lows":
            v = low.iloc[-1] < low.tail(5).min()
            add(t, "Bearish" if v else "Neutral", 40 if v else 60, "Recent lows pattern")
        elif t == "Day return positive":
            v = last > prev
            add(t, "Bullish" if v else "Bearish", 62 if v else 38, "Latest day return")
        elif t == "3-day return positive":
            v = last > prev3
            add(t, "Bullish" if v else "Bearish", 61 if v else 39, "3-day change")
        elif t == "5-day return positive":
            v = last > prev5
            add(t, "Bullish" if v else "Bearish", 60 if v else 40, "5-day change")
        elif t == "10-day return positive":
            v = last > prev10
            add(t, "Bullish" if v else "Bearish", 59 if v else 41, "10-day change")
        elif t == "20-day return positive":
            v = last > prev20
            add(t, "Bullish" if v else "Bearish", 58 if v else 42, "20-day change")
        elif t == "RSI 14":
            score = 70 - abs((rsi14 or 50) - 50)
            add(t, "Bullish" if (rsi14 or 50) < 70 else "Bearish", score, "RSI14")
        elif t == "RSI 9":
            add(t, "Neutral", 50, "Placeholder RSI9")
        elif t == "MACD line above signal":
            v = macd_v > macd_s
            add(t, "Bullish" if v else "Bearish", 67 if v else 33, "MACD crossover")
        elif t == "MACD above zero":
            v = macd_v > 0
            add(t, "Bullish" if v else "Bearish", 66 if v else 34, "MACD level")
        elif t == "Stochastic %K above %D":
            v = last > prev
            add(t, "Bullish" if v else "Bearish", 60 if v else 40, "Stochastic proxy")
        elif t == "Bollinger upper touch":
            v = last >= upper
            add(t, "Neutral", 55 if v else 45, "Upper band touch")
        elif t == "Bollinger lower touch":
            v = last <= lower
            add(t, "Neutral", 55 if v else 45, "Lower band touch")
        elif t == "Bollinger squeeze":
            v = (upper - lower) / last < 0.08 if last else False
            add(t, "Neutral", 55 if v else 45, "Band width")
        elif t == "ATR rising":
            v = atr > (high - low).rolling(14).mean().shift(5).iloc[-1] if len(high) >= 19 else False
            add(t, "Neutral", 55 if v else 50, "ATR trend")
        elif t == "ATR falling":
            v = atr < (high - low).rolling(14).mean().shift(5).iloc[-1] if len(high) >= 19 else False
            add(t, "Neutral", 55 if v else 50, "ATR trend")
        elif t == "Volatility below average":
            v = r.std() < r.rolling(20).std().mean() if len(r) >= 20 else False
            add(t, "Bullish" if v else "Neutral", 58 if v else 50, "Return volatility")
        elif t == "Volatility above average":
            v = r.std() > r.rolling(20).std().mean() if len(r) >= 20 else False
            add(t, "Bearish" if v else "Neutral", 58 if v else 50, "Return volatility")
        elif t == "Volume above 20-day avg":
            v = len(vol) and vol.iloc[-1] > vol20
            add(t, "Bullish" if v else "Neutral", 60 if v else 40, "Volume vs avg")
        elif t == "Volume spike":
            v = len(vol) and vol.iloc[-1] > (vol20 * 1.5 if vol20 else vol.iloc[-1])
            add(t, "Bullish" if v else "Neutral", 58 if v else 42, "Volume spike")
        elif t == "On-balance volume rising":
            v = len(vol) and vol.tail(5).sum() > vol.tail(10).head(5).sum() if len(vol) >= 10 else False
            add(t, "Bullish" if v else "Neutral", 57 if v else 50, "OBV proxy")
        elif t == "OBV falling":
            v = len(vol) and vol.tail(5).sum() < vol.tail(10).head(5).sum() if len(vol) >= 10 else False
            add(t, "Bearish" if v else "Neutral", 57 if v else 50, "OBV proxy")
        elif t == "Price-volume confirmation":
            v = (last > prev) and (len(vol) and vol.iloc[-1] > vol20)
            add(t, "Bullish" if v else "Bearish", 61 if v else 39, "Price with volume")
        elif t == "VWAP above price":
            v = last < close.mean()
            add(t, "Bearish" if v else "Neutral", 54 if v else 46, "VWAP proxy")
        elif t == "VWAP below price":
            v = last > close.mean()
            add(t, "Bullish" if v else "Neutral", 54 if v else 46, "VWAP proxy")
        elif t == "Support bounce":
            v = last > supports
            add(t, "Bullish" if v else "Bearish", 60 if v else 40, "Support zone")
        elif t == "Resistance break":
            v = last > resist
            add(t, "Bullish" if v else "Neutral", 60 if v else 40, "Resistance zone")
        elif t == "Gap up":
            v = last > prev * 1.01
            add(t, "Bullish" if v else "Neutral", 56 if v else 44, "Gap proxy")
        elif t == "Gap down":
            v = last < prev * 0.99
            add(t, "Bearish" if v else "Neutral", 56 if v else 44, "Gap proxy")
        elif t == "Close near high":
            v = last >= close.tail(20).max() * 0.97
            add(t, "Bullish" if v else "Neutral", 57 if v else 43, "Near recent high")
        elif t == "Close near low":
            v = last <= close.tail(20).min() * 1.03
            add(t, "Bearish" if v else "Neutral", 57 if v else 43, "Near recent low")
        elif t == "Candlestick bullish":
            v = bullish_candle
            add(t, "Bullish" if v else "Neutral", 55 if v else 45, "Bullish candle proxy")
        elif t == "Candlestick bearish":
            v = bearish_candle
            add(t, "Bearish" if v else "Neutral", 55 if v else 45, "Bearish candle proxy")
        elif t == "Doji presence":
            v = doji
            add(t, "Neutral", 52 if v else 48, "Doji proxy")
        elif t == "Hammer pattern":
            v = hammer
            add(t, "Bullish" if v else "Neutral", 58 if v else 42, "Hammer proxy")
        elif t == "Shooting star":
            v = shooting
            add(t, "Bearish" if v else "Neutral", 58 if v else 42, "Shooting star proxy")
        elif t == "ADX strong trend":
            v = adx > 1
            add(t, "Bullish" if v else "Neutral", 60 if v else 40, "ADX proxy")
        elif t == "ADX weak trend":
            v = adx <= 1
            add(t, "Neutral", 60 if v else 40, "ADX proxy")
        elif t == "Directional +DI above -DI":
            v = plus_di > minus_di
            add(t, "Bullish" if v else "Bearish", 61 if v else 39, "+DI vs -DI")
        elif t == "Directional -DI above +DI":
            v = minus_di > plus_di
            add(t, "Bearish" if v else "Bullish", 61 if v else 39, "-DI vs +DI")
        elif t == "Momentum 3":
            v = momentum3 > 0
            add(t, "Bullish" if v else "Bearish", 59 if v else 41, "3-period momentum")
        elif t == "Momentum 7":
            v = momentum7 > 0
            add(t, "Bullish" if v else "Bearish", 59 if v else 41, "7-period momentum")
        elif t == "Momentum 14":
            v = momentum14 > 0
            add(t, "Bullish" if v else "Bearish", 59 if v else 41, "14-period momentum")
        elif t == "ROC positive":
            v = roc > 0
            add(t, "Bullish" if v else "Bearish", 60 if v else 40, "Rate of change")
        elif t == "ROC negative":
            v = roc < 0
            add(t, "Bearish" if v else "Bullish", 60 if v else 40, "Rate of change")
        elif t == "52-week high proximity":
            v = near_high
            add(t, "Bullish" if v else "Neutral", 58 if v else 42, "52-week high proximity")
        elif t == "52-week low proximity":
            v = near_low
            add(t, "Bearish" if v else "Neutral", 58 if v else 42, "52-week low proximity")
        else:
            add(t, "Neutral", 50, "Generic")
    return out

@app.get("/")
def home():
    return {"status": "ok", "message": "Stock analyser backend running"}

@app.get("/api/yahoo/list")
def ticker_list():
    syms = load_nifty500_symbols()
    return {"count": len(syms), "symbols": syms, "sample": syms[:20]}

@app.get("/api/yahoo/quote")
def quote(symbol: str = Query(...)):
    now = datetime.now(IST)
    is_open = now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)
    interval = "30m" if is_open else "1d"
    period = "5d" if is_open else "1mo"
    hist = get_history(symbol, interval, period)
    if hist.empty:
        return {"symbol": symbol.upper(), "error": "No data"}
    close = float(hist.iloc[-1]["Close"])
    return {
        "symbol": symbol.upper(),
        "price": close,
        "previousClose": float(hist.iloc[-2]["Close"]) if len(hist) > 1 else close,
        "market_status": "open" if is_open else "closed",
        "timestamp": str(hist.index[-1]),
        "source": f"yfinance {interval}"
    }

@app.get("/api/yahoo/analyse")
def analyse(symbol: str = Query(...)):
    now = datetime.now(IST)
    is_open = now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)
    interval = "30m" if is_open else "1d"
    period = "5d" if is_open else "6mo"
    hist = get_history(symbol, interval, period)
    if hist.empty:
        return {"symbol": symbol.upper(), "error": "No data"}
    close = hist["Close"].astype(float).tolist()
    techs = compute_techniques(hist)
    scores = [t['score'] for t in techs]
    overall = round(float(np.mean(scores)), 1)
    short = round(float(np.mean(scores[:20])), 1)
    long = round(float(np.mean(scores[20:40])), 1)
    risk = round(float(np.mean(scores[40:])), 1)
    signal = "Bullish" if overall >= 70 else "Positive" if overall >= 55 else "Neutral" if overall >= 42 else "Cautious"
    verdict_reason = "Strong technical mix with broad confirmation." if signal == "Bullish" else "Mixed signals across trend, momentum, and risk." if signal == "Neutral" else "Weak trend or elevated risk across multiple techniques."
    return {
        "symbol": symbol.upper(),
        "name": symbol.upper(),
        "short": short,
        "long": long,
        "fund": 50,
        "risk": risk,
        "overall": overall,
        "signal": signal,
        "techniques": techs,
        "technique_count": len(techs),
        "verdict_reason": verdict_reason,
        "market_status": "open" if is_open else "closed",
        "close_used": close[-1],
        "source": f"yfinance {interval}"
    }

@app.get("/api/yahoo/chart")
def chart(symbol: str = Query(...), period: str = Query("1mo"), interval: str = Query("1d")):
    hist = get_history(symbol, interval, period)
    if hist.empty:
        return {"symbol": symbol.upper(), "error": "No data"}
    hist = hist.reset_index()
    time_col = hist.columns[0]
    out = hist[[time_col, "Close"]].copy()
    out.columns = ["time", "close"]
    out["time"] = out["time"].astype(str)
    return {"symbol": symbol.upper(), "points": out.tail(200).to_dict(orient="records")}