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

TECHNIQUES = [
    "Price above 5 EMA", "Price above 10 EMA", "Price above 20 SMA", "Price above 50 SMA", "Price above 200 SMA",
    "5 EMA vs 10 EMA", "10 EMA vs 20 EMA", "20 SMA vs 50 SMA", "50 SMA vs 200 SMA", "Slope of 20 SMA",
    "Slope of 50 SMA", "Higher highs", "Higher lows", "Lower highs", "Lower lows", "Day return positive",
    "3-day return positive", "5-day return positive", "10-day return positive", "20-day return positive",
    "RSI 14", "RSI 9", "MACD line above signal", "MACD above zero", "Stochastic %K above %D",
    "Bollinger upper touch", "Bollinger lower touch", "Bollinger squeeze", "ATR rising", "ATR falling",
    "Volatility below average", "Volatility above average", "Volume above 20-day avg", "Volume spike",
    "On-balance volume rising", "OBV falling", "Price-volume confirmation", "VWAP above price",
    "VWAP below price", "Support bounce", "Resistance break", "Gap up", "Gap down", "Close near high",
    "Close near low", "Candlestick bullish", "Candlestick bearish", "Doji presence", "Hammer pattern",
    "Shooting star", "ADX strong trend", "ADX weak trend", "Directional +DI above -DI",
    "Directional -DI above +DI", "Momentum 3", "Momentum 7", "Momentum 14", "ROC positive",
    "ROC negative", "52-week high proximity", "52-week low proximity"
]


def clean_number(value):
    try:
        if value is None:
            return None
        if isinstance(value, (np.floating, float)):
            if np.isnan(value) or np.isinf(value):
                return None
            return round(float(value), 2)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if pd.isna(value):
            return None
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                return None
            return value
        return round(float(value), 2) if isinstance(value, (int, float)) else value
    except Exception:
        return None


def format_percent(value):
    try:
        if value is None:
            return None
        v = float(value)
        if abs(v) <= 1:
            v *= 100
        return f"{round(v, 2)}%"
    except Exception:
        return None


def format_large_number(value):
    try:
        if value is None:
            return None
        v = float(value)
        if abs(v) >= 1e12:
            return f"{round(v / 1e12, 2)}T"
        if abs(v) >= 1e9:
            return f"{round(v / 1e9, 2)}B"
        if abs(v) >= 1e7:
            return f"{round(v / 1e7, 2)}Cr"
        if abs(v) >= 1e5:
            return f"{round(v / 1e5, 2)}L"
        return f"{round(v, 2)}"
    except Exception:
        return clean_number(value)


def safe_dict_get(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for key in keys:
        if key in d and d[key] not in [None, "", "N/A", "nan"]:
            return d[key]
    return default


def get_fast_info_dict(ticker):
    try:
        fi = ticker.fast_info
        if fi is None:
            return {}
        try:
            return dict(fi)
        except Exception:
            out = {}
            for k in dir(fi):
                if k.startswith("_"):
                    continue
                try:
                    val = getattr(fi, k)
                    if not callable(val):
                        out[k] = val
                except Exception:
                    pass
            return out
    except Exception:
        return {}


def merge_info(ticker):
    info = {}
    try:
        raw = ticker.info or {}
        if isinstance(raw, dict):
            info.update(raw)
    except Exception:
        pass

    fast = get_fast_info_dict(ticker)
    alias_map = {
        "market_cap": "marketCap",
        "last_price": "currentPrice",
        "previous_close": "previousClose",
        "shares": "sharesOutstanding",
        "currency": "currency",
        "exchange": "exchange",
        "year_high": "fiftyTwoWeekHigh",
        "year_low": "fiftyTwoWeekLow",
    }

    for k, v in fast.items():
        info.setdefault(k, v)
        if k in alias_map:
            info.setdefault(alias_map[k], v)

    return info


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
        hist = t.history(interval="1d", period="6mo", auto_adjust=False, prepost=False)
    if hist is None or hist.empty:
        return pd.DataFrame()
    return hist.dropna(how="all")


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

    adx = abs(
        (high.diff().abs().rolling(14).mean().iloc[-1] if len(high) >= 14 else 0)
        - (low.diff().abs().rolling(14).mean().iloc[-1] if len(low) >= 14 else 0)
    )
    plus_di = high.diff().clip(lower=0).rolling(14).mean().iloc[-1] if len(high) >= 14 else 0
    minus_di = (-low.diff().clip(upper=0)).rolling(14).mean().iloc[-1] if len(low) >= 14 else 0

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


def statement_to_list(df, limit=8):
    try:
        if df is None or df.empty:
            return []
        x = df.copy().replace([np.inf, -np.inf], np.nan).fillna(np.nan)
        cols = list(x.columns)[:limit]
        out = []
        for row_name in x.index[:20]:
            item = {"metric": str(row_name)}
            has_value = False
            for col in cols:
                key = str(col.date()) if hasattr(col, "date") else str(col)
                val = clean_number(x.loc[row_name, col])
                item[key] = val
                if val is not None:
                    has_value = True
            if has_value:
                out.append(item)
        return out
    except Exception:
        return []


def make_quarterly_data(quarterly_financials):
    rows = statement_to_list(quarterly_financials, limit=4)
    return rows[:12]


def make_corp_action(actions_df):
    try:
        if actions_df is None or actions_df.empty:
            return []
        df = actions_df.reset_index().fillna("")
        out = []
        for _, row in df.tail(10).iterrows():
            date_col = row.iloc[0]
            action = None
            value = None
            for col in df.columns[1:]:
                if row[col] not in ["", None, 0]:
                    action = str(col)
                    value = clean_number(row[col])
                    break
            out.append({
                "date": str(date_col.date()) if hasattr(date_col, "date") else str(date_col),
                "action": action or "Corporate action",
                "value": value
            })
        return out[::-1]
    except Exception:
        return []


def make_news(news_items):
    out = []
    try:
        for item in (news_items or [])[:10]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            url = content.get("canonicalUrl", {}).get("url") or content.get("clickThroughUrl", {}).get("url") or item.get("link")
            publisher = content.get("provider", {}).get("displayName") or content.get("publisher")
            pub_date = content.get("pubDate") or item.get("providerPublishTime")
            summary = content.get("summary") or ""
            if title:
                out.append({
                    "title": title,
                    "publisher": publisher,
                    "link": url,
                    "published": str(pub_date) if pub_date is not None else None,
                    "summary": summary[:220] if summary else ""
                })
    except Exception:
        return []
    return out


def make_ratios(info, hist=None):
    current_price = safe_dict_get(info, ["currentPrice", "lastPrice", "regularMarketPrice", "last_price"])
    previous_close = safe_dict_get(info, ["previousClose", "regularMarketPreviousClose", "previous_close"])
    market_cap = safe_dict_get(info, ["marketCap", "market_cap"])
    shares_outstanding = safe_dict_get(info, ["sharesOutstanding", "shares"])

    if market_cap is None and current_price is not None and shares_outstanding is not None:
        try:
            market_cap = float(current_price) * float(shares_outstanding)
        except Exception:
            pass

    fifty_two_high = safe_dict_get(info, ["fiftyTwoWeekHigh", "yearHigh", "year_high"])
    fifty_two_low = safe_dict_get(info, ["fiftyTwoWeekLow", "yearLow", "year_low"])

    if hist is not None and not hist.empty:
        try:
            h52 = hist["High"].tail(252).max()
            l52 = hist["Low"].tail(252).min()
            fifty_two_high = fifty_two_high if fifty_two_high is not None else clean_number(h52)
            fifty_two_low = fifty_two_low if fifty_two_low is not None else clean_number(l52)
        except Exception:
            pass

    items = [
        ("Market Cap", format_large_number(market_cap)),
        ("Current Price", clean_number(current_price)),
        ("Previous Close", clean_number(previous_close)),
        ("Trailing PE", clean_number(safe_dict_get(info, ["trailingPE"]))),
        ("Forward PE", clean_number(safe_dict_get(info, ["forwardPE"]))),
        ("Price to Book", clean_number(safe_dict_get(info, ["priceToBook"]))),
        ("Dividend Yield", format_percent(safe_dict_get(info, ["dividendYield"]))),
        ("ROE", format_percent(safe_dict_get(info, ["returnOnEquity"]))),
        ("ROA", format_percent(safe_dict_get(info, ["returnOnAssets"]))),
        ("Debt to Equity", clean_number(safe_dict_get(info, ["debtToEquity"]))),
        ("Current Ratio", clean_number(safe_dict_get(info, ["currentRatio"]))),
        ("Quick Ratio", clean_number(safe_dict_get(info, ["quickRatio"]))),
        ("Profit Margin", format_percent(safe_dict_get(info, ["profitMargins"]))),
        ("Operating Margin", format_percent(safe_dict_get(info, ["operatingMargins"]))),
        ("Revenue Growth", format_percent(safe_dict_get(info, ["revenueGrowth"]))),
        ("Earnings Growth", format_percent(safe_dict_get(info, ["earningsGrowth"]))),
        ("52 Week High", clean_number(fifty_two_high)),
        ("52 Week Low", clean_number(fifty_two_low)),
    ]

    out = [{"label": k, "value": v} for k, v in items if v is not None]
    if not out:
        out = [{"label": "Status", "value": "Yahoo fundamentals unavailable for this symbol"}]
    return out


def make_shareholding(info):
    insiders = safe_dict_get(info, ["heldPercentInsiders"])
    institutions = safe_dict_get(info, ["heldPercentInstitutions"])
    float_shares = safe_dict_get(info, ["floatShares"])
    shares_outstanding = safe_dict_get(info, ["sharesOutstanding", "shares"])
    implied_shares = safe_dict_get(info, ["impliedSharesOutstanding"])

    holders = [
        ("Promoter / Insider Holding", format_percent(insiders)),
        ("Institution Holding", format_percent(institutions)),
        ("Float Shares", format_large_number(float_shares)),
        ("Shares Outstanding", format_large_number(shares_outstanding)),
        ("Implied Shares Outstanding", format_large_number(implied_shares)),
    ]

    out = [{"label": k, "value": v} for k, v in holders if v is not None]
    if not out:
        out = [{"label": "Status", "value": "Shareholding data unavailable from Yahoo"}]
    return out


def make_investors(info):
    items = [
        ("Company", safe_dict_get(info, ["longName", "shortName"])),
        ("Sector", safe_dict_get(info, ["sector"])),
        ("Industry", safe_dict_get(info, ["industry"])),
        ("Website", safe_dict_get(info, ["website"])),
        ("Country", safe_dict_get(info, ["country"])),
        ("Employees", clean_number(safe_dict_get(info, ["fullTimeEmployees"]))),
        ("Exchange", safe_dict_get(info, ["exchange"])),
        ("Currency", safe_dict_get(info, ["currency"])),
    ]
    out = [{"label": k, "value": v} for k, v in items if v is not None]
    if not out:
        out = [{"label": "Status", "value": "Investor profile unavailable"}]
    return out


def make_reports(symbol, info):
    website = safe_dict_get(info, ["website"])
    company_name = safe_dict_get(info, ["longName", "shortName"], symbol.upper())
    reports = []
    if website:
        reports.append({
            "title": f"{company_name} Website",
            "type": "Company",
            "link": website
        })
    reports.append({
        "title": f"{company_name} on Yahoo Finance",
        "type": "Market profile",
        "link": f"https://finance.yahoo.com/quote/{symbol}"
    })
    return reports


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

    ticker = yf.Ticker(symbol)
    hist = get_history(symbol, "1d", "1mo")
    if hist.empty:
        return {"symbol": symbol.upper(), "error": "No data"}

    latest_price = None
    previous_close = None
    source = "yfinance history"

    try:
        fi = get_fast_info_dict(ticker)
        latest_price = safe_dict_get(fi, ["lastPrice", "last_price", "regularMarketPrice"])
        previous_close = safe_dict_get(fi, ["previousClose", "previous_close", "regularMarketPreviousClose"])
        if latest_price is not None:
            source = "yfinance fast_info"
    except Exception:
        pass

    if latest_price is None:
        latest_price = clean_number(hist.iloc[-1]["Close"])

    if previous_close is None:
        previous_close = clean_number(hist.iloc[-2]["Close"]) if len(hist) > 1 else latest_price

    return {
        "symbol": symbol.upper(),
        "price": clean_number(latest_price),
        "previousClose": clean_number(previous_close),
        "market_status": "open" if is_open else "closed",
        "timestamp": str(hist.index[-1]),
        "source": source
    }


@app.get("/api/yahoo/analyse")
def analyse(symbol: str = Query(...)):
    now = datetime.now(IST)
    is_open = now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)
    interval = "30m" if is_open else "1d"
    period = "5d" if is_open else "6mo"

    ticker = yf.Ticker(symbol)
    hist = get_history(symbol, interval, period)
    if hist.empty:
        return {"symbol": symbol.upper(), "error": "No data"}

    techs = compute_techniques(hist)
    scores = [t["score"] for t in techs]
    overall = round(float(np.mean(scores)), 1)
    short = round(float(np.mean(scores[:20])), 1)
    long = round(float(np.mean(scores[20:40])), 1)
    risk = round(float(np.mean(scores[40:])), 1)

    info = merge_info(ticker)

    fund_parts = [
        safe_dict_get(info, ["returnOnEquity"]),
        safe_dict_get(info, ["profitMargins"]),
        safe_dict_get(info, ["operatingMargins"]),
        safe_dict_get(info, ["revenueGrowth"]),
        safe_dict_get(info, ["earningsGrowth"]),
    ]

    fund_values = []
    for x in fund_parts:
        try:
            fx = float(x)
            fund_values.append(fx * 100 if abs(fx) <= 1 else fx)
        except Exception:
            pass

    fund = round(float(np.mean(fund_values)), 1) if fund_values else 50.0
    fund = max(0.0, min(100.0, fund))

    signal = "Bullish" if overall >= 70 else "Positive" if overall >= 55 else "Neutral" if overall >= 42 else "Cautious"
    verdict_reason = (
        "Strong technical mix with broad confirmation." if signal == "Bullish"
        else "Constructive setup with more positives than negatives." if signal == "Positive"
        else "Mixed signals across trend, momentum, and risk." if signal == "Neutral"
        else "Weak trend or elevated risk across multiple techniques."
    )

    quarterly_fin = None
    yearly_fin = None
    balance_sheet = None
    cashflow = None
    actions = None
    news_items = []

    try:
        quarterly_fin = ticker.quarterly_financials
    except Exception:
        pass
    try:
        yearly_fin = ticker.financials
    except Exception:
        pass
    try:
        balance_sheet = ticker.balance_sheet
    except Exception:
        pass
    try:
        cashflow = ticker.cashflow
    except Exception:
        pass
    try:
        actions = ticker.actions
    except Exception:
        pass
    try:
        news_items = ticker.news or []
    except Exception:
        news_items = []

    ratios = make_ratios(info, hist)
    shareholding = make_shareholding(info)
    quarterly = make_quarterly_data(quarterly_fin)
    pnl = statement_to_list(yearly_fin, limit=4)
    balance_sheet_list = statement_to_list(balance_sheet, limit=4)
    cashflow_list = statement_to_list(cashflow, limit=4)
    corp_action = make_corp_action(actions)
    investors = make_investors(info)
    reports = make_reports(symbol, info)
    news = make_news(news_items)

    name = safe_dict_get(info, ["longName", "shortName"], symbol.upper())

    debug = {
        "info_key_count": len(info.keys()) if isinstance(info, dict) else 0,
        "has_trailingPE": safe_dict_get(info, ["trailingPE"]) is not None,
        "has_priceToBook": safe_dict_get(info, ["priceToBook"]) is not None,
        "has_heldPercentInsiders": safe_dict_get(info, ["heldPercentInsiders"]) is not None,
        "has_heldPercentInstitutions": safe_dict_get(info, ["heldPercentInstitutions"]) is not None,
        "has_marketCap": safe_dict_get(info, ["marketCap", "market_cap"]) is not None,
        "has_sharesOutstanding": safe_dict_get(info, ["sharesOutstanding", "shares"]) is not None,
    }

    return {
        "symbol": symbol.upper(),
        "name": name,
        "short": short,
        "long": long,
        "fund": fund,
        "risk": risk,
        "overall": overall,
        "signal": signal,
        "techniques": techs,
        "technique_count": len(techs),
        "verdict_reason": verdict_reason,
        "market_status": "open" if is_open else "closed",
        "close_used": clean_number(hist["Close"].iloc[-1]),
        "source": f"yfinance {interval}",
        "ratios": ratios,
        "shareholding": shareholding,
        "quarterly": quarterly,
        "pnl": pnl,
        "balanceSheet": balance_sheet_list,
        "cashflow": cashflow_list,
        "corpAction": corp_action,
        "investors": investors,
        "reports": reports,
        "news": news,
        "meta": {
            "sector": safe_dict_get(info, ["sector"]),
            "industry": safe_dict_get(info, ["industry"]),
            "exchange": safe_dict_get(info, ["exchange"]),
            "currency": safe_dict_get(info, ["currency"]),
            "website": safe_dict_get(info, ["website"])
        },
        "debug": debug
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
