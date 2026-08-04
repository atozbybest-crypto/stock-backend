from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Stock Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

IST = ZoneInfo("Asia/Kolkata")


TECHNIQUE_INFO = {
    "VWAP": {
        "definition": "VWAP estimates the average traded price during a session, weighted by volume.",
        "how_to_read": "Price above VWAP supports an intraday bullish bias. Price below VWAP supports a bearish bias.",
        "how_traders_use_it": "Use VWAP as an intraday bias line and as a reference for pullbacks, entries, and exits.",
    },
    "EMA 9/20": {
        "definition": "The 9-period and 20-period exponential moving averages show short-term direction.",
        "how_to_read": "EMA 9 above EMA 20 supports bullish momentum. EMA 9 below EMA 20 supports bearish momentum.",
        "how_traders_use_it": "Use EMA 9/20 as a trend filter or pullback confirmation.",
    },
    "Volume": {
        "definition": "Volume shows how much trading activity supports a price move.",
        "how_to_read": "A move with volume above its recent average is more strongly confirmed than a low-volume move.",
        "how_traders_use_it": "Compare current volume with the recent average and check whether it confirms price direction.",
    },
    "Support-Resistance": {
        "definition": "Support and resistance are price areas where buying or selling pressure has previously appeared.",
        "how_to_read": "Support can act as a floor and resistance as a ceiling, but either level can fail.",
        "how_traders_use_it": "Use these zones to plan entries, stop-loss locations, and targets.",
    },
    "ATR": {
        "definition": "ATR measures typical price movement and volatility, including gaps.",
        "how_to_read": "Higher ATR means the stock is moving more and may need a wider stop or smaller position.",
        "how_traders_use_it": "Use ATR for volatility-aware stops, position sizing, and target planning.",
    },
    "RSI": {
        "definition": "RSI measures the speed and change of price movements on a bounded scale.",
        "how_to_read": "High RSI shows strong or potentially stretched momentum. Low RSI shows weak or potentially stretched momentum.",
        "how_traders_use_it": "Use RSI with trend and price structure. Do not trade only because RSI is high or low.",
    },
    "MACD": {
        "definition": "MACD compares moving averages to show changes in momentum and trend direction.",
        "how_to_read": "MACD above its signal line and above zero supports bullish momentum. The opposite supports bearish momentum.",
        "how_traders_use_it": "Use MACD as confirmation after identifying trend and key price levels.",
    },
    "ADX": {
        "definition": "ADX estimates trend strength rather than directly predicting price direction.",
        "how_to_read": "Higher ADX suggests a stronger trend. Low ADX often suggests a range or weak trend.",
        "how_traders_use_it": "Use ADX to decide whether a trend-following setup is worth considering.",
    },
    "Breakout": {
        "definition": "A breakout occurs when price moves beyond an established range or level.",
        "how_to_read": "A close beyond the level with confirming volume is stronger than a brief move through it.",
        "how_traders_use_it": "Wait for confirmation or a retest and define the invalidation level before entering.",
    },
    "Price Action": {
        "definition": "Price action reads swings, candles, ranges, and market structure directly from price.",
        "how_to_read": "Higher highs and higher lows support an uptrend. Lower highs and lower lows support a downtrend.",
        "how_traders_use_it": "Use price action as the final context check around levels, moving averages, VWAP, and breakouts.",
    },
}


def is_market_open():
    now = datetime.now(IST)
    return (
        now.weekday() < 5
        and time(9, 15) <= now.time() <= time(15, 30)
    )


def clean_number(value, digits=2):
    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return round(value, digits)
    except Exception:
        return None


def normalise_history(data):
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in data.columns
        ]

    data.columns = [str(column).title() for column in data.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    available = [column for column in required if column in data.columns]

    data = data[available]
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])

    return data


def get_history(symbol, interval="1d", period="1y"):
    try:
        ticker = yf.Ticker(symbol)

        data = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False,
        )

        data = normalise_history(data)

        if not data.empty:
            return data

    except Exception:
        pass

    return pd.DataFrame()


def calculate_indicators(data):
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].fillna(0).astype(float)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    loss = -delta.clip(upper=0).ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.fillna(50)

    macd_line = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )

    macd_signal = macd_line.ewm(
        span=9,
        adjust=False,
    ).mean()

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        window=14,
        min_periods=5,
    ).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0,
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0,
    )

    atr14 = true_range.rolling(
        window=14,
        min_periods=5,
    ).mean().replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.rolling(14, min_periods=5).sum()
        / atr14
    )

    minus_di = (
        100
        * minus_dm.rolling(14, min_periods=5).sum()
        / atr14
    )

    direction_index = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    adx = direction_index.rolling(
        window=14,
        min_periods=5,
    ).mean().fillna(0)

    typical_price = (high + low + close) / 3

    cumulative_volume = volume.cumsum().replace(0, np.nan)

    vwap = (
        typical_price * volume
    ).cumsum() / cumulative_volume

    average_volume = volume.rolling(
        window=20,
        min_periods=5,
    ).mean()

    support = low.shift(1).rolling(
        window=20,
        min_periods=5,
    ).min()

    resistance = high.shift(1).rolling(
        window=20,
        min_periods=5,
    ).max()

    return {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "ema9": ema9,
        "ema20": ema20,
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "atr": atr,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "vwap": vwap,
        "average_volume": average_volume,
        "support": support,
        "resistance": resistance,
    }


def make_technique(name, signal, score, note):
    info = TECHNIQUE_INFO[name]

    return {
        "name": name,
        "signal": signal,
        "verdict": signal,
        "score": round(float(score), 1),
        "note": note,
        "definition": info["definition"],
        "how_to_read": info["how_to_read"],
        "how_traders_use_it": info["how_traders_use_it"],
    }


def compute_techniques(data):
    if len(data) < 20:
        return [
            make_technique(
                name,
                "Neutral",
                50,
                "Not enough data.",
            )
            for name in TECHNIQUE_INFO
        ]

    values = calculate_indicators(data)

    close = values["close"]
    high = values["high"]
    low = values["low"]
    volume = values["volume"]

    last = float(close.iloc[-1])
    previous = float(close.iloc[-2])

    ema9 = float(values["ema9"].iloc[-1])
    ema20 = float(values["ema20"].iloc[-1])

    rsi = float(values["rsi"].iloc[-1])

    macd = float(values["macd"].iloc[-1])
    macd_signal = float(values["macd_signal"].iloc[-1])

    atr_series = values["atr"].dropna()
    atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0

    adx = float(values["adx"].iloc[-1])
    plus_di = float(values["plus_di"].iloc[-1] or 0)
    minus_di = float(values["minus_di"].iloc[-1] or 0)

    vwap_value = values["vwap"].iloc[-1]
    vwap = float(vwap_value) if pd.notna(vwap_value) else last

    average_volume_value = values["average_volume"].iloc[-1]
    average_volume = (
        float(average_volume_value)
        if pd.notna(average_volume_value)
        else 0
    )

    support_value = values["support"].iloc[-1]
    resistance_value = values["resistance"].iloc[-1]

    support = (
        float(support_value)
        if pd.notna(support_value)
        else last
    )

    resistance = (
        float(resistance_value)
        if pd.notna(resistance_value)
        else last
    )

    current_volume = float(volume.iloc[-1])

    volume_confirmed = (
        current_volume > average_volume
        if average_volume > 0
        else False
    )

    ema_bullish = ema9 > ema20
    price_above_vwap = last > vwap

    rsi_bullish = 50 <= rsi < 70
    rsi_bearish = rsi < 40 or rsi > 75

    macd_bullish = (
        macd > macd_signal
        and macd > 0
    )

    macd_bearish = (
        macd < macd_signal
        and macd < 0
    )

    adx_bullish = adx >= 20 and plus_di > minus_di
    adx_bearish = adx >= 20 and minus_di > plus_di

    breakout_bullish = (
        last > resistance
        and volume_confirmed
    )

    breakout_bearish = (
        last < support
        and volume_confirmed
    )

    price_action_bullish = (
        last > previous
        and last > float(close.iloc[-3])
    )

    price_action_bearish = (
        last < previous
        and last < float(close.iloc[-3])
    )

    if last > support and last > previous:
        sr_signal = "Bullish"
        sr_score = 65
    elif last < support:
        sr_signal = "Bearish"
        sr_score = 35
    else:
        sr_signal = "Neutral"
        sr_score = 50

    if breakout_bullish:
        breakout_signal = "Bullish"
        breakout_score = 75
    elif breakout_bearish:
        breakout_signal = "Bearish"
        breakout_score = 25
    else:
        breakout_signal = "Neutral"
        breakout_score = 50

    if price_action_bullish:
        price_action_signal = "Bullish"
        price_action_score = 65
    elif price_action_bearish:
        price_action_signal = "Bearish"
        price_action_score = 35
    else:
        price_action_signal = "Neutral"
        price_action_score = 50

    return [
        make_technique(
            "VWAP",
            "Bullish" if price_above_vwap else "Bearish",
            70 if price_above_vwap else 30,
            f"Close {last:.2f}; VWAP {vwap:.2f}.",
        ),
        make_technique(
            "EMA 9/20",
            "Bullish" if ema_bullish else "Bearish",
            70 if ema_bullish else 30,
            f"EMA9 {ema9:.2f}; EMA20 {ema20:.2f}.",
        ),
        make_technique(
            "Volume",
            "Bullish" if volume_confirmed else "Neutral",
            68 if volume_confirmed else 45,
            f"Current volume {current_volume:.0f}; average {average_volume:.0f}.",
        ),
        make_technique(
            "Support-Resistance",
            sr_signal,
            sr_score,
            f"Support {support:.2f}; resistance {resistance:.2f}.",
        ),
        make_technique(
            "ATR",
            "Neutral",
            50,
            f"ATR is {atr:.2f}. ATR measures volatility, not direction.",
        ),
        make_technique(
            "RSI",
            "Bullish" if rsi_bullish else "Bearish" if rsi_bearish else "Neutral",
            65 if rsi_bullish else 35 if rsi_bearish else 50,
            f"RSI14 is {rsi:.2f}.",
        ),
        make_technique(
            "MACD",
            "Bullish" if macd_bullish else "Bearish" if macd_bearish else "Neutral",
            68 if macd_bullish else 32 if macd_bearish else 50,
            f"MACD {macd:.3f}; signal {macd_signal:.3f}.",
        ),
        make_technique(
            "ADX",
            "Bullish" if adx_bullish else "Bearish" if adx_bearish else "Neutral",
            65 if adx_bullish else 35 if adx_bearish else 50,
            f"ADX {adx:.2f}; +DI {plus_di:.2f}; -DI {minus_di:.2f}.",
        ),
        make_technique(
            "Breakout",
            breakout_signal,
            breakout_score,
            f"Close {last:.2f}; resistance {resistance:.2f}; support {support:.2f}.",
        ),
        make_technique(
            "Price Action",
            price_action_signal,
            price_action_score,
            "Based on recent price movement and short-term structure.",
        ),
    ]


def overall_scores(techniques):
    scores = np.array(
        [item["score"] for item in techniques],
        dtype=float,
    )

    overall = float(scores.mean()) if len(scores) else 50
    short_score = float(scores[:5].mean())
    long_score = float(scores[5:].mean())

    bullish_count = sum(
        item["signal"] == "Bullish"
        for item in techniques
    )

    bearish_count = sum(
        item["signal"] == "Bearish"
        for item in techniques
    )

    if overall >= 62 and bullish_count >= bearish_count + 2:
        signal = "Bullish"
    elif overall <= 38 and bearish_count >= bullish_count + 2:
        signal = "Bearish"
    else:
        signal = "Neutral"

    risk_score = 50 + abs(bullish_count - bearish_count) * 3
    risk_score = min(100, max(0, risk_score))

    return (
        round(overall, 1),
        round(short_score, 1),
        round(long_score, 1),
        round(risk_score, 1),
        signal,
    )


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Stock Analyzer backend running",
    }


@app.get("/api/yahoo/quote")
def quote(symbol: str = Query(...)):
    symbol = symbol.strip().upper()
    data = get_history(symbol, "1d", "1mo")

    if data.empty:
        return {
            "symbol": symbol,
            "error": "No data found for this symbol.",
        }

    latest_price = float(data["Close"].iloc[-1])

    previous_close = (
        float(data["Close"].iloc[-2])
        if len(data) > 1
        else latest_price
    )

    return {
        "symbol": symbol,
        "price": clean_number(latest_price),
        "previousClose": clean_number(previous_close),
        "market_status": "open" if is_market_open() else "closed",
        "timestamp": str(data.index[-1]),
        "source": "yfinance",
    }


@app.get("/api/yahoo/analyse")
def analyse(symbol: str = Query(...)):
    symbol = symbol.strip().upper()

    market_open = is_market_open()

    interval = "5m" if market_open else "1d"
    period = "5d" if market_open else "1y"

    data = get_history(symbol, interval, period)

    if data.empty or len(data) < 20:
        data = get_history(symbol, "1d", "1y")

    if data.empty:
        return {
            "symbol": symbol,
            "error": "No data found for this symbol.",
        }

    techniques = compute_techniques(data)

    overall, short_score, long_score, risk_score, signal = (
        overall_scores(techniques)
    )

    bullish_count = sum(
        item["signal"] == "Bullish"
        for item in techniques
    )

    bearish_count = sum(
        item["signal"] == "Bearish"
        for item in techniques
    )

    neutral_count = 10 - bullish_count - bearish_count

    reason = (
        f"{bullish_count} bullish, "
        f"{bearish_count} bearish, and "
        f"{neutral_count} neutral technique verdicts."
    )

    return {
        "symbol": symbol,
        "name": symbol.replace(".NS", ""),
        "short": short_score,
        "long": long_score,
        "risk": risk_score,
        "overall": overall,
        "signal": signal,
        "verdict_reason": reason,
        "techniques": techniques,
        "technique_count": len(techniques),
        "source": f"yfinance {interval}",
        "market_status": "open" if market_open else "closed",
    }


@app.get("/api/yahoo/chart")
def chart(
    symbol: str = Query(...),
    period: str = Query("1mo"),
    interval: str = Query("1d"),
):
    symbol = symbol.strip().upper()
    data = get_history(symbol, interval, period)

    if data.empty:
        return {
            "symbol": symbol,
            "error": "No data found for this symbol.",
        }

    points = []

    for index, row in data.tail(300).iterrows():
        points.append(
            {
                "time": str(index),
                "close": clean_number(row["Close"]),
            }
        )

    return {
        "symbol": symbol,
        "points": points,
    }
