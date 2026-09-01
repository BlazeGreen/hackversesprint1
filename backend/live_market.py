"""
Live market data for the NSE-listed tickers this app covers, pulled from
Yahoo Finance's public chart endpoint (no API key required). Falls back to
the static sample data in data/market_feed.json if the network call fails,
so a flaky connection degrades gracefully instead of breaking the demo.
"""
import time
import httpx

YAHOO_SYMBOLS = {"TCS": "TCS.NS", "ZOMATO": "ETERNAL.NS", "HDFCBANK": "HDFCBANK.NS"}
CACHE_TTL_SECONDS = 120

_client = httpx.Client(timeout=8, headers={"User-Agent": "Mozilla/5.0"})
_cache = {}


def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(-period, 0):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses += -delta
    avg_gain, avg_loss = gains / period, losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


def fetch_live_quote(ticker: str, fallback: dict) -> dict:
    symbol = YAHOO_SYMBOLS.get(ticker)
    if not symbol:
        return {**fallback, "data_source": "sample"}

    cached = _cache.get(ticker)
    if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        resp = _client.get(
            f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": "2mo", "interval": "1d"},
        )
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        meta = result["meta"]
        quote = result["indicators"]["quote"][0]
        closes = [c for c in quote["close"] if c is not None]
        volumes = [v for v in quote["volume"] if v is not None]

        avg_volume_30d = int(sum(volumes[-30:]) / len(volumes[-30:])) if volumes else 0
        momentum_5d = (
            round((closes[-1] - closes[-6]) / closes[-6] * 100, 2)
            if len(closes) >= 6 else 0
        )

        data = {
            **fallback,
            "price": round(meta["regularMarketPrice"], 2),
            "change_pct": round(meta.get("regularMarketChangePercent", 0), 2),
            "volume": meta.get("regularMarketVolume", volumes[-1] if volumes else 0),
            "avg_volume_30d": avg_volume_30d,
            "momentum_5d": momentum_5d,
            "rsi": _rsi(closes) or fallback.get("rsi", 50),
            "exchange": meta.get("fullExchangeName", "NSE"),
            "data_source": "live",
        }
        _cache[ticker] = (time.time(), data)
        return data
    except Exception:
        return {**fallback, "data_source": "sample"}
