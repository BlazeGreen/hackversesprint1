import asyncio
import json
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from agents import run_pipeline, stream_pipeline
from live_market import fetch_live_quote
from log import log_session, read_recent_logs

app = FastAPI(title="Retail Financial Intelligence - Multi-Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_json(fname):
    with open(os.path.join(DATA_DIR, fname)) as f:
        return json.load(f)


@app.get("/tickers")
def list_tickers():
    market = _load_json("market_feed.json")
    return {"tickers": list(market.keys())}


@app.get("/quotes")
async def quotes():
    """Live snapshot for all covered tickers -- pure market data, no LLM calls.
    Powers the ticker tape without touching Gemini quota."""
    market = _load_json("market_feed.json")
    result = {}
    for ticker, fallback in market.items():
        result[ticker] = await asyncio.to_thread(fetch_live_quote, ticker, fallback)
    return result


@app.get("/profiles")
def list_profiles():
    return _load_json("user_profiles.json")


def _resolve_profile(profile_id: str | None, profile_json: str | None) -> dict:
    """Either a saved preset (profile_id) or a user-built profile from the
    onboarding form, passed as a URL-encoded JSON blob (profile_json)."""
    if profile_json:
        try:
            return json.loads(profile_json)
        except json.JSONDecodeError:
            raise HTTPException(400, "profile_json is not valid JSON")
    profiles = _load_json("user_profiles.json")
    if profile_id not in profiles:
        raise HTTPException(404, f"No such profile {profile_id}")
    return profiles[profile_id]


@app.get("/analyze/{ticker}")
async def analyze(ticker: str, profile_id: str = "conservative_investor", profile_json: str = None):
    market = _load_json("market_feed.json")

    if ticker not in market:
        raise HTTPException(404, f"No market data for {ticker}")
    user_profile = _resolve_profile(profile_id, profile_json)

    market_data = await asyncio.to_thread(fetch_live_quote, ticker, market[ticker])

    start = time.time()
    result = await run_pipeline(ticker, market_data, user_profile)
    total_latency_ms = int((time.time() - start) * 1000)

    log_entry = log_session(ticker, user_profile, result, total_latency_ms)

    return {
        "ticker": ticker,
        "market_data": market_data,
        "user_profile": user_profile,
        "pipeline": result,
        "total_latency_ms": total_latency_ms,
        "log_entry": log_entry,
    }


@app.get("/analyze_stream/{ticker}")
async def analyze_stream(ticker: str, profile_id: str = "conservative_investor", profile_json: str = None):
    """
    Server-Sent Events endpoint. Streams one JSON event per line as each
    agent finishes, instead of making the client wait for the full pipeline.
    Frontend consumes this with EventSource.
    """
    market = _load_json("market_feed.json")

    if ticker not in market:
        raise HTTPException(404, f"No market data for {ticker}")
    user_profile = _resolve_profile(profile_id, profile_json)

    market_data = await asyncio.to_thread(fetch_live_quote, ticker, market[ticker])

    async def event_generator():
        start = time.time()
        specialists_seen = []
        synthesis_seen = None
        async for event in stream_pipeline(ticker, market_data, user_profile):
            if event["type"] == "agent_result":
                specialists_seen.append(event["result"])
            if event["type"] == "synthesis":
                synthesis_seen = event["result"]
            # also echo back market/profile context on the first event for the UI
            payload = dict(event)
            yield f"data: {json.dumps(payload)}\n\n"

        total_latency_ms = int((time.time() - start) * 1000)
        pipeline_result = {"ticker": ticker, "specialists": specialists_seen, "synthesis": synthesis_seen}
        log_entry = log_session(ticker, user_profile, pipeline_result, total_latency_ms)
        yield f"data: {json.dumps({'type': 'log', 'log_entry': log_entry, 'market_data': market_data, 'user_profile': user_profile})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/logs")
def get_logs():
    return {"logs": read_recent_logs()}


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the custom frontend. Must be mounted LAST so it doesn't shadow API routes above.
static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend_v2")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
