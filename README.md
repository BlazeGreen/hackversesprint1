# DESK — Multi-Agent Financial Intelligence

**HACKVERSE: Into the Web, Sprint 1 · PS-01** — IEEE Robotics & Automation Society, VIT Chennai Student Chapter

A multi-agent AI system that turns live market data, regulatory filings, and
a user's own risk profile into an explainable, personalized investment
recommendation — not a generic "the stock is up" signal, but a reasoned,
cited answer to "what does this mean for *this specific person*."

See **[ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)** for the full
agent architecture and decision-logic writeup.

## How it works, in short

1. User fills a short profile form (risk tolerance, horizon, goal, capacity,
   drawdown comfort, holdings) and picks a ticker.
2. Three specialist agents run in parallel against real Gemini calls:
   **momentum** (technicals), **sentiment** (news), and **filings**
   (RAG-grounded — retrieves and cites an actual filing excerpt).
3. A **synthesizer** agent reads all three outputs plus the user's profile
   and produces one personalized BUY/SELL/HOLD/AVOID call, explaining how
   the profile shaped it.
4. Every step streams to the UI live as it happens (Server-Sent Events) —
   no single blocking spinner.

Covers 6 NSE-listed tickers with **live** price/volume/RSI/momentum data
(Yahoo Finance, no API key needed for that part): TCS, ZOMATO, HDFCBANK,
RELIANCE, INFY, ICICIBANK.

## Running it

**Requirements:** Python 3.10+, and a [Gemini API key](https://aistudio.google.com/apikey)
(the free tier works, but is capped at ~20 requests/day and each analysis
run costs 4 — budget accordingly if testing repeatedly).

```bash
cd backend
pip install -r requirements.txt
```

Then set your key and start the server. **The exact command depends on your shell:**

**macOS / Linux / Git Bash (Windows):**
```bash
export GEMINI_API_KEY='your-key-here'
python3 -m uvicorn main:app --port 8000
```

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="your-key-here"
python -m uvicorn main:app --port 8000
```

**Windows Command Prompt:**
```cmd
set GEMINI_API_KEY=your-key-here
python -m uvicorn main:app --port 8000
```

Then open **http://localhost:8000** — that one process serves both the API
and the frontend, no separate build step.

> If `python`/`python3` isn't found on Windows, try the `py` launcher
> instead: `py -m uvicorn main:app --port 8000`.

## Project structure

```
backend/
  main.py              FastAPI app — routes + SSE streaming
  agents.py             The 4 agents (momentum/sentiment/filings/synthesizer)
  live_market.py         Live Yahoo Finance quotes + RSI, with sample-data fallback
  retriever.py            Keyword-overlap retrieval over the filings corpus
  log.py                   Session performance logging
  data/
    market_feed.json        Sample market data (used only as a fallback)
    user_profiles.json      Legacy preset profiles (superseded by the intake form)
    filings/*.txt            Sample regulatory filing excerpts (RAG corpus)
    session_log.jsonl        Per-session performance log (git-ignored, generated at runtime)
frontend_v2/
  index.html              The entire UI — profile intake form + live dashboard
frontend/
  app.py                  Older Streamlit UI (backup, non-streaming)
```

## Known limitations

- **RAG retrieval is lexical (keyword overlap), not embedding-based semantic
  search.** Reliable for this corpus size, but doesn't capture paraphrase/
  synonym matches the way vector embeddings would. See
  [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) for detail.
- **News headlines are curated, not live** — Yahoo's free news search isn't
  reliably ticker-specific, so market data is live but headlines are sample
  data.
- **Gemini free-tier quota is tight** (20 requests/day, 4 per run) — if every
  agent suddenly shows "degraded" with a 429 error, this is almost always
  the cause, not a bug.
