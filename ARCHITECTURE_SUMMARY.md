# VANTAGE — Architecture & Decision Logic Summary

**PS-01 — Multi-Agent Autonomous Financial Intelligence System for Retail Investors**
HACKVERSE: Into the Web, Sprint 1 · IEEE RAS VIT Chennai Student Chapter

## What it does

A user fills a short profile form (risk tolerance, horizon, goal, monthly
capacity, drawdown comfort, current holdings), picks a ticker, and the
system dispatches three independent specialist agents in parallel — momentum,
sentiment, and filings (RAG-grounded) — each producing a classified signal
with a confidence score and cited reasoning. A fourth synthesis agent reads
all three outputs plus the user's profile and produces one personalized
BUY/SELL/HOLD/AVOID recommendation, explaining how the profile shaped the
call and flagging any degraded or conflicting inputs. Every step streams to
the UI live over Server-Sent Events as it happens.

## Agent architecture

| Agent | Role | Input | Output |
|---|---|---|---|
| **Momentum** | Technical analysis | Live price, RSI, volume, 5-day momentum | Signal + confidence + reasoning |
| **Sentiment** | News/sentiment analysis | Recent headlines | Signal + confidence + reasoning |
| **Filings (RAG)** | Fundamentals, grounded in source documents | Retrieved filing excerpts | Signal + confidence + reasoning + cited source filename |
| **Synthesizer** | Personalized decision synthesis | All 3 specialist outputs + user profile | Final recommendation + confidence + reasoning + conflict/degradation flags |

The three specialists execute in parallel (`asyncio.gather` / `asyncio.as_completed`
in `backend/agents.py`); the synthesizer runs after, once all three are in.
Each agent is a single call to Gemini (`gemini-3.6-flash`) with a strict
JSON output contract (`responseMimeType: application/json` plus an explicit
schema in the prompt), so every agent's output is structured and directly
consumable by the synthesis layer — no free-text parsing.

## Data sources

- **Live market data**: real-time price, volume, and technical indicators
  (RSI-14 computed locally, 5-day momentum, 30-day volume average) pulled
  from Yahoo Finance's public chart API for 6 NSE-listed equities (TCS,
  ZOMATO, HDFCBANK, RELIANCE, INFY, ICICIBANK). Falls back to static sample
  data if the live fetch fails, tagged transparently in the UI (`LIVE · NSE`
  vs `SAMPLE DATA` badge) so a network hiccup degrades gracefully instead of
  breaking the demo.
- **Regulatory filings corpus**: quarterly-filing-style excerpts per company
  (`backend/data/filings/*.txt`), used as the retrieval corpus for the RAG
  agent.
- **User profile**: captured through a 6-field onboarding form, held
  client-side and passed to the backend per-request — no fixed presets, so
  every recommendation is generated against the actual person using it.

## Retrieval-augmented generation

The filings agent queries `backend/retriever.py` before reasoning: it scores
each filing document by keyword overlap with the query (with a bonus for
filename-ticker matches), returns the top-scoring excerpt(s), and the agent
is instructed to reason *only* over the retrieved text and cite the source
filename — visible to the user next to the agent's output.

**Known limitation**: retrieval here is lexical (keyword/count overlap), not
embedding-based semantic search. For a 6-document corpus at hackathon scope
this reliably retrieves the right filing per ticker, but it doesn't capture
paraphrase/synonym matches the way a vector-embedding retriever would. Flagged
here deliberately rather than silently — the natural next step is to embed
the corpus once (small, one-time cost) and the query per call, then retrieve
by cosine similarity instead of keyword count.

## Personalization

The synthesizer receives the full user profile object (risk tolerance,
horizon, goal, monthly capacity, max drawdown tolerance, current holdings)
verbatim in its prompt and is explicitly instructed to weigh the specialist
signals against it — e.g. a high-conviction BULLISH momentum read can still
resolve to HOLD or AVOID for a low-risk-tolerance profile with a small
drawdown budget, with the reasoning stating why. Because the profile is
user-built per session rather than a fixed preset, running the same ticker
against two different profiles produces two different, individually-justified
recommendations from identical market inputs.

## Reliability / degraded-data handling

Every agent call is wrapped in a try/except: on any failure (network error,
timeout, malformed response, upstream rate limit), the agent returns a
structured `"degraded": true` result with `signal: "UNKNOWN"` and the error
reason, instead of raising and killing the pipeline. The synthesizer is
explicitly instructed to call out degraded inputs in its reasoning rather
than silently ignore them, and the UI marks degraded agent cards distinctly.
This was exercised for real during development against actual upstream
`429` rate-limit responses from the model provider, not just simulated.

## Interface

A single self-contained page (`frontend_v2/index.html`, no build step)
renders: a live ticker tape across all covered names, the onboarding profile
form, per-agent cards that fill in as each specialist completes (via SSE, not
a single blocking spinner), the synthesized recommendation with a confidence
bar, and current market signal + user profile state.

## Performance logging

Every session appends one line to `backend/data/session_log.jsonl`
(`backend/log.py`) capturing: total pipeline latency, per-agent latency,
a portfolio concentration score (1/N over current holdings), the final
recommendation, and which agents (including the synthesizer) came back
degraded — retrievable via `GET /logs`.

## Stack

FastAPI + httpx (backend, `backend/`), one vanilla HTML/CSS/JS page
(`frontend_v2/index.html`) served directly by FastAPI's `StaticFiles`,
Gemini 3.6 Flash as the underlying model for all four agents, Yahoo Finance
for live market data. No vector DB, no agent framework — the orchestration
is plain `asyncio`.
