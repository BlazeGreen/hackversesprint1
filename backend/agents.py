import os
import json
import time
import asyncio
import traceback
import httpx
from retriever import retrieve

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

AGENT_JSON_INSTRUCTIONS = (
    "Respond with ONLY a JSON object, no markdown fences, no preamble. "
    'Schema: {"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": <0-100 integer>, '
    '"reasoning": "<2-3 sentence explanation>", "sources": ["<short source label>"]}'
)

_http_client = httpx.AsyncClient(
    timeout=30,
    transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"),
)


async def _call_agent(system_prompt: str, user_prompt: str, agent_name: str, timeout=45):
    start = time.time()
    try:
        resp = await asyncio.wait_for(
            _http_client.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "generationConfig": {
                        "temperature": 0.3,
                        "responseMimeType": "application/json",
                        # Gemini 3.x's extended "thinking" adds real latency for
                        # not much benefit on these short classification tasks;
                        # "low" keeps calls fast enough to fit the timeout.
                        "thinkingConfig": {"thinkingLevel": "low"},
                    },
                },
            ),
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        data["agent"] = agent_name
        data["latency_ms"] = int((time.time() - start) * 1000)
        data["degraded"] = False
        return data
    except Exception as e:
        # Degraded-data handling: never let a failure kill the pipeline
        body = getattr(getattr(e, "response", None), "text", None)
        print(f"[{agent_name}] FULL ERROR:\n{traceback.format_exc()}\nresponse body: {body}")
        return {
            "agent": agent_name,
            "signal": "UNKNOWN",
            "confidence": 0,
            "reasoning": f"Agent failed or timed out: {str(e)[:120]}",
            "sources": [],
            "latency_ms": int((time.time() - start) * 1000),
            "degraded": True,
        }


async def momentum_agent(ticker: str, market_data: dict):
    system = (
        "You are a momentum analysis agent for a retail investing tool. "
        "Evaluate price momentum, volume anomaly, and RSI. "
        + AGENT_JSON_INSTRUCTIONS
    )
    user = f"Ticker: {ticker}\nMarket data: {json.dumps(market_data)}"
    return await _call_agent(system, user, "momentum_agent")


async def sentiment_agent(ticker: str, market_data: dict):
    system = (
        "You are a news/sentiment analysis agent for a retail investing tool. "
        "Evaluate the tone and implication of recent headlines. "
        + AGENT_JSON_INSTRUCTIONS
    )
    headlines = market_data.get("news_headlines", [])
    user = f"Ticker: {ticker}\nHeadlines: {json.dumps(headlines)}"
    return await _call_agent(system, user, "sentiment_agent")


async def filings_agent(ticker: str):
    """RAG-grounded agent: retrieves real snippet(s) then reasons over them."""
    retrieved = retrieve(query=f"{ticker} earnings risk margin growth", ticker=ticker, top_k=2)
    context = "\n\n".join([f"[{r['source']}]: {r['snippet']}" for r in retrieved])
    system = (
        "You are a fundamentals/filings analysis agent. Base your reasoning ONLY on the "
        "retrieved filing excerpts provided. Cite the source filename in 'sources'. "
        + AGENT_JSON_INSTRUCTIONS
    )
    user = f"Ticker: {ticker}\nRetrieved filing excerpts:\n{context}"
    result = await _call_agent(system, user, "filings_agent")
    result["retrieved_context"] = retrieved  # kept for UI attribution display
    return result


async def synthesizer_agent(ticker: str, agent_outputs: list, user_profile: dict):
    system = (
        "You are the synthesis agent in a multi-agent financial intelligence system. "
        "You receive outputs from 3 specialist agents (momentum, sentiment, filings) plus "
        "a user's risk profile. Produce ONE final recommendation personalized to this user. "
        "If any specialist output is marked degraded, explicitly say so in your reasoning "
        "rather than ignoring it. If two agents conflict, state the conflict and resolve it "
        "conservatively. Respond with ONLY JSON: "
        '{"recommendation": "BUY|SELL|HOLD|AVOID", "confidence": <0-100>, '
        '"reasoning": "<4-6 sentences, reference specific agent findings and how the user '
        'risk profile shaped this>", "flags": ["<any degraded/conflicting inputs noted>"]}'
    )
    user = (
        f"Ticker: {ticker}\n"
        f"User profile: {json.dumps(user_profile)}\n"
        f"Specialist agent outputs: {json.dumps(agent_outputs)}"
    )
    return await _call_agent(system, user, "synthesizer_agent")


async def run_pipeline(ticker: str, market_data: dict, user_profile: dict):
    """Runs the 3 specialists in parallel, then synthesizes. (non-streaming version)"""
    momentum, sentiment, filings = await asyncio.gather(
        momentum_agent(ticker, market_data),
        sentiment_agent(ticker, market_data),
        filings_agent(ticker),
    )
    specialist_outputs = [momentum, sentiment, filings]
    synthesis = await synthesizer_agent(ticker, specialist_outputs, user_profile)
    return {
        "ticker": ticker,
        "specialists": specialist_outputs,
        "synthesis": synthesis,
    }


async def stream_pipeline(ticker: str, market_data: dict, user_profile: dict):
    """
    Async generator version: yields a dict the moment each event happens,
    instead of waiting for the whole pipeline. Powers the live SSE endpoint
    so the UI can show each agent reporting in as it finishes, not a single
    spinner for the whole run.
    """
    yield {"type": "agents_started", "agents": ["momentum_agent", "sentiment_agent", "filings_agent"]}

    tasks = {
        asyncio.ensure_future(momentum_agent(ticker, market_data)): "momentum_agent",
        asyncio.ensure_future(sentiment_agent(ticker, market_data)): "sentiment_agent",
        asyncio.ensure_future(filings_agent(ticker)): "filings_agent",
    }
    specialist_outputs = []
    for coro in asyncio.as_completed(list(tasks.keys())):
        result = await coro
        specialist_outputs.append(result)
        yield {"type": "agent_result", "result": result}

    yield {"type": "synthesizing"}
    synthesis = await synthesizer_agent(ticker, specialist_outputs, user_profile)
    yield {"type": "synthesis", "result": synthesis}
    yield {"type": "done"}
