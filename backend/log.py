"""
Minimal session/performance logger. Writes JSON lines to a local file.
Captures 3 metrics per session as required: total latency, per-agent latency,
and a portfolio concentration score.
"""
import json
import time
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "session_log.jsonl")


def portfolio_concentration_score(holdings: list) -> float:
    """Toy concentration metric: 1/N (lower = more diversified). Fine for demo purposes."""
    if not holdings:
        return 0.0
    return round(1 / len(holdings), 3)


def log_session(ticker: str, user_profile: dict, pipeline_result: dict, total_latency_ms: int):
    entry = {
        "timestamp": time.time(),
        "ticker": ticker,
        "user": user_profile.get("name"),
        "total_latency_ms": total_latency_ms,
        "agent_latencies_ms": {
            a["agent"]: a["latency_ms"] for a in pipeline_result["specialists"]
        },
        "portfolio_concentration_score": portfolio_concentration_score(
            user_profile.get("current_holdings", [])
        ),
        "final_recommendation": pipeline_result["synthesis"].get("recommendation"),
        "degraded_agents": [
            a["agent"] for a in pipeline_result["specialists"] if a.get("degraded")
        ],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_recent_logs(n=20):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()[-n:]
    return [json.loads(l) for l in lines]
