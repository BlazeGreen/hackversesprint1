"""
Lightweight retrieval layer. No vector DB - just TF-IDF style keyword overlap.
Good enough to demonstrate "RAG with visible attribution" in a 4-hour build.
Swap for real embeddings later if you have time left.
"""
import os
import re
from collections import Counter

FILINGS_DIR = os.path.join(os.path.dirname(__file__), "data", "filings")


def _load_docs():
    docs = {}
    for fname in os.listdir(FILINGS_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(FILINGS_DIR, fname)
            with open(path, "r") as f:
                docs[fname] = f.read()
    return docs


def _tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def retrieve(query: str, ticker: str = None, top_k: int = 1):
    """
    Returns list of {source, snippet, score} sorted by relevance.
    If ticker is given, biases toward filings whose filename contains it.
    """
    docs = _load_docs()
    query_tokens = Counter(_tokenize(query))

    scored = []
    for fname, text in docs.items():
        doc_tokens = Counter(_tokenize(text))
        overlap = sum((query_tokens & doc_tokens).values())
        # bonus if filename matches ticker
        if ticker and ticker.lower() in fname.lower():
            overlap += 5
        if overlap > 0 or (ticker and ticker.lower() in fname.lower()):
            # grab a representative snippet (first 300 chars) for citation display
            snippet = text.strip().replace("\n", " ")[:300] + "..."
            scored.append({"source": fname, "snippet": snippet, "score": overlap})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k] if scored else [{"source": "none", "snippet": "No matching filing found.", "score": 0}]
