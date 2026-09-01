import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="Retail Financial Intelligence", layout="wide")
st.title("📊 Multi-Agent Financial Intelligence System")
st.caption("Momentum · Sentiment · Filings (RAG) agents → synthesized, personalized recommendation")

col_a, col_b = st.columns(2)
with col_a:
    tickers = requests.get(f"{API}/tickers").json()["tickers"]
    ticker = st.selectbox("Ticker", tickers)
with col_b:
    profiles = requests.get(f"{API}/profiles").json()
    profile_id = st.selectbox(
        "User Profile", list(profiles.keys()),
        format_func=lambda k: profiles[k]["name"]
    )

if st.button("Run Multi-Agent Analysis", type="primary"):
    with st.spinner("Dispatching parallel agents..."):
        resp = requests.get(f"{API}/analyze/{ticker}", params={"profile_id": profile_id})
    if resp.status_code != 200:
        st.error(resp.json())
    else:
        data = resp.json()
        pipeline = data["pipeline"]
        synthesis = pipeline["synthesis"]

        st.markdown("---")
        st.subheader(f"Final Recommendation for {ticker}")
        rec = synthesis.get("recommendation", "N/A")
        color = {"BUY": "green", "SELL": "red", "HOLD": "orange", "AVOID": "red"}.get(rec, "gray")
        st.markdown(f"### :{color}[{rec}]  —  confidence {synthesis.get('confidence', 0)}%")
        st.write(synthesis.get("reasoning", ""))
        if synthesis.get("flags"):
            for flag in synthesis["flags"]:
                st.warning(f"⚠️ {flag}")

        st.markdown("---")
        st.subheader("Agent Reasoning Trace")
        cols = st.columns(3)
        for i, agent in enumerate(pipeline["specialists"]):
            with cols[i]:
                badge = "🔴 DEGRADED" if agent.get("degraded") else "🟢 OK"
                st.markdown(f"**{agent['agent']}** — {badge}")
                st.markdown(f"Signal: `{agent.get('signal')}`  |  Confidence: {agent.get('confidence')}%")
                st.caption(agent.get("reasoning", ""))
                if agent.get("sources"):
                    st.caption(f"Sources: {', '.join(agent['sources'])}")
                if agent.get("retrieved_context"):
                    with st.expander("Retrieved filing excerpt"):
                        for r in agent["retrieved_context"]:
                            st.text(f"[{r['source']}]\n{r['snippet']}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Current Market Signal")
            st.json(data["market_data"])
        with c2:
            st.subheader("User Profile / Watchlist State")
            st.json(data["user_profile"])

        st.markdown("---")
        st.subheader("Session Performance Metrics")
        st.json(data["log_entry"])

st.markdown("---")
with st.expander("📜 Recent Session Logs"):
    logs = requests.get(f"{API}/logs").json()["logs"]
    st.json(logs)
