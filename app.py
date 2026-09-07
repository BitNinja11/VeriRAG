"""VeriRAG Streamlit demo.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from agents.adjudicator import evidence_scores  # noqa: E402
from agents.llm import LLMClient, LLMError  # noqa: E402
from pipeline.vanilla_rag import VanillaRAG  # noqa: E402
from pipeline.verirag import VeriRAG  # noqa: E402
from visualization.evidence_graph import (  # noqa: E402
    build_evidence_graph,
    graph_summary,
    render_matplotlib,
)

st.set_page_config(page_title="VeriRAG", page_icon="🔍", layout="wide")

RELATION_STYLE = {
    "CONTRADICTION": ("🔴", "Contradiction"),
    "SUPPORT": ("🟢", "Support"),
    "DIFFERENT_SCOPE": ("🟡", "Different scope"),
    "IRRELEVANT": ("⚪", "Irrelevant"),
}

EXAMPLES = [
    "How many paid annual leave days do full-time employees get?",
    "What is the refund window for a software licence?",
    "How many leave days do interns get?",
    "How many unused leave days can be carried forward?",
    "What is the company's parental leave entitlement?",
    "What is the minimum password length?",
]


@st.cache_resource(show_spinner="Building index...")
def load_system(provider: str):
    client = LLMClient(provider)
    system = VeriRAG(client=client)
    baseline = VanillaRAG(system.retriever, client)
    return system, baseline, client


def main() -> None:
    st.title("VeriRAG")
    st.caption(
        "Conflict-aware retrieval-augmented generation. Detects contradictory "
        "sources, adjudicates between them, and shows its reasoning."
    )

    with st.sidebar:
        st.header("Configuration")
        provider = st.selectbox(
            "LLM provider",
            ["offline", "groq", "gemini", "openai", "anthropic"],
            index=["offline", "groq", "gemini", "openai", "anthropic"].index(
                config.LLM_PROVIDER
            )
            if config.LLM_PROVIDER
            in ["offline", "groq", "gemini", "openai", "anthropic"]
            else 0,
            help=(
                "'offline' runs the deterministic rule-based backend with no "
                "API key. It is also the rules-only control in the ablation."
            ),
        )
        show_baseline = st.checkbox("Compare against vanilla RAG", value=True)
        show_graph = st.checkbox("Show evidence graph", value=True)
        show_debug = st.checkbox("Show retrieval debug", value=False)

    try:
        system, baseline, client = load_system(provider)
    except LLMError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.divider()
        st.caption(
            f"**Corpus** {len(system.documents)} docs → {len(system.chunks)} chunks  \n"
            f"**Embedder** `{system.embedder.backend}`  \n"
            f"**Reranker** `{system.reranker.backend}`  \n"
            f"**Model** `{client.model}`"
        )
        if system.embedder.is_degraded:
            st.warning(
                "sentence-transformers unavailable; using TF-IDF+SVD. "
                "Treat results as a separate degraded-backend condition; "
                "they are not a guaranteed lower bound on transformer retrieval.",
                icon="⚠️",
            )

    st.subheader("Ask a question")
    choice = st.selectbox("Example questions", ["(type your own)"] + EXAMPLES)
    default = "" if choice == "(type your own)" else choice
    question = st.text_input("Question", value=default, key="q")

    if not st.button("Ask", type="primary") or not question.strip():
        st.info("Pick an example or type a question, then press Ask.")
        return

    with st.spinner("Running pipeline..."):
        state = system.run(question)

    conflict = any(c.is_conflict for c in state.conflicts)

    col1, col2, col3 = st.columns(3)
    col1.metric("Conflict detected", "Yes" if conflict else "No")
    col2.metric("Adjudicator", "Invoked" if state.adjudicator_invoked else "Skipped")
    col3.metric("Latency", f"{state.latency:.2f}s")

    st.subheader("Answer")
    if conflict:
        st.warning("Sources disagreed. The answer below reflects adjudication.")
    st.markdown(f"**{state.answer}**")
    if state.citations:
        st.caption("Cited: " + " · ".join(state.citations))

    if state.adjudication is not None:
        st.subheader("Adjudication")
        if state.adjudication.resolved:
            st.success(
                f"Deciding criterion: **"
                f"{state.adjudication.criterion.replace('_', ' ')}**"
            )
        else:
            st.error("Unresolved — the system abstained rather than guess.")
        st.write(state.adjudication.reason)

    st.subheader("Extracted claims")
    scores = evidence_scores(state.evidence)
    if state.adjudication is not None and state.adjudication.scores:
        # The adjudicator scores only the actual conflict candidates so an
        # unrelated retrieved passage cannot change recency normalisation.
        # Show those exact scores in the UI rather than recomputing a different
        # number over the full evidence list.
        scores.update(state.adjudication.scores)
    for ev in state.evidence:
        selected = (
            state.adjudication is not None
            and state.adjudication.selected_claim_id == ev.claim_id
        )
        icon = "✅" if selected else ("📄" if ev.supports_question else "➖")
        with st.expander(
            f"{icon} [{ev.claim_id}] {ev.source} — {ev.value or 'no value'}"
            f"  ·  score {scores.get(ev.claim_id, 0):.3f}",
            expanded=selected,
        ):
            left, right = st.columns([2, 1])
            left.write(ev.claim)
            if ev.quote:
                left.caption(f"Quote: “{ev.quote}”")
            right.write(
                f"**Type** {ev.source_type}  \n"
                f"**Date** {ev.date or 'n/a'}  \n"
                f"**Scope** {ev.scope}  \n"
                f"**Property** {ev.property}  \n"
                f"**Supersedes** {ev.supersedes_previous}"
            )

    relations = [c for c in state.conflicts if c.relationship != "IRRELEVANT"]
    if relations:
        st.subheader("Claim relationships")
        for rel in relations:
            icon, name = RELATION_STYLE.get(rel.relationship, ("⚪", rel.relationship))
            st.write(f"{icon} **{rel.claim_ids}** {name} — {rel.reason}")

    if show_graph:
        st.subheader("Evidence graph")
        graph = build_evidence_graph(
            question, state.evidence, state.conflicts, state.adjudication
        )
        summary = graph_summary(graph)
        st.caption(
            f"{summary['nodes']} nodes · {summary['edges']} edges · "
            f"{summary['contradictions']} contradictions · "
            f"{summary['different_scope']} scope separations"
        )
        st.pyplot(render_matplotlib(graph))

    if show_baseline:
        st.subheader("Vanilla RAG baseline")
        with st.spinner("Running baseline..."):
            base = baseline.run(question)
        st.info(base.answer)
        st.caption(
            "Dense retrieval only with the same final context budget; no claim "
            "extraction, authority model, or conflict handling."
        )

    if show_debug:
        st.subheader("Retrieval debug")
        st.write("**Hybrid candidates (pre-rerank)**")
        st.dataframe(
            [
                {
                    "chunk": r.chunk.chunk_id,
                    "fused": round(r.score, 4),
                    "arm": r.retrieval_source,
                    "dense_rank": r.dense_rank,
                    "bm25_rank": r.bm25_rank,
                }
                for r in state.retrieved_chunks
            ],
            use_container_width=True,
        )
        st.write(f"**After rerank (`{system.reranker.backend}`)**")
        st.dataframe(
            [
                {
                    "chunk": r.chunk.chunk_id,
                    "rerank_score": round(r.rerank_score, 4),
                }
                for r in state.reranked_chunks
            ],
            use_container_width=True,
        )
        st.write("**Stage timings (s)**")
        st.json(state.stage_timings)


if __name__ == "__main__":
    main()
