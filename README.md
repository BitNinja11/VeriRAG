# VeriRAG — Conflict-Aware Retrieval-Augmented Generation

![tests](https://github.com/BitNinja11/VeriRAG/actions/workflows/tests.yml/badge.svg)

Standard RAG asks **"which passages are relevant?"** and hands those passages to
a generator. That is not enough when the retrieved sources disagree. A system
also has to notice the disagreement, decide whether it is a real contradiction
or just two claims about different things, and justify which source it trusts.

VeriRAG adds that evidence-handling layer:

1. **Retrieve** candidates with dense + sparse (BM25) retrieval, fused by RRF.
2. **Rerank** the candidates with a cross-encoder.
3. **Extract** one structured claim from each passage.
4. **Compare** every claim pair as `SUPPORT`, `CONTRADICTION`,
   `DIFFERENT_SCOPE`, or `IRRELEVANT`.
5. **Adjudicate only genuine contradictions**, on an explicit priority ladder:
   trusted supersession → authority → recency → relevance.
6. **Generate** a cited answer from the surviving evidence, and expose the
   evidence graph so a human can audit why one source beat another.

The adjudicator is **conditionally routed**: if no `CONTRADICTION` edge exists,
that stage is skipped entirely rather than paying for a model call that has
nothing to decide.

---

## What the benchmark actually shows

The headline is the **ablation**, not the top-line score. Removing adjudication
is the cleanest isolation of the idea, because it changes one component and
holds retrieval fixed.

| Answer accuracy | Overall | 95% CI | clean | contradictory | outdated | noisy |
|---|---:|:--:|---:|---:|---:|---:|
| Vanilla RAG (dense-only) | 0.469 | [0.31, 0.66] | 1.000 | 0.250 | 0.250 | 0.375 |
| **VeriRAG (full)** | **1.000** | [0.89, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 |
| − reranker | 0.969 | [0.91, 1.00] | 1.000 | 0.875 | 1.000 | 1.000 |
| − adjudicator | 0.719 | [0.56, 0.88] | 1.000 | 0.500 | 0.375 | 1.000 |
| recency-only adjudicator | 0.719 | [0.56, 0.88] | 1.000 | 0.250 | 0.625 | 1.000 |

**Reading this honestly:**

- **Adjudication carries the result.** Removing it costs 28 points overall, and
  the damage lands exactly where predicted — conflicting (1.000 → 0.500) and
  outdated (1.000 → 0.375) questions.
- **Recency alone is not a trust model.** The corpus deliberately contains a
  marketing blog that is the *newest* document and factually wrong. A
  newest-wins policy collapses to 0.250 on the contradictory split. This is a
  **predicted failure, reproduced** — a stronger result than another win.
- **The 1.000 is a controlled-regression result, not a generalisation claim.**
  32 synthetic questions over 9 planted documents, with a deterministic
  extractor whose property ontology was written for this corpus. The
  bootstrap CI already says a perfect score on n=32 only supports a lower
  bound of 0.89.
- **The reranker's effect is small but now measurable** (1.000 → 0.969), where
  an earlier version of this repo reported it as exactly zero. See below.

Per-question output: `evaluation/results.csv`. Aggregates: `evaluation/summary.csv`.

---

## Backend conditions matter, and are recorded

Retrieval results depend on which backends are installed. This is not a
footnote — it changed a published number in this repo.

An earlier version shipped a hand-written BM25-Okapi fallback described as a
drop-in for `rank_bm25`. It is not equivalent: on this corpus it ranks chunks
differently (chunks 2 and 9 swap for the annual-leave query), and that
difference propagates through fusion and reranking far enough to flip a
benchmark item — which is why the no-reranker ablation used to read as a
perfect 1.000 tie. The fallback has been **deleted**; `rank_bm25` is now a hard
dependency.

To stop that class of error recurring, `summary.csv` now stamps every row with
the condition that produced it:

```
provider, model, embedder_backend, reranker_backend
```

and the evaluator prints a loud warning when the transformer stack is missing.
`embedder=tfidf-svd, reranker=lexical` is a **separate backend condition**, not
a lower bound. The table above was produced with `rank_bm25` + `faiss`; the
sentence-transformers row is the one open reproduction item (see below).

---

## Architecture

```text
                        USER QUESTION
                              |
              +---------------+---------------+
              |                               |
         Dense retrieval                 BM25 retrieval
              |                               |
              +---------------+---------------+
                              |
                    Reciprocal Rank Fusion
                              |
                           Reranker
                              |
                        top-5 evidence
                              |
                  Agent 1: EVIDENCE ANALYST
                 passage -> structured claim
                              |
                  Agent 2: CONFLICT DETECTOR
        SUPPORT / CONTRADICTION / DIFFERENT_SCOPE / IRRELEVANT
                              |
                     contradiction found?
                         /          \
                       NO            YES
                       |              |
                       |      Agent 3: ADJUDICATOR
                       |      trusted supersession > authority
                       |        > recency > relevance
                       |              |
                       +------+-------+
                              |
                       ANSWER GENERATOR
                              |
            Answer + citations + confidence + evidence graph
```

`DIFFERENT_SCOPE` is the label that stops the system inventing conflicts.
"Interns get 12 days" and "full-time employees get 24 days" differ numerically
but are jointly satisfiable, so scope disjunction is checked **before** value
comparison.

---

## Quick start

```bash
pip install -r requirements.txt

python -m unittest discover -s tests -v      # 14 regression tests
python -m evaluation.evaluate --ablation     # offline benchmark, no API key
streamlit run app.py                         # demo UI
```

### With a real LLM

```bash
export GROQ_API_KEY=...            # or GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
python -m evaluation.evaluate --provider groq --ablation --strong-baseline
```

Provider REST APIs are called directly via `urllib`; no SDKs required. Agent
outputs are schema-validated and fall back to deterministic logic on malformed
output or provider failure — so inspect the `extraction_backend`,
`detection_backend` and adjudication `backend` fields before reading a run as a
pure LLM result.

---

## Two open experiments

These are stated up front rather than buried, because they are the questions a
reviewer asks first.

**1. The strong single-call baseline (`--strong-baseline`).**
The obvious objection to any staged pipeline is: *why not hand five passages
plus their metadata to one capable model and tell it to prefer authoritative
sources?* `pipeline/strong_baseline.py` implements exactly that, deliberately
generously — same retrieval, same reranked top-5, same priority order in the
prompt, full source metadata. Only the staged machinery is removed. It requires
a real provider, so it is **not yet run**. Either outcome is a result: if the
staged pipeline wins, the architecture is justified against its strongest
objection; if it ties, the honest conclusion is that the value is auditability
rather than accuracy.

**2. The transformer backend row.**
The table above ran with `rank_bm25` + `faiss` but with `sentence-transformers`
unavailable (blocked network), so the embedder was TF-IDF+SVD and the reranker
lexical. Reproducing it with the full stack installed is a single command and
is the remaining verification step.

---

## Why the offline backend exists

`VERIRAG_LLM_PROVIDER=offline` (the default) is a deterministic **control
condition**, not an LLM and not a performance floor. It uses rule-based
question/property extraction, rule-based pairwise relation typing, and
deterministic adjudication from explicit metadata and authority priors.

That makes the project runnable with no API key and separates two questions
usually conflated:

1. Does the conflict-aware architecture work on the controlled benchmark?
2. Does a particular LLM improve on the deterministic implementation?

**The table above answers only the first.**

---

## Repository layout

```text
├── app.py                      # Streamlit demo
├── config.py                   # every tunable, in one file
├── data/documents/             # 9-document controlled corpus
├── rag/
│   ├── loader.py               # frontmatter/metadata parsing
│   ├── chunker.py              # paragraph-aware overlapping chunks
│   ├── embeddings.py           # sentence-transformers -> TF-IDF/SVD fallback
│   ├── vector_store.py         # FAISS -> numpy fallback
│   ├── bm25.py                 # rank_bm25 (hard dependency)
│   ├── hybrid_retriever.py     # RRF fusion
│   └── reranker.py             # cross-encoder -> lexical fallback
├── agents/
│   ├── llm.py                  # multi-provider client + JSON repair
│   ├── evidence_analyst.py     # structured claim extraction
│   ├── conflict_detector.py    # pairwise evidence relations
│   ├── adjudicator.py          # conflict resolution
│   └── generator.py            # grounded answer / vanilla generator
├── pipeline/
│   ├── verirag.py              # full conditional pipeline
│   ├── vanilla_rag.py          # dense-only baseline
│   └── strong_baseline.py      # single-call conflict-aware LLM baseline
├── evaluation/
│   ├── questions.json          # 32 benchmark questions
│   ├── metrics.py              # scoring + bootstrap CIs
│   └── evaluate.py             # harness + ablations
├── tests/test_regressions.py   # 14 regression tests
└── visualization/evidence_graph.py
```

---

## Evaluation definitions

Deterministic scoring, not an LLM judge — on a 32-item benchmark an LLM judge
adds a second source of error that is harder to audit than the thing measured.

- **Answer accuracy** — the expected value appears in the system's *primary
  assertion*, not merely inside its disclosure of a rejected source.
- **Abstention recall / false-abstention rate** — reported separately, so a
  system that never abstains cannot look good just because most questions are
  answerable.
- **Primary gold-source rate** vs **gold-citation hit rate** — kept separate
  because a conflict explanation legitimately cites rejected sources too.
- **Distractor-value rate** — counts a planted wrong value only when the answer
  is *also* wrong. Naming a value you rejected is transparency, not error.
- **95% CIs** — percentile bootstrap (10k resamples, fixed seed); Wilson bound
  for degenerate all-correct/all-wrong samples.

## Benchmark design

32 questions over the planted "Northwind" corpus, ground truth defined from the
corpus and its metadata rather than outside-world facts:

| Condition | n | Purpose |
|---|---:|---|
| `clean` | 8 | one relevant answer; no conflict expected |
| `contradictory` | 8 | authority must beat naive recency |
| `outdated` | 8 | recency and/or explicit supersession matter |
| `noisy` | 8 | distractors, incl. 5 unanswerable items requiring abstention |

---

## Limitations

1. **Synthetic, small corpus.** 9 documents, 20 chunks — enough to test
   mechanics, not generalisation.
2. **Small benchmark.** n=32; one item moves the score ~3 points. Hence CIs.
3. **Domain-oriented deterministic ontology.** The offline extractor recognises
   properties present in *this* policy corpus. Transparent and testable, but
   not a general semantic parser.
4. **Hand-set authority prior.** `official_policy > handbook > FAQ > blog` is a
   design assumption in `config.py`, not a learned trust model.
5. **Hand-tuned diagnostic evidence score.** Not calibrated probability.
6. **Evidence window is benchmark-aware.** Top-5 for both systems; five was
   retained because this corpus needs that depth to expose intended
   distractors. Retune on external data.
7. **Vanilla-vs-full is not a single-component ablation** — vanilla is
   dense-only. Use the no-adjudicator and recency-only rows for clean isolation.
8. **Pairwise conflict detection is O(n²).** Fine at top-5; larger evidence
   sets need pruning or clustering.
9. **Temporal/scope reasoning is shallow.** No interval arithmetic, no
   jurisdictional exceptions, no multi-hop revision chains.
10. **Provider integrations are not live-tested** in the offline audit.
11. **The system adjudicates evidence; it does not establish truth.** If every
    retrieved source is wrong, good adjudication still yields a wrong answer.

## Future work

- run the strong single-call baseline and the transformer backend row;
- evaluate on an external conflict benchmark rather than only this corpus;
- replace rule-based conflict typing with a calibrated NLI model and report
  per-backend precision/recall on pair typing;
- add query decomposition for multi-part questions;
- learn or externally verify source trust instead of hand-set tiers.

## Related work

- Cattan et al., *DRAGged into Conflicts: Detecting and Addressing Conflicting
  Sources in Search-Augmented LLMs* (2025, arXiv:2506.08500; COLM 2025).
- *From Facts to Conclusions: Integrating Deductive Reasoning in
  Retrieval-Augmented LLMs* (2025, arXiv:2512.16795).
- Chan et al., *ChatEval: Towards Better LLM-based Evaluators through
  Multi-Agent Debate* (ICLR 2024).

See `AUDIT.md` for the detailed before/after audit of this codebase, including
four evaluation metrics that were found to be inflating results and were
rewritten to be stricter.

## License

MIT — see `LICENSE`.
