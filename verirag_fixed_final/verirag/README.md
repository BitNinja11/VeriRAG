# VeriRAG — Conflict-Aware Retrieval-Augmented Generation

Standard RAG asks **which passages are relevant?** and then gives those passages
to a generator. That is not enough when retrieved sources disagree: a system
also needs to identify the disagreement, decide whether it is a real
contradiction, and justify which source it trusts.

VeriRAG adds that evidence-handling layer:

1. **Retrieve** candidate passages with dense + sparse retrieval.
2. **Rerank** the candidates.
3. **Extract** a structured claim from each passage.
4. **Compare** claims as `SUPPORT`, `CONTRADICTION`, `DIFFERENT_SCOPE`, or
   `IRRELEVANT`.
5. **Adjudicate only genuine contradictions** using an explicit priority
   ladder: trusted supersession → authority → recency → relevance.
6. **Generate** a cited answer from verified evidence and expose the evidence
   graph for inspection.

The project has two execution modes:

- real LLM providers for the agent stages; and
- a deterministic `offline` control that runs without an API key and makes the
  included benchmark reproducible.

> **Scope of the headline result:** the numbers below are from a deliberately
> small, synthetic, 32-question benchmark over 9 documents. They validate the
> implementation on this controlled corpus; they are **not** evidence of 100%
> accuracy on general RAG workloads.

---

## Reproducible benchmark results

Run:

```bash
python -m evaluation.evaluate --ablation
```

Final audited offline results:

| Metric | Vanilla RAG | **VeriRAG** |
|---|---:|---:|
| Answer accuracy | 0.469 | **1.000** |
| Abstention recall on unanswerable items | 0.000 | **1.000** |
| False-abstention rate on answerable items | 0.000 | **0.000** |
| Conflict-label accuracy | 0.500 | **1.000** |
| Primary gold-source rate | 0.556 | **1.000** |
| Gold-citation hit rate | 0.556 | **1.000** |
| Retrieval hit rate | 1.000 | **1.000** |
| Distractor-value rate | 0.250 | **0.000** |

### Ablation study

| Answer accuracy | Overall | clean | contradictory | outdated | noisy |
|---|---:|---:|---:|---:|---:|
| Vanilla RAG | 0.469 | 1.000 | 0.250 | 0.250 | 0.375 |
| **VeriRAG (full)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| − reranker | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| − adjudicator | 0.719 | 1.000 | 0.500 | 0.375 | 1.000 |
| recency-only adjudicator | 0.719 | 1.000 | 0.250 | 0.625 | 1.000 |

What the table supports — and what it does not:

- **Adjudication is useful on this benchmark.** Removing it drops overall
  answer accuracy from 1.000 to 0.719, with the largest damage on conflicting
  and outdated questions. This ablation is a cleaner isolation of conflict
  resolution than the vanilla-vs-full comparison, because vanilla also differs
  in retrieval strategy and evidence processing.
- **Recency alone is not a sufficient trust rule.** The planted corpus includes
  newer, low-authority content that conflicts with official policy; the
  recency-only ablation falls to 0.250 on the contradictory split.
- **The reranker has no measurable effect at this corpus size.** Removing it
  changes no answer-accuracy result. It remains in the architecture for larger
  retrieval sets, but this benchmark does not prove its value.
- **The 1.000 result is a controlled-regression result, not a generalization
  claim.** The deterministic extractor contains a transparent, hand-written
  property ontology suited to the included policy corpus.

The full per-question outputs are written to `evaluation/results.csv`; aggregate
results are written to `evaluation/summary.csv`.

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
                         top candidates
                              |
                           Reranker
                              |
                        top evidence
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

The adjudicator is conditionally routed. If no `CONTRADICTION` edge exists,
the pipeline skips that stage rather than paying for an unnecessary model call.

---

## Quick start

```bash
pip install -r requirements.txt

# Offline deterministic benchmark; no API key required
python -m evaluation.evaluate --ablation

# Regression tests
python -m unittest discover -s tests -v

# Demo UI
streamlit run app.py
```

### Using a real LLM

The project calls provider REST APIs directly; provider SDKs are not required.

```bash
export GROQ_API_KEY=...            # or GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
export VERIRAG_LLM_PROVIDER=groq   # groq | gemini | openai | anthropic | offline
python -m evaluation.evaluate --provider groq --ablation
```

The offline audit in this repository does **not** claim that every remote
provider/model combination has been integration-tested. Model/API behavior can
change independently of this code. Agent outputs are schema-validated and fall
back to deterministic logic on malformed model output or provider failure.
Inspect the `extraction_backend`, `detection_backend`, and adjudication `backend`
fields when evaluating a real provider so a fallback is not mistaken for a pure
LLM result.

---

## Why the offline backend exists

`VERIRAG_LLM_PROVIDER=offline` is a deterministic **control condition**, not an
LLM and not a guaranteed performance floor. It uses:

- rule-based question/property and value extraction;
- rule-based pairwise relation typing; and
- deterministic adjudication from explicit metadata and source-authority priors.

That makes the project runnable without network access and separates two
questions that are often conflated:

1. Does the conflict-aware architecture work on the controlled benchmark?
2. Does a particular LLM improve on the deterministic implementation?

The included headline table answers the first question only.

---

## Repository layout

```text
verirag/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── AUDIT.md
├── data/
│   └── documents/                 # 9-document controlled corpus
├── rag/
│   ├── loader.py                  # metadata/frontmatter parsing
│   ├── chunker.py                 # paragraph-aware overlapping chunks
│   ├── embeddings.py              # sentence-transformers -> TF-IDF/SVD fallback
│   ├── vector_store.py            # FAISS -> numpy fallback
│   ├── bm25.py                    # rank_bm25 -> built-in BM25 fallback
│   ├── hybrid_retriever.py        # RRF / weighted fusion
│   └── reranker.py                # cross-encoder -> lexical fallback
├── agents/
│   ├── llm.py                     # provider client + JSON repair
│   ├── evidence_analyst.py        # structured claim extraction
│   ├── conflict_detector.py       # pairwise evidence relations
│   ├── adjudicator.py             # conflict resolution
│   └── generator.py               # final grounded answer / vanilla generator
├── pipeline/
│   ├── verirag.py                 # full conditional pipeline
│   └── vanilla_rag.py             # dense-only baseline
├── evaluation/
│   ├── questions.json             # 32 benchmark questions
│   ├── metrics.py
│   ├── evaluate.py
│   ├── results.csv
│   └── summary.csv
├── tests/
│   └── test_regressions.py
└── visualization/
    └── evidence_graph.py
```

---

## What the audited version fixed

The original project was functional, but several issues either caused real
failures or made the benchmark look stronger/weaker for the wrong reason. The
current version fixes them without removing working architecture.

### Extraction and reasoning

- `remote work ... 8 days per month` is no longer mistaken for leave accrual.
- `refund processing takes 3 to 5 business days` is separated from the
  `30-day refund request window` and numeric ranges are preserved as ranges.
- compound claims such as `14 characters` + `180 days` select the quantity that
  belongs to the question's property.
- categorical values such as `first Saturday` can be extracted.
- document-level supersession metadata reaches every chunk; passive text such
  as `Superseded by ...` no longer makes the old document look like the
  superseding document.
- duplicate overlapping chunks from one document are not counted as independent
  corroboration.
- selecting one claim no longer marks an agreeing claim as rejected merely
  because both are connected to a third contradictory claim.
- a low-authority source cannot outrank an official source simply by claiming
  that it supersedes it.
- deterministic adjudication now implements the documented priority ladder
  directly rather than approximating it with a weighted sum.

### Baseline and evaluation

- the offline vanilla baseline now performs generic question-aware extractive
  sentence selection instead of returning the first quantity it sees. This
  raises the control from the original 0.250 to 0.469 and makes the comparison
  harder rather than easier.
- the ablation runner no longer executes a hidden full VeriRAG pass before each
  ablated pass; latency and LLM-call counts now describe the ablation itself.
- the old `abstention accuracy` headline was replaced with **abstention recall**
  plus **false-abstention rate**. A system that never abstains can no longer look
  good merely because most benchmark questions are answerable.
- citation reporting now distinguishes **primary citation accuracy** from the
  looser **gold-citation hit rate**.
- conflict precision and recall are reported separately from overall label
  accuracy.
- retrieval metrics for the vanilla baseline are based on what it actually
  retrieved; citations are based on what it actually cited.

### Robustness / portability

- prompt-injection instructions embedded in retrieved documents are explicitly
  treated as untrusted data in all model prompts.
- malformed LLM booleans/confidence values are normalized and unsafe fabricated
  verbatim quotes are not displayed as evidence.
- the TF-IDF fallback now works even on a one-document/tiny feature corpus where
  truncated SVD is mathematically invalid.
- lexical fallback reranking ignores common interrogative/functional words.
- evidence-graph relation names now match the detector's canonical labels.
- generated CSV files use stable Unix line endings.
- a regression test suite covers the discovered failures.

See `AUDIT.md` for the detailed before/after audit.

---

## Evaluation definitions

The benchmark uses deterministic scoring rather than an LLM judge.

- **Answer accuracy:** expected answer appears in the system's *primary
  assertion*, not merely inside its disclosure of a rejected source.
- **Abstention recall:** fraction of truly unanswerable benchmark items on which
  the system abstains.
- **False-abstention rate:** fraction of answerable items on which it abstains.
- **Conflict precision/recall:** standard detection metrics for expected
  contradictions.
- **Primary gold-source rate:** first citation supporting the asserted answer
  comes from a benchmark-designated canonical gold document. A factually
  agreeing secondary source can still count as non-gold by design.
- **Gold-citation hit rate:** at least one cited source is a gold document. This
  is deliberately kept separate because conflict explanations may also cite
  rejected sources.
- **Retrieval hit rate:** at least one gold document was retrieved.
- **Distractor-value rate:** the system's answer is wrong and contains a known
  planted distractor value. Mentioning a rejected distractor transparently in a
  correct conflict explanation is not counted as being misled.

---

## Benchmark design

The benchmark contains 32 questions across four conditions:

| Condition | n | Purpose |
|---|---:|---|
| `clean` | 8 | one relevant answer or agreeing evidence; no contradiction expected |
| `contradictory` | 8 | disagreement where source authority should beat naive recency |
| `outdated` | 8 | version conflicts where recency and/or explicit supersession matter |
| `noisy` | 8 | distractors plus 5 unanswerable questions requiring abstention |

Ground truth is defined from the included corpus and its explicit metadata, not
from outside-world facts.

---

## Important limitations

1. **Synthetic, small corpus.** Nine documents and twenty chunks are enough to
   test mechanics, not generalization.
2. **Small benchmark.** Thirty-two questions make individual errors move the
   percentage by several points.
3. **Domain-oriented deterministic ontology.** The offline extractor explicitly
   recognizes properties used in this policy corpus. It is transparent and
   testable, but it is not a general semantic parser.
4. **Hand-set authority prior.** `official_policy > handbook > FAQ > blog` is a
   design assumption in `config.py`, not a learned trust model.
5. **Hand-tuned diagnostic evidence score.** It is not calibrated probability.
6. **No measured reranker benefit here.** The ablation is exactly tied with the
   full pipeline on this tiny corpus.
7. **Evidence-window size is benchmark-aware.** The final context budget is five
   passages for both Vanilla and VeriRAG. Five was retained because this tiny
   planted-conflict corpus needs that depth to expose some intended distractors;
   it should be retuned on external data rather than treated as universal.
8. **Vanilla-vs-full is not a single-component ablation.** Vanilla is dense-only
   while VeriRAG uses hybrid retrieval plus structured evidence handling. Use the
   no-adjudicator and recency-only rows to isolate adjudication more cleanly.
9. **Pairwise conflict detection is O(n²).** Fine for the top-five evidence
   window; larger evidence sets need pruning or clustering.
10. **Temporal/scope reasoning is shallow.** Complex effective-date intervals,
    jurisdictional exceptions, and multi-hop revision chains are not modeled.
11. **Provider integrations were not live-tested in this offline audit.** A
    provider evaluation should report fallback usage and model/version details.
12. **The system adjudicates evidence; it does not establish truth.** If every
    retrieved source is wrong, a well-executed adjudication can still produce a
    wrong answer.

---

## Future work

- evaluate on an external conflict benchmark rather than only the planted
  Northwind corpus;
- add interval-aware temporal reasoning and richer scope constraints;
- replace/augment rule-based conflict typing with a calibrated NLI model;
- learn or externally verify source trust rather than relying only on hand-set
  authority tiers;
- add adversarial/poisoned-document evaluation;
- report real-provider results with model version, cost, latency, fallback rate,
  and repeated-run variance;
- add larger-corpus experiments where reranking can be meaningfully measured.

---

## Related work

The project is conceptually related to:

- **Cattan et al., _DRAGged into Conflicts: Detecting and Addressing Conflicting
  Sources in Search-Augmented LLMs_ (2025, arXiv:2506.08500; COLM 2025)** —
  conflict types and expected behavior under conflicting retrieved evidence.
- **_From Facts to Conclusions: Integrating Deductive Reasoning in
  Retrieval-Augmented LLMs_ (2025, arXiv:2512.16795)** — structured reasoning
  over retrieved evidence and conflict-aware synthesis.
- **Chan et al., _ChatEval: Towards Better LLM-based Evaluators through
  Multi-Agent Debate_ (ICLR 2024)** — relevant multi-agent evaluation context,
  though VeriRAG is a staged evidence pipeline rather than a claim that debate
  itself is superior.

The project contribution is the implementation and controlled evaluation of a
conditional claim-extraction → conflict-detection → adjudication workflow, not
a claim that adding more agents is intrinsically better.
