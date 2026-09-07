# VeriRAG Code Audit and Repair Report

This file records the audit performed on the supplied project and the changes in
this repaired version. The guiding rule was conservative: preserve working
architecture, fix demonstrable bugs or misleading measurements, and add tests
before claiming an improvement.

## 1. Baseline reproduced before changes

The original deterministic benchmark was reproducible:

- Vanilla RAG answer accuracy: **0.250**
- VeriRAG answer accuracy: **0.906**
- no-reranker ablation: **0.906**
- no-adjudicator ablation: **0.750**
- recency-only ablation: **0.688**

So the project did not have fabricated headline numbers. The audit focused on
whether those numbers measured what their labels implied and whether any
correct answers were accidental.

## 2. Concrete failures found

### 2.1 Remote-work property misclassification

Question: `How many days per month may employees work remotely?`

The offline property classifier saw `per month` and mapped the claim to leave
accrual before considering the remote-work semantics. It could return the
nearby `2 days per completed month` leave-accrual statement instead of
`8 days per month` remote work.

**Fix:** property inference now supports multiple plausible properties and uses
question/property agreement when selecting the answer sentence. `per month`
alone is no longer treated as leave accrual.

### 2.2 Refund window vs refund processing

Question: `How many days does refund processing take?`

The old rules mapped any mention of `refund` to the request-window property and
could answer `30 days` instead of `3 to 5 business days`.

**Fix:** separate `refund_window_days` and `refund_processing_days`; use chunk
context when a terse sentence says only `Processing takes ...`; preserve ranges
rather than collapsing them to one endpoint.

### 2.3 Categorical values were unsupported

Question: `On which day is building maintenance scheduled?`

The old value extractor centered on number + unit patterns and could not return
`first Saturday`.

**Fix:** categorical weekday extraction for schedule properties.

### 2.4 Password-rotation result was correct by accident

The sentence contains two quantities:

`Passwords must be at least 14 characters and must be rotated every 180 days.`

For the rotation question, the structured claim could internally record
`14 characters` while the final generated sentence still contained `180 days`,
allowing the benchmark to mark the answer correct. This hid a real extraction
bug.

**Fix:** compound sentences can expose multiple properties; value extraction is
property-aware (`characters` for minimum length, `days` for rotation).

## 3. Conflict/adjudication bugs fixed

### 3.1 Weighted score did not implement the documented priority order

The documentation described a lexicographic policy:

`trusted explicit supersession > authority > recency > relevance`

but the deterministic implementation primarily ranked a weighted score. Those
are not equivalent.

**Fix:** deterministic adjudication now implements an explicit priority ladder.
The weighted evidence score remains visible only as a diagnostic/final
low-level tie-breaker.

### 3.2 Untrusted source could self-declare supersession

Treating `supersedes_previous=True` as an unconditional bonus allows a blog to
claim it supersedes an official policy and potentially gain an inappropriate
advantage.

**Fix:** an explicit supersession signal is decisive only when it comes from a
source at least as authoritative as the competing tier.

### 3.3 Passive supersession wording reversed document roles

Metadata such as `Superseded by the 2026 revision` describes the old document.
The earlier regex could interpret the word `Superseded` itself as evidence that
the old document supersedes another.

**Fix:** distinguish active supersession/replacement language from passive
`superseded by` / `replaced by` wording.

### 3.4 Document metadata was not always available to the extractor

Supersession can be declared in document frontmatter rather than repeated in
every chunk.

**Fix:** propagate `authority_note` metadata into evidence extraction for all
chunks.

### 3.5 Overlapping chunks created fake corroboration

Two overlapping chunks from one document are not independent sources, but they
could increase support counts and create duplicate claims.

**Fix:** de-duplicate same-document supporting claims with the same
property/scope/value and exclude same-document chunks from independent-support
scoring.

### 3.6 Agreeing evidence could be marked rejected

In a conflict component with two agreeing claims and one contradictory claim,
rejecting every non-winner in the connected component incorrectly labels the
second agreeing claim as rejected.

**Fix:** rejected IDs are derived from direct `CONTRADICTION` edges touching the
winner.

### 3.7 Distractors affected recency normalization

The diagnostic recency score was calculated over all retrieved evidence, so an
unrelated very old/new passage could change score normalization for a conflict.

**Fix:** score the actual conflict candidates during adjudication.

## 4. Baseline fairness fixes

The original offline vanilla generator returned the first numeric quantity it
encountered in the top retrieved passages. That is weaker than a reasonable
extractive RAG control and exaggerated the headline gap.

The repaired baseline still has no hybrid conflict handling, no structured
claim extraction, no authority prior, no pairwise conflict detector, and no
adjudicator. It now does only generic question-aware sentence selection over
dense-retrieved chunks, with simple token normalization and a generic
answer-shape preference.

**Effect:** vanilla answer accuracy rises from **0.250 to 0.469**. This makes the
comparison stricter rather than flattering VeriRAG.

### 4.1 Equal final-context budget

The original control retrieved four passages while the full pipeline retained
five after reranking. That difference did not change vanilla answer accuracy,
but it unnecessarily lowered the control's retrieval-hit metric.

**Fix:** both systems now expose the same final context budget of five passages.
Vanilla retrieval hit rises to **1.000** while answer accuracy remains **0.469**.
The choice of five is explicitly benchmark-aware: on this tiny planted corpus it
ensures intended conflict/distractor evidence can enter the evaluation window,
and it should be retuned on external data rather than treated as universal.

## 5. Evaluation bugs fixed

### 5.1 Hidden full-pipeline call inside ablations

The ablation runner contained a call equivalent to `type(vr.run(question))`.
It executed the full pipeline, discarded the result, and then ran the ablated
pipeline. In a real-provider run this could double API calls/cost and corrupt
latency/call-count measurements.

**Fix:** remove the hidden run and instrument the ablated stages directly.

### 5.2 Misleading abstention headline

The old `abstention accuracy` counted answerable questions as successes whenever
the model simply did not abstain. Because most benchmark items are answerable,
a system that never abstains could still score about 84%.

**Fix:** report:

- **abstention recall** on questions that truly require abstention; and
- **false-abstention rate** on answerable questions.

The legacy aggregate field is retained only for CSV/backward compatibility and
is no longer a headline metric.

### 5.3 Citation hit rate was labelled too strongly

The old citation metric passed if *any* cited source was a gold source. A system
could cite the correct source plus several wrong sources and still receive full
credit.

**Fix:** add **primary citation accuracy** for the source supporting the asserted
answer, while retaining the looser metric under the explicit name
**gold-citation hit rate**. Rejected sources may still be cited transparently in
a conflict explanation.

### 5.4 Retrieval/citation artifacts in the vanilla control

The vanilla pipeline does no structured evidence extraction, but the evaluator
expects an evidence-shaped field. Missing that field could make retrieval look
like a total failure; conversely, treating every retrieved chunk as a citation
could make citation quality look artificially strong.

**Fix:** expose retrieved chunks as pseudo-evidence for retrieval scoring, but
only count citations actually emitted in `[n]` form.

### 5.5 Existing primary-assertion scoring retained

The project already needed to distinguish the asserted answer from a disclosure
such as `Sources disagreed: the old FAQ says 18 days`. Correctness continues to
score the primary assertion rather than every value mentioned in the answer.

### 5.6 Range answers could receive endpoint-only credit

A numeric range such as `3 to 5 business days` could be scored too loosely if a
wrong answer shared only one endpoint, for example `5 to 7 days` or isolated
`5 days`.

**Fix:** range-valued questions now require the full expected range. Equivalent
forms such as `3-5`, `3–5`, and `between 3 and 5` normalize to the same range.
The corresponding benchmark item was also tightened so isolated endpoints are
no longer accepted answers.

### 5.7 Recency-only ablation changed more than the selection rule

The earlier recency-only helper could reject every non-winner in a connected
conflict component, while the full adjudicator rejects only claims directly in
contradiction with the winner. That made the ablation differ in output
semantics as well as its trust rule.

**Fix:** recency-only now reuses the same direct-contradiction rejection logic;
only the winner-selection rule is changed to recency.

## 6. Robustness and portability fixes

- Prompt-injection instructions embedded in retrieved text are explicitly
  treated as untrusted data in evidence-analysis, conflict-detection,
  adjudication, and generation prompts.
- Real-model boolean fields such as the string `"false"` are parsed safely
  rather than relying on Python truthiness.
- Confidence values are type-checked and clamped to `[0, 1]`.
- A purported verbatim quote from an LLM is not displayed if it is absent from
  the retrieved chunk.
- TF-IDF fallback supports tiny corpora where truncated SVD cannot be fit.
- Lexical reranking ignores generic question words.
- Evidence-graph edge labels now use the detector's actual canonical relation
  names.
- CSV output uses deterministic Unix newlines.
- Project packaging is flattened so the repository itself is the archive root,
  rather than a ZIP nested inside another ZIP.

## 7. Final regression result

Final deterministic/offline benchmark after all repairs:

| System | Overall | clean | contradictory | outdated | noisy |
|---|---:|---:|---:|---:|---:|
| Vanilla RAG | 0.469 | 1.000 | 0.250 | 0.250 | 0.375 |
| **VeriRAG** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| − reranker | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| − adjudicator | 0.719 | 1.000 | 0.500 | 0.375 | 1.000 |
| recency only | 0.719 | 1.000 | 0.250 | 0.625 | 1.000 |

For full VeriRAG in this controlled benchmark:

- answer accuracy: **1.000**
- abstention recall: **1.000**
- false-abstention rate: **0.000**
- conflict precision: **1.000**
- conflict recall: **1.000**
- primary gold-source rate: **1.000**
- retrieval hit rate: **1.000**
- distractor-value rate: **0.000**

These are exact results on the included 32 items, not an estimate of external
performance.

## 8. Regression tests added

`tests/test_regressions.py` covers 14 targeted/end-to-end regressions:

1. remote work vs leave accrual;
2. refund processing vs refund window and range preservation;
3. password rotation value selection;
4. password minimum-length value selection;
5. categorical maintenance schedule values;
6. active vs passive supersession wording;
7. agreeing-claim rejection in a conflict component;
8. equivalent range/scope normalization;
9. low-authority self-declared supersession;
10. tiny-corpus embedding fallback;
11. abstention recall for a never-abstaining system;
12. full-range answer scoring;
13. gold-hit versus wrong-primary-citation separation; and
14. full 32-question offline benchmark invariants.

Run:

```bash
python -m unittest discover -s tests -v
```

## 9. Intentionally not changed

To avoid making the project worse through speculative redesign, the audit did
**not** replace the main architecture, corpus, retrieval fusion strategy,
provider interfaces, Streamlit UI, source-authority tiers, or optional
transformer/FAISS/cross-encoder paths. Those components were preserved unless a
specific failure was demonstrated.

## 10. Remaining risks

The main remaining weakness is external validity, not the repaired benchmark:

- the corpus and benchmark are synthetic;
- the deterministic ontology is tailored to policy-like properties;
- source authority remains hand-set;
- the top-five evidence window is benchmark-aware and should be retuned
  externally;
- vanilla-vs-full changes retrieval as well as conflict handling, so component
  ablations are the cleaner causal comparison;
- benchmark-designated gold sources represent canonical authority, so an
  agreeing secondary citation can still be scored non-gold;
- complex temporal intervals and multi-hop conflicts are not modeled;
- reranking provides no measured gain on twenty chunks;
- real LLM/provider integrations were not exercised during this offline audit;
- provider fallbacks can mix deterministic and LLM behavior, so any future
  provider benchmark should report backend/fallback rates explicitly.

The next meaningful improvement is therefore **external evaluation**, not more
rules added to make this same 32-question benchmark even easier.

---

# Round 2 audit (repository cleanup and reproducibility)

## 11. The BM25 "fallback" was not equivalent

The previous version shipped a hand-written BM25-Okapi implementation used
whenever `rank_bm25` was unavailable, described as a drop-in fallback.

Measured on this corpus, it is not. For `How many paid annual leave days do
full-time employees get?`:

```
rank_bm25 top-5 : [7, 2, 9, 12, 8]
fallback  top-5 : [7, 9, 2, 12, 8]
```

Chunks 2 and 9 swap. That ordering difference survives RRF fusion and
reranking and flips one benchmark item, which is why the no-reranker ablation
previously reported an exact 1.000 tie with the full pipeline. Under
`rank_bm25` the reranker shows a small but real effect (1.000 vs 0.969, with
the contradictory split at 0.875).

**Fix:** the hand-written fallback is deleted and `rank_bm25` is a hard
dependency. A fallback that silently changes published numbers is worse than a
missing dependency.

## 12. Results did not record the backend that produced them

Retrieval results are backend-dependent (see above), but `summary.csv` recorded
only scores. A reader could not tell whether a row came from
sentence-transformers or the TF-IDF fallback.

**Fix:** every summary row is stamped with `provider`, `model`,
`embedder_backend` and `reranker_backend`, and the evaluator prints a loud
warning when the transformer stack is missing.

## 13. No uncertainty on any reported number

Every headline was a bare point estimate on n=32.

**Fix:** percentile bootstrap 95% CIs (10k resamples, fixed seed), with a
Wilson interval for degenerate all-correct samples. A perfect score on 32 items
reports as `1.000 [0.893, 1.000]`.

## 14. The strongest baseline was missing

Vanilla RAG is dense-only with no conflict handling — a weak control. The
obvious objection to the whole architecture is "why not one strong LLM call
over the same passages with a conflict-aware prompt?", and nothing in the repo
answered it.

**Fix:** `pipeline/strong_baseline.py` implements that baseline generously
(same retrieval, same reranked top-5, full source metadata, same priority order
in the prompt). It requires a real provider and is not yet run; the README
states this as an open experiment rather than implying it was tested.

## 15. Dead surface area removed

- hand-written `_FallbackBM25` (~50 lines) — deleted, see §11
- `weighted` fusion mode — configurable, never used, never ablated
- `build_graph()` — returned a topology dict nothing consumed
- `abstention_accuracy` — superseded metric kept "for compatibility" with
  nothing

## 16. Packaging

Repository flattened from `verirag_fixed_final/verirag/` to the root; added
`LICENSE`, `pyproject.toml`, `.gitignore`, and a GitHub Actions workflow that
runs the regression suite and the offline benchmark on core dependencies only.
