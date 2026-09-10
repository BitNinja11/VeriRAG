# Audit log

This file records defects found in VeriRAG after the first working version, and
what was changed. The rule I followed was conservative: keep architecture that
works, fix defects that are demonstrable, and add a test before claiming an
improvement.

Most of what follows is not "the code crashed". It is "the number measured
something other than what its label said", which is the harder class of bug and
the reason this file exists.

## 1. Baseline before any changes

The original deterministic benchmark reproduced:

| System | Answer accuracy |
|---|---:|
| Vanilla RAG | 0.250 |
| VeriRAG | 0.906 |
| no reranker | 0.906 |
| no adjudicator | 0.750 |
| recency only | 0.688 |

So the headline numbers were real. The audit was about whether they measured
what they claimed, and whether any correct answers were accidental.

## 2. Extraction defects

**2.1 Remote work read as leave accrual.** The property classifier saw
`per month` and mapped to leave accrual before considering remote-work
semantics, so `How many days per month may employees work remotely?` could
return the nearby `2 days per completed month` accrual figure.
*Fix:* property inference returns all plausible properties and uses
question/property agreement to select the answer sentence.

**2.2 Refund window vs refund processing.** Any mention of `refund` mapped to
the request-window property, so `How many days does refund processing take?`
could answer `30 days`.
*Fix:* separate `refund_window_days` and `refund_processing_days`, use chunk
context when a sentence says only `Processing takes ...`, and preserve ranges
instead of collapsing them to an endpoint.

**2.3 Categorical values unsupported.** The extractor was built around
number+unit patterns and could not return `first Saturday`.
*Fix:* weekday extraction for schedule properties.

**2.4 Password rotation was correct by accident.** The sentence
`Passwords must be at least 14 characters and must be rotated every 180 days`
contains two quantities. For the rotation question the structured claim
recorded `14 characters` while the generated sentence still contained
`180 days`, so the benchmark marked it correct and hid the extraction bug.
*Fix:* compound sentences expose multiple properties; value extraction filters
by the unit the property implies.

**2.5 Intern entitlement was correct by accident (same class as 2.4).** In
`Interns engaged for a period of six months or less are entitled to 12 days of
paid leave`, the eligibility window precedes the entitlement, so taking the
first quantity by position produced `6 months`. The answer still scored correct
because the generator quotes the whole sentence. Found by tracing extraction
output rather than by a failing test, which is why 2.4's fix had not caught it.
*Fix:* a `_DAY_VALUED_PROPERTIES` set, so properties whose answer is always a
day count filter to day-valued quantities regardless of position.

## 3. Conflict and adjudication defects

**3.1 The weighted score did not implement the documented priority order.**
Documentation described a lexicographic policy (trusted supersession >
authority > recency > relevance); the implementation ranked a weighted sum.
Those are not equivalent.
*Fix:* deterministic adjudication implements the ladder directly. The weighted
score remains as a diagnostic and final tie-breaker only.

**3.2 An untrusted source could self-declare supersession.** Treating
`supersedes_previous` as an unconditional bonus lets a blog claim it supersedes
official policy.
*Fix:* supersession is decisive only from a source at least as authoritative as
its competitors.

**3.3 Passive wording reversed document roles.** `Superseded by the 2026
revision` describes the *old* document, but the regex matched the bare word
`superseded`.
*Fix:* negative lookahead distinguishes active supersession from passive
`superseded by` / `replaced by`.

**3.4 Document metadata did not reach every chunk.** Supersession can be
declared in frontmatter rather than repeated in each chunk.
*Fix:* propagate `authority_note` into extraction for all chunks.

**3.5 Overlapping chunks created false corroboration.** Two overlapping chunks
from one document are not independent sources but could raise support counts.
*Fix:* de-duplicate same-document supporting claims by property/scope/value and
exclude same-document chunks from independent-support scoring.

**3.6 Agreeing evidence could be marked rejected.** In a conflict component
with two agreeing claims and one contradictory claim, rejecting every non-winner
mislabels the second agreeing claim.
*Fix:* rejected ids derive from direct `CONTRADICTION` edges touching the winner.

**3.7 Distractors shifted recency normalisation.** The diagnostic recency score
was computed over all retrieved evidence, so an unrelated passage could change
normalisation for a conflict.
*Fix:* score only the claims actually under adjudication.

## 4. Baseline fairness

The original offline vanilla generator returned the first numeric quantity it
found, which is weaker than a reasonable extractive control and exaggerated the
headline gap.

The repaired baseline still has no hybrid retrieval, no structured claim
extraction, no authority prior, no conflict detector and no adjudicator. It
performs generic question-aware sentence selection over dense-retrieved chunks.
Vanilla accuracy rises from **0.250 to 0.469**, which makes the comparison
stricter rather than more flattering.

Both systems were also given the same final context budget of five passages.
The choice of five is benchmark-aware: on this corpus it ensures the intended
distractors enter the evaluation window, and it should be retuned on external
data rather than treated as universal.

## 5. Evaluation defects

**5.1 A hidden full-pipeline call inside the ablations.** The ablation runner
executed the full pipeline, discarded the result, then ran the ablated pipeline,
which would double API cost and corrupt latency and call counts.
*Fix:* removed; ablated stages are instrumented directly.

**5.2 A misleading abstention headline.** The old metric counted answerable
questions as successes whenever the system did not abstain, so a system that
never abstains scored about 84%.
*Fix:* report abstention recall on questions that require abstention, and
false-abstention rate on answerable ones, separately.

**5.3 Citation credit was too loose.** The metric passed if *any* cited source
was gold, so citing everything won.
*Fix:* add primary citation accuracy for the source supporting the asserted
answer; retain the looser measure under the explicit name gold-citation hit rate.

**5.4 Retrieval and citation artifacts in the control.** Missing evidence-shaped
output made retrieval look like total failure; counting every retrieved chunk as
a citation made citation quality look artificially strong.
*Fix:* expose retrieved chunks as pseudo-evidence for retrieval scoring, but
count only citations actually emitted as `[n]` markers.

**5.5 Correctness scored the whole answer.** A conflict-aware answer reads
`<answer>. Sources disagreed: X says 18, Y says 24.` A substring check over the
whole string finds the correct value inside the list of *rejected* claims. This
is how the recency-only ablation once tied the full system while actually
asserting the blog's 35 days.
*Fix:* correctness is judged on the primary assertion, cut at disclosure markers.

**5.6 Range answers could win on one endpoint.** `3 to 5 business days` could be
matched by `5 to 7 days` or a bare `5 days`.
*Fix:* range-valued questions require the full range; `3-5`, `3 to 5` and
`between 3 and 5` normalise to the same value.

**5.7 The recency-only ablation changed two things at once.** It rejected every
non-winner in a conflict component while the full adjudicator rejects only
direct contradictions, so output semantics differed as well as the trust rule.
*Fix:* the ablation reuses the same rejection logic; only winner selection changes.

**5.8 No uncertainty on any number.** Every headline was a point estimate on
n=32.
*Fix:* percentile bootstrap 95% intervals (10k resamples, fixed seed), with a
Wilson interval for degenerate all-correct samples. A perfect score on 32 items
reports as `1.000 [0.893, 1.000]`.

**5.9 The strongest baseline did not exist.** Vanilla is dense-only with no
conflict handling, which is a weak control. The obvious objection to the whole
architecture is "why not one strong LLM call over the same passages with a
conflict-aware prompt?", and nothing answered it.
*Fix:* `pipeline/strong_baseline.py` implements that baseline with the same
retrieval, the same reranked top-k, full source metadata and the same priority
order in the prompt. It requires a real provider and is reported as an open
experiment rather than implied to have been run.

## 6. A fallback that silently substituted a different system

**6.1 The BM25 "drop-in fallback" was not equivalent.** A hand-written
BM25-Okapi implementation ran whenever `rank_bm25` was missing. Measured on this
corpus it ranks chunks differently:

```
rank_bm25 top-5 : [7, 2, 9, 12, 8]
fallback  top-5 : [7, 9, 2, 12, 8]
```

That ordering difference survives fusion and reranking and flips one benchmark
item, which is why the no-reranker ablation previously reported an exact 1.000
tie. Under `rank_bm25` the reranker shows a small real effect (1.000 vs 0.969).
*Fix:* the fallback is deleted and `rank_bm25` is a hard dependency. A fallback
that silently changes published numbers is worse than a missing dependency.

**6.2 Results did not record the backend that produced them.** Retrieval results
are backend-dependent, but `summary.csv` recorded only scores.
*Fix:* every summary row is stamped with provider, model, embedder backend and
reranker backend, and the evaluator warns when the transformer stack is absent.

## 7. Provider integration defects

Found on the first run against a real provider. All three would have been
invisible without the reporting added in 7.2.

**7.1 No User-Agent, so every request was blocked.** Outbound requests set only
`Content-Type` and `Authorization`, so `urllib` supplied its default
`User-Agent: Python-urllib/3.x`. Providers behind Cloudflare reject that
fingerprint before the API sees it:

```
HTTP 403 from groq: error code: 1010
```

The body is plain text, not JSON, because no API produced it, which makes the
failure look like a bad key or a retired model.
*Fix:* `_post` sets a User-Agent for every provider, overridable via
`VERIRAG_USER_AGENT`.

**7.2 A run where every call failed reported a perfect score.** Each agent falls
back to deterministic logic on error so one bad response cannot kill a
32-question run. With every call failing, VeriRAG fell back at every stage and
scored **1.000**, identical to the offline control and indistinguishable from it.
The only visible symptom was `Mean LLM calls/query = 19`, which decomposes as
`5 extraction x 3 retries + 1 detection x 3 retries + 1 generation`.
*Fix:* `LLMClient` counts provider failures and prints the first immediately;
the evaluator reports a per-system backend mix distinguishing `rules` (offline
control, intended) from `rules-fallback` (provider failed); a fallback rate
above 50% triggers an explicit refusal to treat the run as a provider result.

**7.3 Rate limits were retried for 0.4s when the provider asked for 4s.** On a
free tier with a tokens-per-minute cap, `complete_json` slept
`0.4 * (attempt + 1)` while the 429 body said *"Please try again in 4.035s"*.
Every rate limit therefore exhausted its attempts in about a second and fell
back to rules, converting a transient, self-healing condition into permanent
measurement loss. Retrying was also in the wrong layer: `complete_json` covers
only the JSON stages, leaving the generator and vanilla baseline with no
rate-limit handling at all.
*Fix:* `RateLimitError` carries the delay the provider asked for, taken from the
`Retry-After` header or parsed from the body; rate-limit retries moved into
`complete()` so every caller benefits; a 429 that later succeeds is not counted
as a failure, one that defeats the retry budget is.

## 8. Robustness and portability

- Prompt-injection instructions inside retrieved text are treated as untrusted
  data in every model prompt.
- Model boolean fields such as the string `"false"` are parsed safely rather
  than relying on Python truthiness; confidences are clamped to `[0, 1]`.
- A purported verbatim quote is not displayed if it is absent from the chunk.
- The TF-IDF fallback works on corpora too small for truncated SVD.
- Lexical reranking ignores generic question words.
- Evidence-graph edge labels use the detector's canonical relation names.
- CSV output uses Unix line endings so diffs are stable.

## 9. Final results

Deterministic offline benchmark after all repairs, with `rank_bm25` and `faiss`
installed and the transformer backends absent:

| System | Overall | 95% CI | clean | contradictory | outdated | noisy |
|---|---:|:--:|---:|---:|---:|---:|
| Vanilla RAG | 0.469 | [0.31, 0.66] | 1.000 | 0.250 | 0.250 | 0.375 |
| **VeriRAG** | **1.000** | [0.89, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 |
| no reranker | 0.969 | [0.91, 1.00] | 1.000 | 0.875 | 1.000 | 1.000 |
| no adjudicator | 0.719 | [0.56, 0.88] | 1.000 | 0.500 | 0.375 | 1.000 |
| recency only | 0.719 | [0.56, 0.88] | 1.000 | 0.250 | 0.625 | 1.000 |

Also for the full system: abstention recall 1.000, false-abstention rate 0.000,
conflict precision and recall 1.000, primary gold-source rate 1.000, retrieval
hit rate 1.000, distractor-value rate 0.000.

These are exact results on 32 items, not an estimate of external performance.

## 10. Tests

`tests/test_regressions.py` covers 15 regressions, each pinned to a defect above:
remote work vs leave accrual; refund processing vs window and range
preservation; password rotation and minimum-length selection; intern entitlement
vs eligibility window; categorical schedule values; active vs passive
supersession; agreeing-claim rejection; range and scope normalisation;
low-authority self-declared supersession; tiny-corpus embedding fallback;
abstention recall for a never-abstaining system; full-range answer scoring;
gold-hit versus wrong-primary-citation separation; and the full 32-question
offline benchmark invariants.

## 11. What was not changed

To avoid making the project worse through speculative redesign, the audit did
not replace the architecture, the corpus, the fusion strategy, the provider
interfaces, the Streamlit UI, the authority tiers, or the optional
transformer/FAISS/cross-encoder paths. Those were preserved unless a specific
failure was demonstrated.

## 12. Remaining risks

The main weakness is external validity, not the repaired benchmark:

- the corpus and benchmark are synthetic and small;
- the deterministic ontology is written for policy-like properties in this
  corpus, so the offline result does not demonstrate generalisation;
- source authority is hand-set;
- the five-passage evidence window is benchmark-aware;
- vanilla-vs-full changes retrieval as well as conflict handling, so the
  component ablations are the cleaner causal comparison;
- complex temporal intervals and multi-hop conflicts are not modelled;
- reranking shows only a small effect at twenty chunks;
- the strong single-call baseline and the transformer backend row have not been
  run.

The next meaningful improvement is external evaluation, not more rules to make
this same 32-question benchmark easier.
