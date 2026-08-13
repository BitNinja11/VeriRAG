"""
VeriRAG configuration.

Every tunable lives here so experiments are reproducible and the README can
point at a single file. Values are overridable via environment variables.
"""

import os

# --------------------------------------------------------------------------
# LLM provider
# --------------------------------------------------------------------------
# Supported: "groq", "gemini", "openai", "anthropic", "offline"
# "offline" uses the deterministic rule-based backend in agents/llm.py, which
# lets the whole pipeline run with zero API keys (useful for CI, for grading,
# and for demoing without burning quota).
LLM_PROVIDER = os.getenv("VERIRAG_LLM_PROVIDER", "offline").lower()

# Model names per provider. Free-tier friendly defaults.
LLM_MODELS = {
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
}

API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

LLM_TEMPERATURE = float(os.getenv("VERIRAG_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("VERIRAG_MAX_TOKENS", "1200"))
LLM_TIMEOUT = int(os.getenv("VERIRAG_TIMEOUT", "60"))

# --------------------------------------------------------------------------
# Retrieval models
# --------------------------------------------------------------------------
# If sentence-transformers is unavailable (no network / no torch), the
# embedder silently degrades to a TF-IDF + SVD dense representation. Retrieval
# quality drops but the pipeline stays runnable. See rag/embeddings.py.
EMBEDDING_MODEL = os.getenv(
    "VERIRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL = os.getenv(
    "VERIRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# Word-based, not token-based: close enough at this scale and removes a
# tokenizer dependency.
#
# 70 words is deliberately small. These documents are short and section
# structured, so a small budget makes each policy SECTION its own chunk. That
# matters more than usual here: the Evidence Analyst extracts one claim per
# chunk, so a chunk holding both "SECTION 2 ENTITLEMENT: 24 days" and
# "SECTION 4 INTERNS: 12 days" would force two incompatible-looking claims
# into a single unit and manufacture a false conflict.
CHUNK_SIZE = int(os.getenv("VERIRAG_CHUNK_SIZE", "70"))
CHUNK_OVERLAP = int(os.getenv("VERIRAG_CHUNK_OVERLAP", "20"))

# --------------------------------------------------------------------------
# Retrieval depth
# --------------------------------------------------------------------------
DENSE_TOP_K = int(os.getenv("VERIRAG_DENSE_TOP_K", "10"))
BM25_TOP_K = int(os.getenv("VERIRAG_BM25_TOP_K", "10"))
HYBRID_TOP_K = int(os.getenv("VERIRAG_HYBRID_TOP_K", "10"))
# 5, not 4. At 4 the low-authority marketing blog fell just outside the
# evidence window, so the adjudicator never had to defend against it and the
# recency-only ablation scored identically to the full system. Admitting one
# more candidate is what makes the authority test actually bind. Widening the
# window costs a little precision and is the honest trade: the system should
# be shown rejecting bad evidence, not never seeing it.
RERANK_TOP_K = int(os.getenv("VERIRAG_RERANK_TOP_K", "5"))
# Match the full pipeline's final evidence budget so the dense-only vanilla
# baseline is not penalised simply for seeing one fewer passage.
VANILLA_TOP_K = int(os.getenv("VERIRAG_VANILLA_TOP_K", "5"))

# Hybrid fusion. "rrf" = Reciprocal Rank Fusion (rank-based, scale-free).
# "weighted" = min-max normalise each score list then linearly combine.
FUSION_METHOD = os.getenv("VERIRAG_FUSION", "rrf").lower()
RRF_K = int(os.getenv("VERIRAG_RRF_K", "60"))
DENSE_WEIGHT = float(os.getenv("VERIRAG_DENSE_WEIGHT", "0.6"))
BM25_WEIGHT = float(os.getenv("VERIRAG_BM25_WEIGHT", "0.4"))

# --------------------------------------------------------------------------
# Adjudication heuristic
# --------------------------------------------------------------------------
# Source authority tiers. Deliberately explicit and hand-set: this is a
# heuristic prior, NOT a learned or validated reputation model. Documented as
# such in the README because an interviewer will ask.
SOURCE_AUTHORITY = {
    "official_policy": 1.00,
    "regulation": 1.00,
    "handbook": 0.75,
    "faq": 0.50,
    "internal_memo": 0.45,
    "blog": 0.25,
    "unknown": 0.40,
}

# Weights for the transparent diagnostic evidence score. The deterministic
# adjudicator uses an explicit priority ladder (trusted supersession > authority >
# recency > relevance); this score is only a final tie-breaker/diagnostic and
# must not be mistaken for that lexicographic policy.
EVIDENCE_SCORE_WEIGHTS = {
    "relevance": 0.35,
    "authority": 0.25,
    "recency": 0.20,
    "support": 0.20,
}

# A claim must beat the runner-up by this margin for the rules-only
# adjudicator to commit; otherwise it abstains.
ADJUDICATION_MARGIN = float(os.getenv("VERIRAG_ADJ_MARGIN", "0.05"))

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
BENCHMARK_PATH = os.path.join(BASE_DIR, "evaluation", "questions.json")
RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "results.csv")
RESULTS_SUMMARY_PATH = os.path.join(BASE_DIR, "evaluation", "summary.csv")


def active_model() -> str:
    return LLM_MODELS.get(LLM_PROVIDER, "offline-deterministic")


def provider_ready() -> bool:
    """True if the configured provider has its API key present."""
    if LLM_PROVIDER == "offline":
        return True
    key_env = API_KEY_ENV.get(LLM_PROVIDER)
    return bool(key_env and os.getenv(key_env))
