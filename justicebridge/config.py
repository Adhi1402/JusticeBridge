"""
Central configuration — one place for every path, threshold, and backend
choice so nothing is hard-coded across the agent modules.

Paths point at the EXISTING corpus.json + chroma_db that were already built
at the repo root (E:\\qualcomm), so we don't re-embed anything. If you move
this package, only these two paths need to change.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Offline-first: the whole product's pitch is "runs without internet." Once
# the embedding model is cached (first run), there is no reason for
# HuggingFace/sentence-transformers to ever touch the network again — but by
# default they still call the Hub each run to check for updates, which is
# what was causing the repeated "downloading model" delay + the
# "unauthenticated requests to HF Hub" warning on every invocation.
# Forcing offline mode here makes every subsequent run load purely from
# ~/.cache/huggingface. If you ever need a NEW/uncached model, temporarily
# run with JB_HF_OFFLINE=0 once to let it download, then leave it unset.
# ---------------------------------------------------------------------------
if os.environ.get("JB_HF_OFFLINE", "1") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Paths — the package is self-contained (its own data/, pdfs/, chroma_db/) so
# the repo can be cloned and rebuilt standalone. Each source file resolves to
# the package-local copy if present, else the repo-root copy (back-compat with
# the original layout where these lived one level up).
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
PDF_DIR = PACKAGE_DIR / "pdfs"


def _first_existing(*candidates, default=None):
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return str(default if default is not None else candidates[0])


# Real indiacode source (used only to (re)build the corpus from Act PDFs).
INDIACODE_JSON = os.environ.get("JB_INDIACODE") or _first_existing(
    DATA_DIR / "indiacode.json",
    REPO_ROOT / "indiacode (3).json",
    default=DATA_DIR / "indiacode.json",
)
CORPUS_FILE = os.environ.get("JB_CORPUS", str(DATA_DIR / "corpus.json"))
VECTOR_DIR = os.environ.get("JB_VECTOR_DIR", os.environ.get("JB_CHROMA", str(PACKAGE_DIR / "chroma_db")))
EMBEDDING_MODEL = os.environ.get("JB_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The catalogue of legal-topic KB stores lives in kb_registry.py. Each store
# is its own FAISS index. build_corpus.py reads acts_by_store() to know
# which Act PDFs to fetch and which store to tag each chunk with.
from .kb_registry import KB_STORES, acts_by_store  # noqa: E402  (single source of truth)

ACTS_BY_VERTICAL = acts_by_store()  # back-compat alias used by build tooling

DLSA_FILE = DATA_DIR / "dlsa_directory.json"
DEADLINES_FILE = DATA_DIR / "deadlines.json"
ELIGIBILITY_FILE = DATA_DIR / "eligibility.json"

# ---------------------------------------------------------------------------
# LLM backend (pluggable — the graph is model-agnostic, per the arch doc)
#
#   JB_LLM_BACKEND = "geniex"     -> Qualcomm GenieX running a pre-compiled
#                                    Hexagon-NPU bundle (QnnHtp backend) via
#                                    QAIRT. THE default on real Snapdragon
#                                    silicon: ~15x faster than the CPU ONNX
#                                    path on this hardware (1.9s vs 28.7s for
#                                    an equivalent short prompt, verified),
#                                    same answer quality. Requires a Qualcomm
#                                    AI Hub account (free) + `qai-hub configure
#                                    --api_token <token>` once, then:
#                                      pip install qai-hub qai_hub_models qai_hub_models_cli platformdirs
#                                      qai-hub-models fetch qwen3-4b-instruct-2507 -r geniex_qairt -p w4a16 -c "Snapdragon X Elite"
#                                    (swap the chipset name for your device;
#                                    `qai-hub-models chipsets` lists them).
#                                    Falls back to "onnx" automatically if the
#                                    bundle isn't fetched / GenieX can't load
#                                    (e.g. a non-Snapdragon dev box).
#                  = "onnx"        -> onnxruntime-genai running Phi-3-mini-4k
#                                    -instruct-onnx (CPU int4) fully on-device.
#                                    Universal fallback: works on any x64/arm64
#                                    machine, no special hardware or account
#                                    required, just slower.
#                  = "onnx_qnn"   -> onnxruntime-genai with the QNN execution
#                                    provider, running a Qualcomm AI Hub
#                                    -exported genai_config.json bundle
#                                    directly on the Hexagon NPU. Lower-level
#                                    alternative to "geniex" for the same NPU
#                                    if you have a raw ONNX QNN export instead
#                                    of a GenieX bundle.
#                  = "openai"     -> any OpenAI-compatible /v1 endpoint, e.g. a
#                                    llama.cpp server a teammate stood up
#                                    separately. Dev-machine convenience only.
#                  = "extractive" -> NO LLM: build the answer directly from
#                                    retrieved statute text. Always available,
#                                    fully offline, zero hallucination. Last-
#                                    resort fallback if no model can load.
#
# The reasoning agent tries the configured backend and, if it is unavailable
# (missing model, disk full, generation error, ...) or errors, silently
# degrades to "extractive" — so a demo device without a loaded model still
# produces a correct, grounded answer instead of crashing.
# ---------------------------------------------------------------------------
LLM_BACKEND = os.environ.get("JB_LLM_BACKEND", "geniex")

# Offline ONNX Runtime backend — Phi-3-mini-4k-instruct-onnx, CPU int4 build.
# Downloaded once via huggingface_hub into ONNX_MODEL_CACHE_ROOT, then loaded
# purely from disk on every subsequent run (see llm.py:_ensure_onnx_model_cached).
ONNX_MODEL_REPO = os.environ.get("JB_ONNX_MODEL_REPO", "microsoft/Phi-3-mini-4k-instruct-onnx")
ONNX_MODEL_SUBFOLDER = os.environ.get(
    "JB_ONNX_MODEL_SUBFOLDER", "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
)
ONNX_MODEL_CACHE_ROOT = os.environ.get(
    "JB_ONNX_MODEL_CACHE_ROOT", str(PACKAGE_DIR / "models" / "phi3-mini-onnx")
)
ONNX_MODEL_DIR = os.environ.get(
    "JB_ONNX_MODEL_DIR", str(Path(ONNX_MODEL_CACHE_ROOT) / ONNX_MODEL_SUBFOLDER)
)
ONNX_MAX_LENGTH = int(os.environ.get("JB_ONNX_MAX_LENGTH", "4000"))  # Phi-3-mini-4k context is 4096

# GenieX model source — a Qualcomm AI Hub pre-compiled NPU bundle, fetched
# once via:
#   qai-hub-models fetch qwen3-4b-instruct-2507 -r geniex_qairt -p w4a16 -c "Snapdragon X Elite"
# which downloads a real Hexagon-NPU-targeted (QnnHtp backend) bundle into
# GenieX's own cache (~/.cache/geniex/models/<GENIEX_MODEL>). Verified on this
# Snapdragon X Elite device: ~15x faster than the CPU ONNX backend (1.9s vs
# 28.7s for the same short prompt) with equivalent answer quality. See
# https://github.com/qualcomm/GenieX
GENIEX_MODEL = os.environ.get("JB_GENIEX_MODEL", "qualcomm/Qwen3-4B-Instruct-2507")
GENIEX_DEVICE_MAP = os.environ.get("JB_GENIEX_DEVICE_MAP", "qairt")
GENIEX_PRECISION = os.environ.get("JB_GENIEX_PRECISION", "Q4_0")  # only used for GGUF models
GENIEX_MAX_NEW_TOKENS = int(os.environ.get("JB_GENIEX_MAX_NEW_TOKENS", "512"))

# onnxruntime-genai model bundle dir (contains genai_config.json + tokenizer +
# QNN context binaries), produced by `qai_hub_models.models.<model>.export`.
# See https://onnxruntime.ai/docs/genai/tutorials/snapdragon.html
ONNX_QNN_MODEL_DIR = os.environ.get("JB_ONNX_QNN_MODEL_DIR", "")
ONNX_QNN_MAX_LENGTH = int(os.environ.get("JB_ONNX_QNN_MAX_LENGTH", "1024"))

OPENAI_BASE_URL = os.environ.get("JB_OPENAI_BASE_URL", "http://localhost:8080/v1")
OPENAI_MODEL = os.environ.get("JB_OPENAI_MODEL", "llama-3-8b-instruct")
OPENAI_API_KEY = os.environ.get("JB_OPENAI_API_KEY", "sk-no-key-required")

LLM_TIMEOUT = float(os.environ.get("JB_LLM_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Speech / Vision / Translation backends (the "Part A" input tier) — all
# fully offline/on-device. Every agent follows the same graceful-degradation
# contract as the LLM backend: try the configured backend, degrade to
# "extractive"-style passthrough on failure, never hard-crash.
#
#   JB_ASR_BACKEND    = "whisper"     -> Whisper via transformers, fully
#                                        on-device/offline. Only backend.
#
#   JB_TTS_BACKEND    = "mms"         -> Meta MMS-TTS (facebook/mms-tts-<lang>),
#                                        a real neural on-device model with a
#                                        dedicated checkpoint per language
#                                        (en/hi/ta/te). Default — the ONLY
#                                        backend that can correctly speak
#                                        translated Hindi/Tamil/Telugu answers.
#                     = "pyttsx3"     -> offline OS TTS (SAPI5/NSSpeech/espeak).
#                                        Only as good as whatever voices the OS
#                                        has installed — commonly English-only
#                                        on a stock Windows box, which silently
#                                        mispronounces non-Latin-script text
#                                        rather than failing loudly.
#                     = "none"        -> skip spoken output.
#
#   JB_VISION_BACKEND = "tesseract"   -> fully offline OCR. Only backend.
#
#   JB_TRANSLATION_BACKEND = "nllb"    -> facebook/nllb-200-distilled-600M, a
#                                        real ON-DEVICE MT model (200
#                                        languages incl. Hindi/Tamil/Telugu),
#                                        lazy-loaded via transformers — no
#                                        network needed once cached. Public/
#                                        ungated (ai4bharat/indictrans2 is
#                                        India-specific and in principle
#                                        higher quality, but is a GATED HF
#                                        repo requiring manual access approval
#                                        — set JB_TRANSLATION_BACKEND back to
#                                        it manually once you have access).
#                     = "none"        -> skip translation, English passthrough.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

ASR_BACKEND = os.environ.get("JB_ASR_BACKEND", "whisper")
TTS_BACKEND = os.environ.get("JB_TTS_BACKEND", "mms")
VISION_BACKEND = os.environ.get("JB_VISION_BACKEND", "tesseract")
TRANSLATION_BACKEND = os.environ.get("JB_TRANSLATION_BACKEND", "nllb")

NLLB_MODEL = os.environ.get("JB_NLLB_MODEL", "facebook/nllb-200-distilled-600M")
# Beam search width for NLLB decoding. NLLB runs on plain CPU (no NPU path
# exists for it) and is, measured, the single slowest agent in the pipeline
# (~75s at num_beams=5 for a typical answer). Lower beams trade a small amount
# of fluency for meaningfully lower latency on CPU beam search.
NLLB_NUM_BEAMS = int(os.environ.get("JB_NLLB_NUM_BEAMS", "2"))
INDICTRANS2_MODEL = os.environ.get("JB_INDICTRANS2_MODEL", "ai4bharat/indictrans2-en-indic-dist-200M")
WHISPER_MODEL = os.environ.get("JB_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("JB_WHISPER_DEVICE", "cpu")          # cpu | cuda
WHISPER_COMPUTE_TYPE = os.environ.get("JB_WHISPER_COMPUTE_TYPE", "int8")
TESSERACT_CMD = os.environ.get("JB_TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_K = int(os.environ.get("JB_RETRIEVAL_K", "8"))
# Below this fused-retrieval signal, the Reasoning agent flags
# insufficient_context and the graph loops back to Retrieval (bounded).
RETRIEVAL_MIN_SIM = float(os.environ.get("JB_RETRIEVAL_MIN_SIM", "0.02"))  # floor, not a cosine
# Each retry re-runs retrieval with k widened by +3 AND, if it's a grounding
# retry, a full extra LLM reasoning call (~13-15s on NPU, much more on CPU).
# Measured: a real query hit both retries, ballooning retrieval from 8 to 14
# sections (mostly low-relevance noise, see RELEVANCE_FLOOR below) and roughly
# doubling total pipeline latency for no accuracy gain. 1 retry (not 2) is
# enough headroom for genuinely thin corpora without paying for a second,
# usually-futile widening pass.
MAX_RETRIEVAL_RETRIES = int(os.environ.get("JB_MAX_RETRIEVAL_RETRIES", "1"))
MAX_GROUNDING_RETRIES = int(os.environ.get("JB_MAX_GROUNDING_RETRIES", "1"))
# Semantic hits below this relevance score are dropped before fusion — without
# this, a store with no real match (e.g. free_aid for an off-topic query)
# still contributes its top-k *worst available* chunks at negative relevance
# scores, which the RRF fusion below treats as normal candidates purely by
# rank. Measured: this is exactly what let 6+ irrelevant Legal-Services-Act
# administrative sections into a wages-query reasoning prompt.
RETRIEVAL_RELEVANCE_FLOOR = float(os.environ.get("JB_RETRIEVAL_RELEVANCE_FLOOR", "0.15"))

# ---------------------------------------------------------------------------
# Risk / severity thresholds
# ---------------------------------------------------------------------------
# A deadline at/under this many days => treat the clock as "running now".
DEADLINE_RED_DAYS = int(os.environ.get("JB_DEADLINE_RED_DAYS", "30"))
DEADLINE_AMBER_DAYS = int(os.environ.get("JB_DEADLINE_AMBER_DAYS", "120"))
# Composite confidence below this forces escalation to a human regardless.
LOW_CONFIDENCE_ESCALATE = float(os.environ.get("JB_LOW_CONFIDENCE_ESCALATE", "0.55"))

# ---------------------------------------------------------------------------
# Fallback control — when an on-device backend is unavailable, agents
# normally degrade silently (LLM -> extractive/keyword).
# Set to "0" to DISABLE silent fallback for an agent: it will surface an
# honest "unavailable" state instead of quietly using a lesser backend. The
# pipeline still never crashes — it just escalates to a human sooner instead
# of pretending a degraded answer is the real thing. Useful for: forcing a
# demo to prove the real LLM/NPU path is live, or a UI team wanting to show
# "reasoning temporarily unavailable" rather than a silently-swapped answer.
#
#   JB_ALLOW_LLM_FALLBACK      -> master switch for both agents below
#   JB_ALLOW_REASONING_FALLBACK -> Reasoning agent only (default = master)
#   JB_ALLOW_PLANNER_FALLBACK   -> Planner agent only   (default = master)
# ---------------------------------------------------------------------------
ALLOW_LLM_FALLBACK = os.environ.get("JB_ALLOW_LLM_FALLBACK", "0") == "1"
ALLOW_REASONING_FALLBACK = os.environ.get(
    "JB_ALLOW_REASONING_FALLBACK", "1" if ALLOW_LLM_FALLBACK else "0"
) == "1"
ALLOW_PLANNER_FALLBACK = os.environ.get(
    "JB_ALLOW_PLANNER_FALLBACK", "1" if ALLOW_LLM_FALLBACK else "0"
) == "1"

# ---------------------------------------------------------------------------
# Optional LLM-assisted upgrades — OFF by default. Both are strictly ADDITIVE
# to the deterministic keyword/regex checks (never replace or override them),
# so turning these on can only find MORE grounded claims / eligibility
# reasons, never fewer. Safe to leave off; useful once a real on-device model
# is loaded and you want higher recall.
#
#   JB_LLM_ASSISTED_GROUNDING=1   -> grounding_agent also asks the LLM
#                                    "does section X actually support claim Y?"
#   JB_LLM_ASSISTED_ELIGIBILITY=1 -> escalation_agent also asks the LLM to spot
#                                    Section-12 categories the keyword list missed
# ---------------------------------------------------------------------------
LLM_ASSISTED_GROUNDING = os.environ.get("JB_LLM_ASSISTED_GROUNDING", "0") == "1"
LLM_ASSISTED_ELIGIBILITY = os.environ.get("JB_LLM_ASSISTED_ELIGIBILITY", "0") == "1"

# ---------------------------------------------------------------------------
# Default kiosk location (which DLSA to surface when the user's district is
# unknown). In a real deployment the UNO Q knows its own district.
# ---------------------------------------------------------------------------
DEFAULT_DISTRICT = os.environ.get("JB_DISTRICT", "Kanchipuram")
DEFAULT_STATE = os.environ.get("JB_STATE", "Tamil Nadu")
