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
CHROMA_DIR = os.environ.get("JB_CHROMA", str(PACKAGE_DIR / "chroma_db"))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The catalogue of legal-topic KB stores lives in kb_registry.py. Each store
# is its own Chroma collection. build_corpus.py reads acts_by_store() to know
# which Act PDFs to fetch and which store to tag each chunk with.
from .kb_registry import KB_STORES, acts_by_store  # noqa: E402  (single source of truth)

ACTS_BY_VERTICAL = acts_by_store()  # back-compat alias used by build tooling

DLSA_FILE = DATA_DIR / "dlsa_directory.json"
DEADLINES_FILE = DATA_DIR / "deadlines.json"
ELIGIBILITY_FILE = DATA_DIR / "eligibility.json"

# ---------------------------------------------------------------------------
# LLM backend (pluggable — the graph is model-agnostic, per the arch doc)
#
#   JB_LLM_BACKEND = "geniex"     -> Qualcomm GenieX (QAIRT/Genie under the
#                                    hood). THE primary path: on the Snapdragon
#                                    AI PC it loads an NPU-compiled Qualcomm AI
#                                    Hub bundle; the same API also runs a GGUF
#                                    model via its llama.cpp backend for local
#                                    dev on a machine with an NPU. Note:
#                                    GenieX's native SDK only ships prebuilt
#                                    binaries for win32/arm64 and linux/aarch64
#                                    (verified: `pip install geniex` fails to
#                                    build on this x64 dev box with
#                                    "Unsupported platform ('win32', 'amd64')"
#                                    — that's expected; it's gated to real
#                                    Snapdragon silicon by design).
#                  = "onnx_qnn"   -> onnxruntime-genai with the QNN execution
#                                    provider, running a Qualcomm AI Hub
#                                    -exported genai_config.json bundle
#                                    directly on the Hexagon NPU. A second,
#                                    lower-level on-device path (Python-native
#                                    ORT session) — pick this if a teammate
#                                    already has an AI-Hub ONNX export instead
#                                    of a GenieX bundle.
#                  = "openai"     -> any OpenAI-compatible /v1 endpoint, e.g. a
#                                    llama.cpp server a teammate stood up
#                                    separately. Dev-machine convenience only.
#                  = "extractive" -> NO LLM: build the answer directly from
#                                    retrieved statute text. Always available,
#                                    fully offline, zero hallucination. This is
#                                    the guaranteed fallback so the pipeline
#                                    NEVER hard-fails on stage, and it's what
#                                    actually runs on this x64 dev machine
#                                    since geniex/onnx_qnn need real Snapdragon
#                                    NPU hardware to test end-to-end.
#
# The reasoning agent tries the configured backend and, if it is unavailable
# (wrong hardware, missing model, network down, ...) or errors, silently
# degrades to "extractive" — so a demo device without a loaded model still
# produces a correct, grounded answer instead of crashing.
# ---------------------------------------------------------------------------
LLM_BACKEND = os.environ.get("JB_LLM_BACKEND", "geniex")

# GenieX model source — EITHER a Qualcomm AI Hub pre-compiled NPU bundle
# ("ai-hub-models/<name>", the real hackathon deployment target) OR a GGUF
# repo on Hugging Face (runs via GenieX's llama.cpp backend). See
# https://github.com/qualcomm/GenieX
GENIEX_MODEL = os.environ.get("JB_GENIEX_MODEL", "ai-hub-models/Llama-v3.1-8B-Instruct")
GENIEX_PRECISION = os.environ.get("JB_GENIEX_PRECISION", "Q4_0")  # only used for GGUF models
GENIEX_MAX_NEW_TOKENS = int(os.environ.get("JB_GENIEX_MAX_NEW_TOKENS", "400"))

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
# Speech / Vision backends (the "Part A" input tier)
#
# All follow the same graceful-degradation contract as the LLM backend: try
# the configured backend, fall back on any failure, never hard-crash.
#
#   JB_ASR_BACKEND    = "sarvam"    -> Sarvam Saaras v3 STT (cloud; 23 Indian
#                                      languages; auto lang-detect). Network +
#                                      SARVAM_API_KEY.
#                     = "whisper"   -> faster-whisper, fully on-device/offline.
#
#   JB_TTS_BACKEND    = "sarvam"    -> Sarvam Bulbul v3 TTS (cloud; speaks the
#                                      answer back in the detected language).
#                     = "pyttsx3"   -> offline OS TTS fallback.
#                     = "none"      -> skip spoken output.
#
#   JB_VISION_BACKEND = "sarvam"    -> Sarvam Document Intelligence OCR (cloud;
#                                      Indian-language documents).
#                     = "tesseract" -> fully offline OCR fallback.
#
# SARVAM_API_KEY is read from the environment (or a .env file — see
# python-dotenv load below). NEVER hard-code the key here: this file is
# committed to git. Put it in .env (git-ignored) or export it.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

ASR_BACKEND = os.environ.get("JB_ASR_BACKEND", "sarvam")
TTS_BACKEND = os.environ.get("JB_TTS_BACKEND", "sarvam")
VISION_BACKEND = os.environ.get("JB_VISION_BACKEND", "sarvam")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_OCR_LANGUAGE = os.environ.get("JB_SARVAM_LANG", "en-IN")
SARVAM_STT_MODEL = os.environ.get("JB_SARVAM_STT_MODEL", "saaras:v3")
SARVAM_TTS_MODEL = os.environ.get("JB_SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.environ.get("JB_SARVAM_TTS_SPEAKER", "priya")
WHISPER_MODEL = os.environ.get("JB_WHISPER_MODEL", "small")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_K = 8
# Below this fused-retrieval signal, the Reasoning agent flags
# insufficient_context and the graph loops back to Retrieval (bounded).
RETRIEVAL_MIN_SIM = 0.02  # RRF scores are small; this is a floor, not a cosine
MAX_RETRIEVAL_RETRIES = 2
MAX_GROUNDING_RETRIES = 2

# ---------------------------------------------------------------------------
# Risk / severity thresholds
# ---------------------------------------------------------------------------
# A deadline at/under this many days => treat the clock as "running now".
DEADLINE_RED_DAYS = 30
DEADLINE_AMBER_DAYS = 120
# Composite confidence below this forces escalation to a human regardless.
LOW_CONFIDENCE_ESCALATE = 0.55

# ---------------------------------------------------------------------------
# Default kiosk location (which DLSA to surface when the user's district is
# unknown). In a real deployment the UNO Q knows its own district.
# ---------------------------------------------------------------------------
DEFAULT_DISTRICT = os.environ.get("JB_DISTRICT", "Kanchipuram")
DEFAULT_STATE = os.environ.get("JB_STATE", "Tamil Nadu")
