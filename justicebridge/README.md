# JusticeBridge — on-device multilingual legal navigator

The **AI-PC orchestration brain** for the Snapdragon Multiverse hackathon: a
LangGraph agent state machine that turns a citizen's **spoken or scanned**
legal problem into **plain-language, statute-grounded guidance + a physical
urgency signal**, and always ends by handing off to a real **free** lawyer
(DLSA). Voice-first, multilingual, privacy-first.

**Legal information, never legal advice.** Every output closes with "this is
general information — for your case contact [a real human]", and no legal
claim is spoken unless it maps to a statute section that was actually
retrieved (the Grounding-Verification agent enforces this).

## Legal verticals (each is its own vector KB)

| Vertical | KB store | Act(s) | Status |
|---|---|---|---|
| Unpaid wages & labour | `kb_wages` | Code on Wages, 2019 | ✅ built |
| Consumer protection | `kb_consumer` | Consumer Protection Act, 2019 | ✅ built |
| Family & domestic protection | `kb_family` | PWDVA 2005 + Hindu Marriage Act 1955 | ✅ built |
| Free legal aid (cross-cutting) | `kb_free_aid` | Legal Services Authorities Act, 1987 | ✅ built |
| Tenancy / eviction, Police/FIR | — | — | 🚧 stub → human handoff |

Adding a vertical = one entry in [`kb_registry.py`](kb_registry.py); its Act
PDF is fetched from `indiacode (3).json` automatically. See *Adding a vertical*.

---

## Quick start

```bash
pip install -r justicebridge/requirements.txt

# 1. build the corpus from the real Act PDFs + one Chroma collection per KB store
python -m justicebridge.build_corpus      # all verticals (or --vertical wages)
python -m justicebridge.build_index

# 2. run a query end-to-end (text)
python -m justicebridge.run_cli "I worked two months but the contractor hasn't paid my wages"

# 2b. or drive it with a real recording / photo
python -m justicebridge.run_cli --audio recording.wav --image notice.jpg

# 3. the gold-standard evaluation (the pitch number)
python -m justicebridge.eval.run_eval

# 4. the demo UI (mic + camera + severity light + spoken answer)
streamlit run justicebridge/app.py
```

Set your Sarvam key (for cloud STT/OCR/TTS) via a **git-ignored** `.env` at
the repo root, or an env var — never hard-code it:

```
SARVAM_API_KEY=sk_...
```

---

## Architecture: the Planner routes to knowledge bases

The core design is **"planner picks the knowledge base(s), retriever searches
only those."** [`kb_registry.py`](kb_registry.py) is the catalogue of every
legal-topic KB store; each store is its own Chroma vector collection built
from that topic's Act(s). The Planner reads the citizen's words, chooses the
relevant store id(s) (e.g. `["wages", "free_aid"]`) and writes them to
`state["kb_stores"]`; the Retrieval agent searches exactly those stores. So a
wage complaint is never matched against family law. The cross-cutting
`free_aid` store is appended to every supported query (a small reserved quota)
because free-aid eligibility applies regardless of topic.

```
START ─► asr ─┐
START ─► vision ─┘─► combine ─► planner
planner ─[supported?]─ no ─► escalation ─► output ─► translation ─► tts ─► END
        └─ yes ─► retrieval ─► reasoning
reasoning ─[insufficient_context]─► retrieval          (loop 1, bounded)
          └─► grounding
grounding ─[needs_redraft]─► reasoning                 (loop 2, bounded)
          └─► risk ─► escalation ─► output ─► translation ─► tts ─► END
```

---

## The agents (each one, in detail)

Every node returns only the keys it changes (partial state). ASR + Vision run
in parallel from `START`. All external-service agents follow the same
**graceful-degradation** contract: try the configured backend, fall back on
any failure, never hard-crash.

### 1. ASR agent — `agents/io_agents.py` → `asr_agent`
- **Role:** speech → text. Voice-first: workers speak, they don't type.
- **Backends** (`JB_ASR_BACKEND`): `sarvam` — Sarvam **Saaras v3** (cloud, 23
  Indian languages, auto language-detect); `whisper` — faster-whisper, fully
  on-device/offline. Tries Sarvam, falls back to Whisper on any failure.
- **Tools:** `sarvam_stt_tool`, `whisper_stt_tool` (LangChain `@tool`).
- **In:** `audio_bytes` (or `text_input` for typed/eval path).
  **Out:** `transcript`, `asr_confidence`, `lang` (detected).
- *Verified live:* Saaras needs `file=` as an **open binary object**, not a
  path; Whisper transcribed synthesized speech at 99.8% confidence.

### 2. Vision agent — `agents/io_agents.py` → `vision_agent`
- **Role:** document photo → text (supplementary — a low OCR score is fine).
- **Backends** (`JB_VISION_BACKEND`): `sarvam` — Sarvam **Document
  Intelligence** (cloud OCR, Indian-language documents); `tesseract` — offline.
  Tries Sarvam, falls back to Tesseract.
- **Tools:** `sarvam_ocr_tool`, `tesseract_ocr_tool`.
- **In:** `image` (PIL). **Out:** `doc_text`, `vision_confidence`.
- *Verified live:* Sarvam's `download_output()` returns a **ZIP** (not raw md)
  — the tool unzips `document.md`; confirmed exact extraction on a test doc.

### 3. Combine — `agents/io_agents.py` → `combine_node`
Merges `transcript` + `doc_text` into `combined_text` for the Planner.

### 4. Planner / Router agent — `agents/planner_agent.py`
- **Role:** the "which knowledge base?" decision. Picks the KB store(s) to
  search from the full [`kb_registry`](kb_registry.py) catalogue.
- **Backends:** LLM classification when a model is live (robust to
  paraphrase/code-mixing) → keyword scoring fallback (always available).
- **In:** `combined_text`. **Out:** `vertical` (primary topic), `kb_stores`
  (search set, incl. `free_aid`), `supported`, `output_template`,
  `planner_backend`.
- Unsupported-but-recognised topics (tenancy, fir) → `supported=False`, which
  short-circuits to the human-handoff branch.

### 5. Retrieval agent — `agents/retrieval_agent.py` + `retrieval.py`
- **Role:** hybrid search over **only** the Planner-selected KB stores.
- **How:** per-store **BM25 + vector (Chroma)** fused with Reciprocal Rank
  Fusion, merged across stores; the substantive topic owns most citation slots
  and `free_aid` gets a small reserved quota. Widens `k` on a retry.
- **In:** `combined_text`, `kb_stores`. **Out:** `retrieved_sections`,
  `retrieval_sim` (a weak-retrieval signal that drives the retry loop).

### 6. Reasoning agent — `agents/reasoning_agent.py`
- **Role:** plain-language explanation citing **only** retrieved sections.
- **Backends:** on-device LLM via `on_device_reasoning_tool` (GenieX /
  onnx_qnn / openai — see LLM table) returning JSON `{answer, claims}`; or the
  **extractive** fallback that builds the answer directly from retrieved
  statute text (zero hallucination, always available).
- **In:** `retrieved_sections`, `combined_text`. **Out:** `draft_answer`,
  `draft_claims` (each tied to a `section_no`), `citations`,
  `insufficient_context` (→ retry Retrieval before drafting from thin air),
  `reasoning_backend`.

### 7. Grounding-Verification agent — `agents/grounding_agent.py` *(trust layer)*
- **Role:** every claim must map to a section that was actually retrieved — the
  line between trustworthy and dangerous in a legal tool.
- **Checks:** citation check (the cited `section_no` was retrieved) + lexical
  support check for LLM drafts (claim terms overlap the cited section text).
- **Out:** `grounded`; ungrounded claims force a bounded redraft
  (`needs_redraft`) or are **stripped** (fail safe, never fail loud).

### 8. Risk / Deadline agent — `agents/risk_agent.py`
- **Role:** turns signals into a **composite confidence** + a **grounded
  urgency colour** (red/amber/green). Urgency is a real legal clock
  (limitation / action window from `data/deadlines.json` keyed per vertical),
  never a vibe.
- **Out:** `composite_confidence`, `deadline_days`, `deadline_basis`,
  `severity`.

### 9. Escalation / Aid agent — `agents/escalation_agent.py` *(headline feature)*
- **Role:** two pure lookups over structured data (zero hallucination):
  **Section 12 eligibility** (scans the citizen's words for automatic free-aid
  categories; per-vertical presumptions: wages→industrial workman,
  family→woman) and the **DLSA handoff** (nearest office, phone, hours, what to
  bring, Tele-Law). Always attaches a human handoff.
- **Out:** `escalate`, `eligibility_reasons`, `dlsa_contact`, `severity`
  (default green on the unsupported path).

### 10. Translation agent — `agents/translation_agent.py`
- **Role:** render the assembled English answer into the citizen's language
  (Tamil/Hindi/Telugu) via IndicTrans2. Lazy-loaded; falls back to English if
  not installed. **Out:** `final_answer_local`.

### 11. Output agent — `agents/output_agent.py`
- **Role:** assemble the spoken script (grounded rights + free-aid headline +
  real deadline + DLSA handoff + mandatory disclaimer) and the multi-device
  **`signal_packet`** (AI PC → UNO Q) + `phone_message`.

### 12. TTS agent — `agents/tts_agent.py`
- **Role:** speak the answer back. **Voice in → voice out** (only runs for
  audio input or `want_tts`).
- **Backends** (`JB_TTS_BACKEND`): `sarvam` — **Bulbul v3** (speaks in the
  detected language); `pyttsx3` — offline; `none`. **Out:** `audio_response`
  (WAV bytes). *Verified live:* Bulbul returns a list of base64 WAV strings.

---

## Backends at a glance

**LLM** (`JB_LLM_BACKEND`, used by Reasoning + Planner) — see [`llm.py`](llm.py):

| value | what |
|---|---|
| `geniex` (default) | Qualcomm **GenieX** (QAIRT/Genie) NPU bundle on the Snapdragon AI PC. Native SDK only builds on `win32/arm64` / `linux/aarch64` — **won't `pip install` on x64** (verified), by design. |
| `onnx_qnn` | `onnxruntime-genai` + QNN EP on the Hexagon NPU, loading a Qualcomm AI Hub bundle (`JB_ONNX_QNN_MODEL_DIR`). |
| `openai` | any OpenAI-compatible `/v1` endpoint (e.g. a llama.cpp server). Dev convenience. |
| `extractive` | no model; grounded answer from retrieved statute text. Always available — what runs on a non-Snapdragon dev box. |

By default Reasoning/Planner **silently degrade to keyword/extractive** if the
LLM is unavailable, and ASR/OCR/TTS degrade the same way to on-device
(Whisper/Tesseract/pyttsx3) — the pipeline never hard-fails. This silent
degrade is itself controllable — see below.

### Fallback control (`JB_ALLOW_*_FALLBACK`)

Sometimes you want to know the real backend is actually live rather than get
a good-looking but silently-downgraded answer (proving the demo is really
using the NPU, or a UI wanting to show "reasoning temporarily unavailable").
Set these to `0` to disable silent fallback for that agent — it still never
crashes, it just surfaces an honest `"...backend": "unavailable"` and lets the
normal low-confidence path escalate to a human sooner:

| var | default | effect when `0` |
|---|---|---|
| `JB_ALLOW_LLM_FALLBACK` | `1` | master switch for both agents below |
| `JB_ALLOW_REASONING_FALLBACK` | = master | Reasoning won't use the extractive answer; `draft_answer` stays empty, `reasoning_backend="unavailable"`, `grounded=False` → confidence capped low → auto-escalates |
| `JB_ALLOW_PLANNER_FALLBACK` | = master | Planner won't use keyword routing when the LLM is down; routes straight to the safe unsupported/handoff branch instead |

### Optional LLM-assisted upgrades (off by default, strictly additive)

Two agents can optionally use a second LLM call to catch things the
deterministic checks miss — both are designed so enabling them can only make
the result **more** grounded / **more** eligibility hits found, never less
(the keyword/regex check is always the floor, the LLM only tightens Grounding
or loosens — i.e. finds more of — Eligibility):

| var | default | agent | what it adds |
|---|---|---|---|
| `JB_LLM_ASSISTED_GROUNDING` | `0` | Grounding-Verify | a second "does this section actually entail this claim?" check beyond lexical overlap — catches a claim that shares vocabulary with a section but inverts its meaning |
| `JB_LLM_ASSISTED_ELIGIBILITY` | `0` | Escalation/Aid | re-checks the Section-12 category list against the citizen's words for **implied** matches the keyword cues miss (e.g. "I've needed a wheelchair since the accident" → disability, without the word "disabled") |

**Deliberately NOT LLM-based:** Risk/Deadline (urgency must stay a real legal
clock from `deadlines.json`, never a model's "vibe") and the DLSA/eligibility
*lookup itself* (must stay a pure data lookup — that's what makes "you qualify
for free legal aid" a zero-hallucination-risk claim in the first place).

---

## Building a UI on top of this backend

Two ways to integrate, both stable contracts a separate frontend team can
build against without touching agent internals:

**1. HTTP API — [`api.py`](api.py)** (recommended for a separate UI team):
```bash
pip install fastapi uvicorn
uvicorn justicebridge.api:app --host 0.0.0.0 --port 8080
```
| endpoint | purpose |
|---|---|
| `GET /health` | which backends are configured/live — for a status bar |
| `GET /kb-stores` | the legal-topic catalogue — for a "what can I ask about" menu |
| `POST /ask` | `{text_input?, audio_base64?, image_base64?, lang, want_tts}` → JSON result (audio/image travel as base64; every other field is plain JSON) |

The response is a deliberate **allowlist** (`api.py::_RESPONSE_FIELDS`), not a
dump of internal state — so new internal fields never leak into the contract
until reviewed and added there.

**2. Direct Python — `graph.get_app().invoke(state_dict)`** for a team working
in the same codebase (e.g. extending `app.py`). Input/output is the
`CaseState` TypedDict in [`state.py`](state.py) — every field is plain
str/int/float/bool/list/dict except `image` (PIL.Image) and `audio_bytes` /
`audio_response` (raw bytes), which only matter for direct Python use, not the
HTTP API (which base64-encodes them).

---

## Structured data (`data/`, zero-hallucination lookups)
- `corpus.json` — statute chunks (full PDF text, nothing dropped; tagged by KB store).
- `eligibility.json` — Legal Services Act **Section 12** categories.
- `deadlines.json` — limitation / action windows → drives the LED colour.
- `dlsa_directory.json` — DLSA / Taluka / Tele-Law contacts by district.

> ⚠️ Every statute reference, limitation period, and phone number here is a
> **starting reference** and must be verified against current law before a real
> deployment. Placeholders are marked `XXXX` / `(VERIFY)`.

---

## Multi-device contract (AI PC → UNO Q)
`output_agent` emits `signal_packet`; `signal_client.py` POSTs it to the UNO Q.
Test without the board:
```bash
python -m justicebridge.uno_q_listener    # terminal 1 (mock UNO Q)
python -m justicebridge.signal_client      # terminal 2 (AI PC sender)
```
Unreachable UNO Q → send fails **softly** (phone-only fallback).

---

## Adding a vertical
1. Add an entry to `KB_STORES` in [`kb_registry.py`](kb_registry.py): topic,
   description, `acts` (exact indiacode `short_title`), `collection`,
   `planner_keywords`, `deadline_key`, `output_template`.
2. `python -m justicebridge.build_corpus --vertical <id>` (fetches the Act PDF,
   chunks it, merges into `corpus.json`).
3. `python -m justicebridge.build_index` (builds its Chroma collection).
4. Optionally add a `deadlines.json` entry. Done — the Planner, Retrieval, and
   Risk agents all read the registry; nothing else changes.

---

## Latest eval (default backend degrading to `extractive`, 25 gold cases across
4 verticals — corpus from the real Act PDFs)
```
Routing (vertical)     : 100%
Routing (support flag) : 100%
Citation hit@k         : 92%
Grounded (supported)   : 100%
Escalation decision    : 100%
Aid handoff present    : 100%
Severity match         : 100%
```
`python -m justicebridge.eval.run_eval`. On the Snapdragon AI PC with
`JB_LLM_BACKEND=geniex` this scores the real on-device reasoning path.

---

## Key environment variables
See `justicebridge/.env.example` for a ready-to-copy template.

**Backends**
| var | default | purpose |
|---|---|---|
| `SARVAM_API_KEY` | — | Sarvam STT/OCR/TTS (from `.env` or env) |
| `JB_ASR_BACKEND` | `sarvam` | `sarvam` \| `whisper` |
| `JB_VISION_BACKEND` | `sarvam` | `sarvam` \| `tesseract` |
| `JB_TTS_BACKEND` | `sarvam` | `sarvam` \| `pyttsx3` \| `none` |
| `JB_LLM_BACKEND` | `geniex` | `geniex` \| `onnx_qnn` \| `openai` \| `extractive` |
| `JB_GENIEX_MODEL` | `ai-hub-models/Llama-v3.1-8B-Instruct` | AI Hub bundle id or a GGUF HF repo |
| `JB_ONNX_QNN_MODEL_DIR` | — | path to an AI Hub `genai_config.json` bundle dir |
| `JB_OPENAI_BASE_URL` | `http://localhost:8080/v1` | for `JB_LLM_BACKEND=openai` |
| `JB_WHISPER_MODEL` | `small` | faster-whisper model size |
| `JB_WHISPER_DEVICE` / `JB_WHISPER_COMPUTE_TYPE` | `cpu` / `int8` | faster-whisper runtime settings |
| `JB_TESSERACT_CMD` | (Windows default path) | Tesseract-OCR binary location |
| `JB_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | retrieval embedding model |

**Fallback control & optional LLM upgrades** — see the *Backends at a glance*
section above for the full explanation.
| var | default |
|---|---|
| `JB_ALLOW_LLM_FALLBACK` / `JB_ALLOW_REASONING_FALLBACK` / `JB_ALLOW_PLANNER_FALLBACK` | `1` |
| `JB_LLM_ASSISTED_GROUNDING` / `JB_LLM_ASSISTED_ELIGIBILITY` | `0` |

**Retrieval, risk & misc**
| var | default | purpose |
|---|---|---|
| `JB_RETRIEVAL_K` | `8` | sections retrieved per query |
| `JB_RETRIEVAL_MIN_SIM` | `0.02` | below this, Reasoning retries Retrieval |
| `JB_MAX_RETRIEVAL_RETRIES` / `JB_MAX_GROUNDING_RETRIES` | `2` / `2` | bounded-loop caps |
| `JB_DEADLINE_RED_DAYS` / `JB_DEADLINE_AMBER_DAYS` | `30` / `120` | severity thresholds |
| `JB_LOW_CONFIDENCE_ESCALATE` | `0.55` | confidence floor that forces escalation |
| `JB_DISTRICT` / `JB_STATE` | `Kanchipuram` / `Tamil Nadu` | which DLSA to surface |
| `JB_HF_OFFLINE` | `1` | keep embeddings/Whisper fully offline after first cache |
