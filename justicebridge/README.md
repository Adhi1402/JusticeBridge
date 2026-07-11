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

Reasoning/Planner **silently degrade to keyword/extractive** if the LLM is
unavailable, so the pipeline never hard-fails. ASR/OCR/TTS degrade the same way
to on-device (Whisper/Tesseract/pyttsx3).

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
| var | default | purpose |
|---|---|---|
| `SARVAM_API_KEY` | — | Sarvam STT/OCR/TTS (from `.env` or env) |
| `JB_ASR_BACKEND` | `sarvam` | `sarvam` \| `whisper` |
| `JB_VISION_BACKEND` | `sarvam` | `sarvam` \| `tesseract` |
| `JB_TTS_BACKEND` | `sarvam` | `sarvam` \| `pyttsx3` \| `none` |
| `JB_LLM_BACKEND` | `geniex` | `geniex` \| `onnx_qnn` \| `openai` \| `extractive` |
| `JB_DISTRICT` | `Kanchipuram` | which DLSA to surface |
| `JB_HF_OFFLINE` | `1` | keep embeddings/Whisper fully offline after first cache |
