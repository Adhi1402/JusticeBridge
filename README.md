# JusticeBridge — offline, on-device legal aid, in a box

## The problem

India has over 5 crore (50 million) pending court cases, and most citizens
can't afford a lawyer. Legal aid deserts are worst in rural India: people
face FIRs, tenancy disputes, and unpaid wages with no one to explain their
rights. Even where the law already protects them, only 1 in 5 people who
qualify for a free government lawyer know it — there's one legal aid clinic
for every 163 villages, and the outreach workforce meant to close that gap
has shrunk 38% in five years. Existing digital legal tools need internet,
charge fees, and send your data to someone else's server — none of that
reaches the people who need this most.

JusticeBridge fixes the actual bottleneck: it reaches people offline, for
free, at a place they already go.

## The solution

Not an app — a kiosk, like an ATM for free legal help, placed in a panchayat
office or community center. Three coordinated components, all Snapdragon
silicon:

1. **A Snapdragon X AI PC** running the full multi-agent reasoning pipeline,
   entirely on-device.
2. **The visitor's own phone**, capturing voice and documents.
3. **An Arduino UNO Q**, an always-on sentinel showing a physical trust
   signal (the severity light).

No cloud. No data plan. No internet, period — every backend below runs
on-device.

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

# 1. build the corpus from the real Act PDFs + one FAISS index per KB store
python -m justicebridge.build_corpus      # all verticals (or --vertical wages)
python -m justicebridge.build_index

# 2. run a query end-to-end (text)
python -m justicebridge.run_cli "I worked two months but the contractor hasn't paid my wages"

# 2b. or drive it with a real recording / photo — voice and document(s) are
#     BOTH optional and never required together; --image can repeat for
#     multiple documents
python -m justicebridge.run_cli --audio recording.wav --image notice.jpg
python -m justicebridge.run_cli --image page1.jpg --image page2.jpg   # multi-doc, no voice

# 3. the gold-standard evaluation (the pitch number)
python -m justicebridge.eval.run_eval

# 4. the demo UI (mic + multi-file upload + severity light + spoken answer)
streamlit run justicebridge/app.py

# 4b. or the React frontend (justicebridge/frontend/) against the HTTP API
uvicorn justicebridge.api:app --host 0.0.0.0 --port 8080   # terminal 1
cd justicebridge/frontend && npm install && npm run dev    # terminal 2
```

**Fully offline, no cloud API keys required.** Every backend (ASR, OCR, TTS,
translation, and the default LLM path) runs on-device. A `.env` at the repo
root is only needed to override defaults (model choice, district, etc.) — see
[`.env.example`](.env.example).

---

## Architecture: the Planner routes to knowledge bases

The core design is **"planner picks the knowledge base(s), retriever searches
only those."** [`kb_registry.py`](kb_registry.py) is the catalogue of every
legal-topic KB store; each store is its own FAISS vector index built
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

## Which LLM does each agent use?

**Only 4 of the 12 agents call an LLM, and all 4 share one configured
backend** (`JB_LLM_BACKEND`, dispatched from `llm.py`) — there's no per-agent
model choice. The other 8 use a dedicated non-LLM model (speech/OCR/
embedding/translation/TTS) or pure code and data lookups, and can never call
an LLM regardless of configuration.

| # | Agent | Uses an LLM? | Which model |
|---|---|---|---|
| 1 | ASR | ❌ No | Whisper (transformers, on-device) — speech-recognition model |
| 2 | Vision | ❌ No | Tesseract — OCR model, on-device |
| 3 | Combine | ❌ No | — (string concatenation) |
| 4 | **Planner** | ✅ Optional | *(shared backend — see below)* |
| 5 | Retrieval | ❌ No | `sentence-transformers/all-MiniLM-L6-v2` — embedding/similarity model |
| 6 | **Reasoning** | ✅ Optional | *(shared backend — see below)* |
| 7 | **Grounding-Verify** | ✅ Optional, off by default (`JB_LLM_ASSISTED_GROUNDING=1`) | *(shared backend — see below)* |
| 8 | Risk/Deadline | ❌ No, deliberately | — (real legal-clock data, never a model) |
| 9 | **Escalation/Aid** | ✅ Optional, off by default (`JB_LLM_ASSISTED_ELIGIBILITY=1`) | *(shared backend — see below)* |
| 10 | Translation | ❌ No | NLLB-200 (on-device) — machine-translation model, not an LLM |
| 11 | Output | ❌ No | — (string templating) |
| 12 | TTS | ❌ No | MMS-TTS / pyttsx3 — speech-synthesis models, on-device |

**The shared backend** (`JB_LLM_BACKEND`, default `geniex`) — set once, used
by Planner/Reasoning/Grounding/Escalation:

| `JB_LLM_BACKEND` | Actual model | Where it runs |
|---|---|---|
| `geniex` (default) | `JB_GENIEX_MODEL`, default **`qualcomm/Qwen3-4B-Instruct-2507`**, a Qualcomm AI Hub pre-compiled Hexagon-NPU bundle (QnnHtp backend via QAIRT) | Snapdragon Hexagon NPU. **Falls back to `onnx` automatically** if the bundle isn't fetched or GenieX can't load (e.g. this x64 dev box) — the one cross-backend fallback in the LLM dispatcher, so it's safe to leave as the default everywhere |
| `onnx` | `JB_ONNX_MODEL_REPO`, default **`microsoft/Phi-3-mini-4k-instruct-onnx`** (CPU int4 build), via `onnxruntime-genai` | Any x64/arm64 CPU — the universal offline fallback, and what actually runs on this dev box |
| `onnx_qnn` | Whatever Qualcomm AI Hub `genai_config.json` bundle `JB_ONNX_QNN_MODEL_DIR` points at — model-agnostic (Llama-3.2-3B, Phi-3.5-mini, etc., per whatever a teammate exported) | Snapdragon Hexagon NPU |
| `openai` | `JB_OPENAI_MODEL`, default `llama-3-8b-instruct` — whatever model a teammate's llama.cpp/OpenAI-compatible server is actually serving | wherever that server runs (dev convenience) |
| `extractive` | **No model at all.** Last-resort fallback if no model can load. | — |

`geniex`/`onnx_qnn` need real Snapdragon ARM64 hardware — `geniex` won't even
`pip install` on x64. On a non-Snapdragon dev box, `geniex`'s automatic
fallback means `onnx` (Phi-3-mini on CPU) answers instead — a real model,
just slower than the NPU path (measured on Snapdragon X Elite: geniex ~1.9s
vs onnx ~28.7s for an equivalent short prompt). `extractive` is the
guaranteed floor if even that fails to load.

---

## The agents (each one, in detail)

Every node returns only the keys it changes (partial state). ASR + Vision run
in parallel from `START`. All external-service agents follow the same
**graceful-degradation** contract: try the configured backend, fall back on
any failure, never hard-crash. Sample input/output below is real output from
an actual run (extractive reasoning backend, no live LLM), not invented.

### 1. ASR agent — `agents/io_agents.py` → `asr_agent`
- **Role:** speech → text. Voice-first — workers speak, they don't type.
- **LLM? No.** Whisper transcribes audio; it doesn't generate text.
- **Tools:** `whisper_stt_tool` (LangChain `@tool`).
- **Backend** (`JB_ASR_BACKEND`, only `whisper`): a local Whisper model
  (default `openai/whisper-small`) via `transformers`' ASR pipeline, fully
  offline. Not `faster-whisper` — its `ctranslate2` dependency has no
  Windows-ARM64 wheel, and `openai-whisper` has the same problem via
  `tiktoken`/`numba`. `transformers` + `torch` (already required) is the
  stack that actually installs on Snapdragon ARM64.
- **In:** `{"audio_bytes": <wav bytes>}` — or `{"text_input": "..."}` for the
  typed/eval path, which passes through with `asr_confidence=1.0`.
- **Out (real):** `{"transcript": "I worked for two months but my contractor
  has not paid my wages.", "asr_confidence": 0.8, "lang": "unknown"}`
- **Note:** `transformers`' ASR pipeline doesn't return real
  confidence/language, so both are fixed placeholders — see *Voice and
  document are both optional* below.

### 2. Vision agent — `agents/io_agents.py` → `vision_agent`
- **Role:** document photo(s) → text — supplementary; a low OCR score, or no
  document at all, is fine.
- **LLM? No.** Tesseract extracts text; it doesn't reason about it.
- **Tools:** `tesseract_ocr_tool`.
- **Backend** (`JB_VISION_BACKEND`, only `tesseract`): fully offline OCR
  (needs the Tesseract-OCR binary installed on the OS, `JB_TESSERACT_CMD`).
- **Multiple documents:** accepts `state["images"]` (a **list** of PIL
  Images) — each is OCR'd independently (one failed page never blocks the
  rest), concatenated and labeled `--- Document N ---`, with the average
  confidence across pages that returned text. `state["image"]` (singular)
  still works for one document.
- **In (2 documents):** `{"images": [<PIL.Image>, <PIL.Image>]}`
- **Out (real — 2 documents OCR'd via Tesseract):**
  ```json
  {"doc_text": "--- Document 1 ---\nI worked two months but the contractor\n\nhas not paid my wages of Rs 18000\n\n--- Document 2 ---\nNotice: contractor name is Ramesh Traders, phone 9876543210",
   "vision_confidence": 1.0}
  ```
- **Verified:** a blank/unreadable image OCRs to `""` at 0.0 confidence
  instead of inventing text.

### 3. Combine — `agents/io_agents.py` → `combine_node`
- **Role:** merges `transcript` + `doc_text` into one `combined_text` string
  for the Planner. Works with either, both, or (if nothing was said or
  scanned) neither.
- **LLM? No. Tools: none.** Pure string concatenation.
- **In:** `{"transcript": "...", "doc_text": "..."}` (either may be empty)
- **Out:** `{"combined_text": "transcript text doc text"}`

### 4. Planner / Router agent — `agents/planner_agent.py`
- **Role:** the "which knowledge base?" decision — picks the KB store(s) to
  search from the [`kb_registry`](kb_registry.py) catalogue.
- **LLM? Optional** (`JB_LLM_BACKEND` != `extractive`/unavailable) — the model
  picks the best-matching topic id (robust to paraphrase/code-mixing) via a
  direct `llm.chat()` call. Falls back to keyword scoring (word-boundary
  matched, see `text_match.py`) if the LLM is unavailable or
  `JB_ALLOW_PLANNER_FALLBACK=0`.
- **Tools: none** (direct call, not a LangChain `@tool`).
- **In:** `{"combined_text": "my employer has not paid my wages"}`
- **Out (real, keyword backend):**
  ```json
  {"vertical": "wages", "supported": true, "kb_stores": ["wages", "free_aid"],
   "output_template": "wage_dispute", "planner_backend": "keyword", "off_topic": false}
  ```
- **Out (off-topic, real):** input `"what is the weather like today, I want
  to know the cricket score"` →
  ```json
  {"vertical": null, "supported": false, "kb_stores": [], "off_topic": true}
  ```
- Unsupported-but-recognised topics (tenancy, FIR) → `supported=False,
  off_topic=False` — a real legal topic the tool doesn't cover yet, different
  from `off_topic=True` (see *Off-topic vs unsupported* below).
- **Known limitation:** only ONE vertical is chosen per query — a query
  mixing two legal issues (e.g. "my husband, a factory worker, hasn't been
  paid and also beats me") routes to only one (verified: routed to `wages`,
  the domestic-violence content was silently not addressed). Not fixed in
  this build; see *Edge cases* below.

### 5. Retrieval agent — `agents/retrieval_agent.py` + `retrieval.py`
- **Role:** hybrid search over **only** the Planner-selected KB stores.
- **LLM? No.** Uses an embedding model (`sentence-transformers/all-
  MiniLM-L6-v2`, similarity not generative) + BM25 keyword search.
- **Tools: none** (`retrieve()` is a plain function).
- **How:** per-store **BM25 + vector (FAISS)** fused with Reciprocal Rank
  Fusion, merged across stores — the substantive topic owns most citation
  slots and `free_aid` gets a small reserved quota. Semantic hits below
  `JB_RETRIEVAL_RELEVANCE_FLOOR` (default `0.15`) are dropped before fusion,
  so a store with no real match (e.g. `free_aid` on an off-topic query) can't
  contribute its worst-available chunks as if they were normal candidates.
  Widens `k` on a retry.
- **In:** `{"combined_text": "my employer has not paid my wages for two months",
  "kb_stores": ["wages", "free_aid"]}`
- **Out (real, k=3):**
  ```json
  {"retrieval_sim": 0.288,
   "retrieved_sections": [
     {"act": "The Code on Wages, 2019", "section_no": "17",
      "title": "Time limit for payment of wages", "store": "wages", "score": 0.0328,
      "text": "Time limit for payment of wages. (iv) monthly basis, before the expiry..."},
     {"act": "The Code on Wages, 2019", "section_no": "2", "title": "Definitions", ...}
   ]}
  ```

### 6. Reasoning agent — `agents/reasoning_agent.py`
- **Role:** plain-language explanation citing **only** retrieved sections.
- **LLM? Optional** (`JB_LLM_BACKEND` != `extractive`/unavailable).
- **Tools:** `on_device_reasoning_tool` (`@tool`) — the ONLY tool-wrapped LLM
  call in the pipeline. Given `{query, sections}`, asks the configured
  backend for JSON `{answer, claims}` where each claim is tied to a
  `section_no`, so Grounding can verify it.
- **Fallback (`_extractive_draft`, always available):** builds the answer
  directly from the retrieved sections' text — every sentence is, by
  construction, tied to a real section, so it can't hallucinate. This is what
  runs on a non-Snapdragon dev box.
- **In:** `{"retrieved_sections": [...], "combined_text": "..."}`
- **Out (real, extractive backend):**
  ```json
  {"reasoning_backend": "extractive", "insufficient_context": false,
   "draft_answer": "Here is what the law says about your situation: Under The Code on Wages, 2019, Section 17 (Time limit for payment of wages): (iv) monthly basis, before the expiry of the seventh day of the succeeding month. ...",
   "draft_claims": [{"claim": "Time limit for payment of wages — The Code on Wages, 2019 s.17", "section_no": "17"}],
   "citations": [{"act": "The Code on Wages, 2019", "section_no": "17", "title": "Time limit for payment of wages"}]}
  ```
- If retrieval was too weak, sets `insufficient_context=true` instead of
  drafting from thin air, and loops back to Retrieval (bounded, max 1 retry
  by default — lowered from 2, since a retry widens retrieval by +3 AND, on a
  grounding retry, costs a full extra LLM call; a measured query hit both
  retries and roughly doubled total latency for no accuracy gain).
- `JB_ALLOW_REASONING_FALLBACK=0` disables the extractive fallback — see
  *Fallback control* below.

### 7. Grounding-Verification agent — `agents/grounding_agent.py` *(trust layer)*
- **Role:** every claim must map to a section that was actually retrieved —
  the line between trustworthy and dangerous in a legal tool.
- **LLM? Optional, off by default** (`JB_LLM_ASSISTED_GROUNDING=1`) — a
  second "does this section actually entail this claim?" check, strictly
  additive: can only reject a claim that already passed the deterministic
  check, never approve one that failed it.
- **Tools: none** (direct LLM call when enabled).
- **Deterministic checks (always run):** citation check (cited `section_no`
  was actually retrieved) + lexical overlap check (claim's key terms overlap
  the cited section's text).
- **In:** `{"draft_claims": [...], "retrieved_sections": [...], "reasoning_backend": "extractive"}`
- **Out (real):**
  ```json
  {"grounded": true, "needs_redraft": false, "ungrounded_claims": [],
   "draft_claims": [{"claim": "...", "section_no": "17"}]}
  ```
- Ungrounded claims force a bounded redraft (`needs_redraft=true`, max 1
  retry by default) or are **stripped** once retries are exhausted — fail
  safe, never fail loud with a hallucinated rule.

### 8. Risk / Deadline agent — `agents/risk_agent.py`
- **Role:** turns signals into a **composite confidence** + a **grounded
  urgency colour** (red/amber/green).
- **LLM? No, deliberately.** Urgency comes from real limitation periods in
  `data/deadlines.json` (keyed per vertical), never a model's "vibe" — an
  intentional design boundary: severity reflects a real legal clock, not how
  urgent the text *sounds* (see *Severity ≠ emotional urgency* in Edge Cases).
- **Tools: none.** Pure arithmetic + JSON lookup.
- **In:** `{"asr_confidence": 1.0, "retrieval_sim": 0.288, "vertical": "wages", "grounded": true, "retry_count": 0}`
- **Out (real):**
  ```json
  {"composite_confidence": 0.667, "severity": "amber",
   "deadline_days": 90, "deadline_basis": "Code on Wages, 2019, s.45 — an application for a claim relating to wages must generally be filed within three years... (VERIFY.)"}
  ```

### 9. Escalation / Aid agent — `agents/escalation_agent.py` *(headline feature)*
- **Role:** two pure lookups over structured data, zero hallucination:
  **Section 12 eligibility** + the **DLSA handoff** (nearest office, phone,
  hours, what to bring, Tele-Law). Always attaches a human handoff, except
  for a genuinely off-topic query — see below.
- **LLM? Optional, off by default** (`JB_LLM_ASSISTED_ELIGIBILITY=1`) —
  re-checks the fixed Section-12 category list for **implied** matches the
  keyword cues miss (e.g. "I've needed a wheelchair since the accident" →
  disability). Strictly additive: can only add a category, never remove one
  the keyword scan found.
- **Tools: none.**
- **Off-topic short-circuit:** if the Planner set `off_topic=true`, this
  agent returns immediately with `escalate=false, eligibility_reasons=[],
  dlsa_contact=null` — a query with no legal content gets no free-aid pitch
  and no DLSA push (fixed bug, see *Edge cases*).
- **In:** `{"combined_text": "...", "vertical": "wages", "composite_confidence": 0.667, "severity": "amber"}`
- **Out (real):**
  ```json
  {"escalate": true,
   "eligibility_reasons": ["Industrial workmen — factory, construction and industrial workers — are entitled to free legal aid. This covers most unpaid-wage cases."],
   "dlsa_contact": {"name": "DLSA Kanchipuram", "phone": "044-2723-XXXX",
     "hours": "Mon-Fri, 10:00-17:00", "bring": "Aadhaar/voter ID, and any proof of work...",
     "tele_law": "Ask at any Common Service Centre (CSC) / VLE for a Tele-Law session."}}
  ```
- **Out (off-topic, real):** `{"escalate": false, "eligibility_reasons": [], "dlsa_contact": null}`

### 10. Translation agent — `agents/translation_agent.py`
- **Role:** render the assembled English answer into the citizen's language
  (Hindi/Tamil/Telugu, and 200 languages total) — text → text.
- **LLM? No.** A dedicated machine-translation model, not a chat LLM — see
  *Why Translation and TTS are separate agents* below.
- **Tools: none** (direct model calls, on-device).
- **Backend** (`JB_TRANSLATION_BACKEND`): `nllb` (default) — **on-device**:
  `facebook/nllb-200-distilled-600M`, lazy-loaded via `transformers`, runs
  fully locally once cached, sentence-chunked and beam-decoded
  (`JB_NLLB_NUM_BEAMS`, default `2`, traded down from 5 for latency — NLLB
  runs on plain CPU with no NPU path and is the single slowest agent in the
  pipeline). `none` skips translation; any failure falls back to English.
- `ai4bharat/indictrans2-en-indic-dist-200M` (India-specific, in principle
  higher-quality) was tried first but is a **gated** HuggingFace repo that
  401s without manual access approval — NLLB is public/ungated, so it's what
  actually works out of the box. Swap `JB_TRANSLATION_BACKEND` +
  `config.INDICTRANS2_MODEL` back once access is approved, for the quality
  bump.
- **In:** `{"final_answer_en": "<~2000+ char full answer>", "lang": "ta"}`
- **Out (real, NLLB, full-length answer):**
  ```json
  {"final_answer_local": "சட்டம் உங்கள் சூழ்நிலையைப் பற்றி என்ன சொல்கிறது என்பது இங்கே: ஊதியக் குறியீடு, 2019 இன் கீழ், பிரிவு 17 ..."}
  ```

### 11. Output agent — `agents/output_agent.py`
- **Role:** assemble the spoken script and the multi-device
  **`signal_packet`** (AI PC → UNO Q) + `phone_message`.
- **LLM? No. Tools: none.** Pure string templating from upstream agent
  outputs (draft_answer + eligibility_reasons + deadline + dlsa_contact).
- **Three distinct answer shapes**, all real/verified:
  1. **Supported** (a built vertical, e.g. wages): rights explanation + aid
     pitch + deadline + DLSA + disclaimer (see Reasoning's sample above).
  2. **Unsupported-but-recognised** (e.g. tenancy): *"Sorry — this kind of
     legal problem is not yet supported... The best next step is to speak to
     a real lawyer for free."* + aid pitch + DLSA + disclaimer.
  3. **Off-topic** (no legal content at all): *"I couldn't find a legal
     problem in what you said. This assistant helps with legal questions —
     for example unpaid wages, a consumer complaint, or a family/domestic
     issue. Please describe what happened and I'll try to help."* — **no**
     aid pitch, **no** DLSA push.
- **Out (`signal_packet`, real):**
  ```json
  {"severity": "amber", "category": "wages", "confidence": 0.667, "deadline_days": 90,
   "dlsa": {"name": "DLSA Kanchipuram", "phone": "044-2723-XXXX", "bring": "..."},
   "qualifies_for_aid": true}
  ```

### 12. TTS agent — `agents/tts_agent.py`
- **Role:** speak the final answer back — text → audio. **Voice in → voice
  out**: only runs if `audio_bytes` was given or `want_tts=true` (a typed
  query doesn't trigger unwanted speech synthesis, e.g. in the eval harness).
- **LLM? No.** A speech-synthesis model, not a chat LLM.
- **Tools: none.**
- **Backends** (`JB_TTS_BACKEND`): `mms` (default) — Meta **MMS-TTS**
  (`facebook/mms-tts-<lang>`), a neural on-device model with a dedicated
  checkpoint per language (en/hi/ta/te) — the only backend that can correctly
  speak a translated Hindi/Tamil/Telugu answer; `pyttsx3` — offline OS voices
  (SAPI5/NSSpeech/espeak), only as good as whatever voices the OS has
  installed (a stock Windows box commonly ships English-only, verified:
  `pyttsx3.init().getProperty('voices')` returned only en-GB/en-US/zh-CN/
  zh-TW — so it silently mispronounces non-Latin-script text instead of
  failing loudly); `none` skips spoken output.
- **In:** `{"final_answer_local": "...", "lang": "ta", "audio_bytes": <mic recording>}`
- **Out (real):** `{"audio_response": <267232 bytes of valid RIFF/WAV audio>}`

---

## Voice and document are BOTH optional

Neither is mandatory, and they're never required together. A query can be
**any combination** of `text_input` / `audio_bytes` / `images`, as long as at
least one is present:

| given | works? | verified |
|---|---|---|
| text only | ✅ | (the default CLI/eval path) |
| voice only | ✅ | ASR sample above |
| document(s) only, no voice/text | ✅ | tested: `run_cli --image doc.png` with no text/audio → routes and answers correctly from OCR text alone |
| voice + document | ✅ | tested: audio + image together → both feed into `combined_text` |
| **nothing at all** | ❌ | CLI/app/API all reject with a clear "please provide..." message rather than silently running on empty input |

Fixed a Streamlit bug where uploading a document with no text/voice was
rejected by validation; `images` now counts toward "at least one input
given."

## Why Translation and TTS are separate agents, not merged

They look adjacent in the pipeline (translate the answer, then speak it) but
are fundamentally different operations:

| | Translation | TTS |
|---|---|---|
| Transform | text → text | text → audio |
| Model class | machine translation (encoder-decoder) | speech synthesis |
| Default on-device model | NLLB-200 distilled (~600M params) | MMS-TTS (per-language checkpoint) |
| Offline fallback | English passthrough | pyttsx3 (OS voices) |

Keeping them separate agents means each can **independently** degrade — if
translation fails, the English text still gets spoken by TTS (better than no
audio at all); if TTS fails, the translated text is still shown/returned to
the caller. Merging them into one "speak the answer" agent would mean one
failure silently kills both.

---

## Edge cases & known limitations

Every row below was actually run through the pipeline, not just reasoned
about — see the "verified" note on each.

| Scenario | What happens | Verified |
|---|---|---|
| **Off-topic query** ("what's the weather today") | Distinct `off_topic=true` response — no eligibility claim, no DLSA push, just "this tool helps with legal problems." | ✅ Fixed a real bug: the eligibility cue `"her "` was matching as a *substring* inside `"weather"`, so this query used to be told "you likely qualify for free legal aid as a woman." |
| **Vague/short query** ("help me") | Routes to `off_topic=true` (no keyword/LLM signal matched anything) rather than guessing a vertical. | ✅ |
| **Multiple legal issues in one query** ("my husband, a factory worker, hasn't been paid, and he beats me") | Only ONE vertical is chosen (routed to `wages`; the domestic-violence content was not addressed at all). **Known limitation, not fixed.** | ✅ tested — confirms the gap is real, not theoretical |
| **False-positive substring keyword matches** (e.g. `"site"` inside `"opposite"`) | Fixed — `text_match.py` requires whole-word matches for both Planner routing keywords and eligibility cues. | ✅ |
| **Document-only input** (no voice, no text) | Works — OCR text alone reaches the Planner via `combined_text`. | ✅ |
| **Voice + multiple documents together** | All feed into one `combined_text`; each document is OCR'd independently. | ✅ |
| **Blank / unreadable image** | OCR returns `""` at 0.0 confidence rather than inventing text; `combined_text` ends up empty or voice-only, routes normally (to `off_topic` if nothing else was said). | ✅ |
| **Invalid/unsupported language code** (e.g. `lang="xyz"`) | Translation backends reject it → falls back to English with an honest error note, doesn't crash. | ✅ |
| **Very weak/no retrieval matches** | Reasoning sets `insufficient_context=true` and the graph retries Retrieval (bounded, default max 1) before ever drafting from thin air; if still nothing, `reasoning_backend="none"`, empty draft, and the low-confidence path escalates to a human. | design-verified (bounded loop, tested in eval) |
| **Low-relevance semantic hits polluting fusion** | Semantic hits below `JB_RETRIEVAL_RELEVANCE_FLOOR` (default `0.15`) are dropped before RRF fusion. | ✅ fixed a real bug: 6+ irrelevant Legal-Services-Act sections were leaking into a wages-query reasoning prompt from the `free_aid` store's worst-available chunks |
| **Severity ≠ emotional urgency** | Severity/deadline come from `deadlines.json`'s real limitation periods per vertical, NOT from how urgent the query *sounds*. A calmly-worded wage complaint and a frantic one get the same `amber` if the underlying legal deadline is the same. This is an intentional design boundary (arch doc: "red means an actual clock is running... not vibes"), not a bug. | by design |
| **Non-Indian-language / unsupported language speech** | Not a primary use case — Whisper's multilingual model covers Hindi/Tamil/Telugu + English reasonably, but the pipeline doesn't surface a real detected-language/confidence score from the `transformers` ASR call (fixed placeholders); a genuinely foreign language may mis-transcribe silently. Not specifically tested. | ⚠️ known gap |
| **All inputs empty** (no text, no audio, no document) | CLI/app/API all reject with a clear message rather than silently invoking the graph on nothing. | ✅ |

---

## Backends at a glance

**LLM** (`JB_LLM_BACKEND`, used by Reasoning + Planner) — see [`llm.py`](llm.py):

| value | what |
|---|---|
| `geniex` (default) | Qualcomm **GenieX** (QAIRT/Genie) NPU bundle on the Snapdragon AI PC. Native SDK only builds on `win32/arm64` / `linux/aarch64` — won't `pip install` on x64, by design. Falls back to `onnx` automatically if the bundle isn't fetched or GenieX can't load. |
| `onnx` | `onnxruntime-genai` running Phi-3-mini-4k-instruct-onnx (CPU int4), fully on-device. Universal fallback — any x64/arm64 machine, no special hardware. |
| `onnx_qnn` | `onnxruntime-genai` + QNN EP on the Hexagon NPU, loading a Qualcomm AI Hub bundle (`JB_ONNX_QNN_MODEL_DIR`). |
| `openai` | any OpenAI-compatible `/v1` endpoint (e.g. a llama.cpp server). Dev convenience. |
| `extractive` | no model; grounded answer built from retrieved statute text. Always available — the guaranteed last-resort floor. |

**ASR/Vision/TTS/Translation** are already fully offline/on-device (no cloud
option to opt into): Whisper, Tesseract, MMS-TTS (fallback pyttsx3), NLLB. By
default, Reasoning/Planner **silently degrade to keyword/extractive** if the
LLM is unavailable — the pipeline never hard-fails. That silent degrade is
itself controllable, below.

### Fallback control (`JB_ALLOW_*_FALLBACK`)

Sometimes you'd rather know the real backend is live than get a good-looking
but silently-downgraded answer — e.g. proving a demo is really using the
NPU, or a UI that wants to show "reasoning temporarily unavailable." Set
these to `0` to disable silent fallback: the agent still never crashes, it
surfaces an honest `"...backend": "unavailable"` and lets the low-confidence
path escalate to a human sooner.

| var | default | effect when `0` |
|---|---|---|
| `JB_ALLOW_LLM_FALLBACK` | `0` | master switch for both agents below |
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
| `GET /health` | which backends are configured/live, all offline — for a status bar |
| `GET /kb-stores` | the legal-topic catalogue — for a "what can I ask about" menu |
| `POST /ask` | `{text_input?, audio_base64?, image_base64?, lang, want_tts}` → one JSON result (audio/image travel as base64; every other field is plain JSON) |
| `POST /ask/stream` | same request shape, but streams newline-delimited JSON: one `{"type": "agent_step", "step": {...}}` line per agent as it finishes, then `{"type": "done", "result": {...}}`. On-device inference can take 20s-2min+ with nothing to send back until the whole graph completes — this lets a UI show live, per-agent progress instead of one long blank wait |

The response is a deliberate **allowlist** (`api.py::_RESPONSE_FIELDS`), not a
dump of internal state — so new internal fields never leak into the contract
until reviewed and added there. It now includes `agent_trace` — every node in
`graph.py` is wrapped (`_instrumented`) to record `{agent, duration_ms,
status, output}` per step, which is what `/ask/stream` streams incrementally
and `/ask` returns in full at the end.

A reference React frontend (Vite + TypeScript + Tailwind) that consumes this
API lives in [`frontend/`](frontend/) — `cd justicebridge/frontend && npm
install && npm run dev`, pointed at the `uvicorn` server above.

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
3. `python -m justicebridge.build_index` (builds its FAISS index).
4. Optionally add a `deadlines.json` entry. Done — the Planner, Retrieval, and
   Risk agents all read the registry; nothing else changes.

---

## Latest eval (default backend degrading to `extractive`, 27 gold cases across
4 verticals + off-topic regression cases — corpus from the real Act PDFs)
```
Routing (vertical)     : 100%
Routing (support flag) : 100%
Citation hit@k         : 92%
Grounded (supported)   : 100%
Escalation decision    : 100%
Aid handoff present    : 100%
Severity match         : 100%
Eligibility detection  : 3/3
```
`python -m justicebridge.eval.run_eval`. On the Snapdragon AI PC with
`JB_LLM_BACKEND=geniex` this scores the real on-device reasoning path.

---

## Key environment variables
See `justicebridge/.env.example` for a ready-to-copy template.

**Backends — fully offline, no cloud API keys required.**
| var | default | purpose |
|---|---|---|
| `JB_ASR_BACKEND` | `whisper` | `whisper` (only backend, on-device) |
| `JB_VISION_BACKEND` | `tesseract` | `tesseract` (only backend, offline) |
| `JB_TTS_BACKEND` | `mms` | `mms` (default, per-language neural voices) \| `pyttsx3` \| `none` |
| `JB_TRANSLATION_BACKEND` | `nllb` | `nllb` (default, on-device, 200 languages) \| `none` |
| `JB_NLLB_MODEL` | `facebook/nllb-200-distilled-600M` | on-device translation model |
| `JB_NLLB_NUM_BEAMS` | `2` | NLLB beam width — lower = faster on CPU, slightly less fluent |
| `JB_INDICTRANS2_MODEL` | `ai4bharat/indictrans2-en-indic-dist-200M` | alt. translation model (gated HF repo, needs manual access approval) |
| `JB_LLM_BACKEND` | `geniex` | `geniex` (default, Snapdragon NPU, auto-falls back to `onnx`) \| `onnx` (CPU) \| `onnx_qnn` \| `openai` \| `extractive` |
| `JB_ONNX_MODEL_REPO` | `microsoft/Phi-3-mini-4k-instruct-onnx` | HF repo for the CPU offline LLM |
| `JB_ONNX_MODEL_SUBFOLDER` | `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` | CPU int4 build subfolder |
| `JB_ONNX_MODEL_CACHE_ROOT` | `justicebridge/models/phi3-mini-onnx` | local cache dir (downloaded once, then fully offline) |
| `JB_ONNX_MAX_LENGTH` | `4000` | Phi-3-mini-4k's context window (4096) minus headroom |
| `JB_GENIEX_MODEL` | `qualcomm/Qwen3-4B-Instruct-2507` | AI Hub bundle id or a GGUF HF repo (Snapdragon only) |
| `JB_GENIEX_DEVICE_MAP` | `qairt` | `qairt` (NPU bundle) — anything else routes to `precision`/`auto` for a GGUF model |
| `JB_ONNX_QNN_MODEL_DIR` | — | path to an AI Hub `genai_config.json` bundle dir (Snapdragon NPU) |
| `JB_OPENAI_BASE_URL` | `http://localhost:8080/v1` | for `JB_LLM_BACKEND=openai` |
| `JB_WHISPER_MODEL` | `small` | Whisper model size (or a full HF repo id) |
| `JB_WHISPER_DEVICE` / `JB_WHISPER_COMPUTE_TYPE` | `cpu` / `int8` | Whisper runtime settings |
| `JB_TESSERACT_CMD` | (Windows default path) | Tesseract-OCR binary location |
| `JB_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | retrieval embedding model |
| `JB_VECTOR_DIR` | `justicebridge/chroma_db` | FAISS index directory (name kept for back-compat) |

**Fallback control & optional LLM upgrades** — see the *Backends at a glance*
section above for the full explanation.
| var | default |
|---|---|
| `JB_ALLOW_LLM_FALLBACK` | `0` |
| `JB_ALLOW_REASONING_FALLBACK` / `JB_ALLOW_PLANNER_FALLBACK` | = master (`JB_ALLOW_LLM_FALLBACK`) |
| `JB_LLM_ASSISTED_GROUNDING` / `JB_LLM_ASSISTED_ELIGIBILITY` | `0` |

**Retrieval, risk & misc**
| var | default | purpose |
|---|---|---|
| `JB_RETRIEVAL_K` | `8` | sections retrieved per query |
| `JB_RETRIEVAL_MIN_SIM` | `0.02` | below this, Reasoning retries Retrieval |
| `JB_RETRIEVAL_RELEVANCE_FLOOR` | `0.15` | semantic hits below this are dropped before RRF fusion |
| `JB_MAX_RETRIEVAL_RETRIES` / `JB_MAX_GROUNDING_RETRIES` | `1` / `1` | bounded-loop caps (lowered from 2 — see *Reasoning agent* above) |
| `JB_DEADLINE_RED_DAYS` / `JB_DEADLINE_AMBER_DAYS` | `30` / `120` | severity thresholds |
| `JB_LOW_CONFIDENCE_ESCALATE` | `0.55` | confidence floor that forces escalation |
| `JB_DISTRICT` / `JB_STATE` | `Kanchipuram` / `Tamil Nadu` | which DLSA to surface |
| `JB_HF_OFFLINE` | `1` | keep embeddings/Whisper/translation models fully offline after first cache |
