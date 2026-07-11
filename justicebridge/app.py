"""
JusticeBridge — Streamlit frontend for the AI-PC brain.

Voice-first, big buttons, colour-coded severity, human handoff. Shows the
recognised speech (STT) and extracted document text (OCR) so the user can
confirm they were understood, then speaks the answer back (TTS).

Run:
    pip install streamlit
    streamlit run justicebridge/app.py

Backends (ASR / OCR / TTS / LLM) are all selectable in the sidebar and each
degrades gracefully — Sarvam (cloud, Indian languages) or on-device
(Whisper / Tesseract / pyttsx3 / GenieX-extractive).
"""

import io
import os
import json

import streamlit as st
from PIL import Image

# allow "streamlit run justicebridge/app.py" (script, not package) to import
try:
    from . import config
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from justicebridge import config

SEV = {
    "red":   ("#c0392b", "🔴", "Act now"),
    "amber": ("#e67e22", "🟠", "Act soon"),
    "green": ("#27ae60", "🟢", "For your awareness"),
}
LANGS = {"English": "en", "தமிழ் (Tamil)": "ta", "हिन्दी (Hindi)": "hi", "తెలుగు (Telugu)": "te"}

st.set_page_config(page_title="JusticeBridge", page_icon="⚖️", layout="centered")
st.title("⚖️ JusticeBridge")
st.caption("On-device, multilingual legal help · connects you to FREE legal aid")

# ---------------------------------------------------------------------------
# Sidebar — language + backend selectors. These set env vars BEFORE the graph
# is imported/built, so config picks them up.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    lang_label = st.radio("Language / மொழி", list(LANGS.keys()), index=0)
    lang = LANGS[lang_label]

    st.divider()
    st.caption("Speech-to-text")
    asr = st.selectbox("ASR backend", ["sarvam", "whisper"],
                       index=["sarvam", "whisper"].index(config.ASR_BACKEND))
    st.caption("Document OCR")
    vision = st.selectbox("Vision backend", ["sarvam", "tesseract"],
                          index=["sarvam", "tesseract"].index(config.VISION_BACKEND))
    st.caption("Speak the answer back")
    tts = st.selectbox("TTS backend", ["sarvam", "pyttsx3", "none"],
                       index=["sarvam", "pyttsx3", "none"].index(config.TTS_BACKEND))

    os.environ["JB_ASR_BACKEND"] = asr
    os.environ["JB_VISION_BACKEND"] = vision
    os.environ["JB_TTS_BACKEND"] = tts

    st.divider()
    st.write(f"**LLM backend:** `{config.LLM_BACKEND}`")
    key_set = bool(config.SARVAM_API_KEY)
    st.caption(("✅ SARVAM_API_KEY detected" if key_set
                else "⚠️ No SARVAM_API_KEY — Sarvam backends will fall back "
                     "to on-device (Whisper/Tesseract/pyttsx3)."))

# import the graph AFTER env vars are set so config reflects the selection
from justicebridge.graph import get_app  # noqa: E402
from justicebridge import config as _cfg  # noqa: E402
# reflect sidebar choices in the live config module
_cfg.ASR_BACKEND, _cfg.VISION_BACKEND, _cfg.TTS_BACKEND = asr, vision, tts

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.subheader("1 · Tell us what happened")
text_q = st.text_area("In your own words (optional if you record/scan)",
                      placeholder="e.g. I worked two months and my contractor hasn't paid my wages…")
audio = st.audio_input("…or speak (tap to record)")
docs = st.file_uploader("…or scan paper(s) — notice / contract / bill",
                        type=["png", "jpg", "jpeg"], accept_multiple_files=True)

go = st.button("Get help", type="primary", use_container_width=True)

if go:
    audio_bytes = audio.getvalue() if audio else None
    images = [Image.open(io.BytesIO(f.getvalue())) for f in (docs or [])]

    # Voice and document are BOTH optional — only text/voice/document as a
    # whole needs at least one; you never need voice AND a document together.
    if not (text_q.strip() or audio_bytes or images):
        st.warning("Please type, speak, or upload a document first.")
        st.stop()

    init = {"lang": lang, "want_tts": True}
    if text_q.strip():
        init["text_input"] = text_q.strip()
    if audio_bytes:
        init["audio_bytes"] = audio_bytes
    if images:
        init["images"] = images  # all uploaded documents, OCR'd and merged

    with st.spinner("Thinking…"):
        state = get_app().invoke(init)

    # ---- show what was heard / read (STT + OCR) ----
    if state.get("transcript"):
        st.markdown("**📝 We heard:**")
        st.info(state["transcript"])
    if state.get("doc_text"):
        with st.expander("📄 Text extracted from your document(s)", expanded=True):
            st.text(state["doc_text"])

    # ---- severity signal (mirrors the UNO Q light) ----
    sev = state.get("severity", "green")
    color, icon, label = SEV.get(sev, SEV["green"])
    days = state.get("deadline_days")
    deadline_txt = f" · act within ~{max(1, round(days/7))} weeks" if days else ""
    qualifies = bool(state.get("eligibility_reasons"))
    sub = (f"{(state.get('vertical') or '').title()}"
           + (" · you likely qualify for FREE legal aid" if qualifies else ""))
    st.markdown(
        f"""<div style="background:{color};color:white;padding:18px;border-radius:12px;
        font-size:20px;font-weight:600;text-align:center">
        {icon} {label}{deadline_txt}<br>
        <span style="font-size:15px;font-weight:400">{sub}</span></div>""",
        unsafe_allow_html=True,
    )

    st.subheader("2 · What the law says")
    answer = state.get("final_answer_local") or state.get("final_answer_en", "")
    st.write(answer)

    # ---- speak the answer back (TTS) ----
    if state.get("audio_response"):
        st.markdown("**🔊 Spoken answer:**")
        st.audio(state["audio_response"], format="audio/wav")
    elif tts != "none":
        st.caption("(Spoken answer unavailable — TTS backend not reachable.)")

    reasons = state.get("eligibility_reasons", [])
    if reasons:
        st.success("**You likely qualify for FREE legal aid (Section 12):**\n\n" +
                   "\n".join(f"- {r}" for r in reasons))

    d = state.get("dlsa_contact") or {}
    if d:
        st.subheader("3 · Talk to a real lawyer — free")
        st.info(f"**{d.get('name','')}**  \n📞 {d.get('phone','')}  \n"
                f"🕑 {d.get('hours','')}  \n🎒 Bring: {d.get('bring','')}  \n"
                f"💻 {d.get('tele_law','')}")

    with st.expander("Cited statute sections (grounding)"):
        for c in state.get("citations", []):
            st.write(f"- **{c['act']}, Section {c['section_no']}** — {c.get('title','')}")
        if state.get("ungrounded_claims"):
            st.caption(f"Stripped ungrounded claims: {state['ungrounded_claims']}")

    with st.expander("Pipeline internals (KB routing + signal packet)"):
        st.write(f"**KB stores searched:** {state.get('kb_stores')}")
        st.write(f"**Planner backend:** {state.get('planner_backend')} · "
                 f"**Reasoning backend:** {state.get('reasoning_backend','n/a')}")
        st.code(json.dumps(state.get("signal_packet", {}), indent=2, ensure_ascii=False), "json")
        st.caption(f"retrieval_sim={state.get('retrieval_sim',0):.2f} · "
                   f"composite_confidence={state.get('composite_confidence',0):.2f}")
        if state.get("error"):
            st.caption("Notes: " + " | ".join(state["error"]))
