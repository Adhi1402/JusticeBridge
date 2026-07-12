"""
Command-line driver for the full JusticeBridge pipeline — text in, grounded
verdict + multi-device signal packet out. This is the fastest way to exercise
and demo the AI-PC brain without the phone/UNO-Q tiers attached.

Usage:
    python -m justicebridge.run_cli "they haven't paid my wages for two months"
    python -m justicebridge.run_cli "my landlord is evicting me"        # stub -> handoff
    python -m justicebridge.run_cli --lang ta "..."                     # translate output
    python -m justicebridge.run_cli                                     # interactive

Manual voice/document testing (no browser needed):
    # record yourself with any app (Voice Recorder, phone voice memo, etc.)
    # and save as a .wav, then:
    python -m justicebridge.run_cli --audio path\\to\\recording.wav

    # take a photo of a document/notice and:
    python -m justicebridge.run_cli --image path\\to\\photo.jpg

    # multiple documents in one query (--image can repeat):
    python -m justicebridge.run_cli --image page1.jpg --image page2.jpg

    # both together (voice + document in one query — neither is mandatory,
    # you can give text/voice/document in any combination, at least one):
    python -m justicebridge.run_cli --audio recording.wav --image notice.jpg
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path

# Windows consoles default to cp1252 and choke on emoji/Tamil output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from .graph import get_app
from . import config, llm


SEV_ICON = {"red": "[RED]", "amber": "[AMBER]", "green": "[GREEN]"}

LOG_DIR = Path(__file__).resolve().parent / "logs"

# Keys that are noisy/binary and not worth dumping into the log.
_HIDE_IN_LOG = {"audio_bytes", "audio_response", "image", "images"}


def _setup_logging(verbose_console: bool):
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("justicebridge")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)

    if verbose_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


def _clean_for_log(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in _HIDE_IN_LOG:
            out[k] = f"<{type(v).__name__}, omitted>"
        else:
            out[k] = v
    return out


def run_once(query=None, lang="en", audio_path=None, image_paths=None, verbose=True):
    logger = logging.getLogger("justicebridge")
    app = get_app()
    init = {"lang": lang}
    if query:
        init["text_input"] = query
    if audio_path:
        with open(audio_path, "rb") as f:
            init["audio_bytes"] = f.read()
    if image_paths:
        from PIL import Image
        init["images"] = [Image.open(p) for p in image_paths]

    logger.info("=" * 72)
    logger.info(f"PIPELINE START  input={_clean_for_log(init)}")

    state = {}
    step_no = 0
    t_start = time.perf_counter()
    for update in app.stream(init, stream_mode="updates"):
        for node_name, node_delta in update.items():
            step_no += 1
            t = time.perf_counter() - t_start
            if verbose:
                logger.info(f"[{t:7.3f}s] STEP {step_no:02d} -> agent: {node_name}")
                for k, v in _clean_for_log(node_delta or {}).items():
                    v_str = repr(v)
                    if len(v_str) > 500:
                        v_str = v_str[:500] + "... (truncated, see log file)"
                    logger.info(f"           {k} = {v_str}")
            logger.debug(f"STEP {step_no} [{node_name}] full delta: {node_delta!r}")
            if node_delta:
                for k, v in node_delta.items():
                    if k == "error" and k in state:
                        state[k] = state[k] + v
                    else:
                        state[k] = v

    logger.info(f"PIPELINE END  total_time={time.perf_counter() - t_start:.3f}s  "
                f"steps={step_no}")
    return state


def _print_report(state):
    print("\n" + "=" * 72)
    if "transcript" in state:
        print(f"ASR TRANSCRIPT ({state.get('asr_confidence', 0):.3f} confidence): "
              f"{state.get('transcript','')!r}")
    if "doc_text" in state:
        print(f"OCR TEXT ({state.get('vision_confidence', 0):.3f} confidence): "
              f"{state.get('doc_text','')!r}")
    if "transcript" in state or "doc_text" in state:
        print("-" * 72)
    print(f"VERTICAL : {state.get('vertical')}   (supported={state.get('supported')})")
    print(f"REASONING BACKEND : {state.get('reasoning_backend', 'n/a')}")
    print(f"RETRIEVAL SIM : {state.get('retrieval_sim', 0):.3f}   "
          f"COMPOSITE CONF : {state.get('composite_confidence', 0):.3f}")
    sev = state.get("severity", "green")
    print(f"SEVERITY : {SEV_ICON.get(sev,'')} {sev.upper()}   "
          f"DEADLINE ~{state.get('deadline_days')} days   "
          f"GROUNDED : {state.get('grounded')}")
    if state.get("ungrounded_claims"):
        print(f"STRIPPED UNGROUNDED CLAIMS : {state['ungrounded_claims']}")

    print("\n--- CITED SECTIONS ---")
    for c in state.get("citations", []):
        print(f"  • {c['act']}, Section {c['section_no']} ({c.get('title','')})")

    print("\n--- ELIGIBILITY (Section 12) ---")
    reasons = state.get("eligibility_reasons", [])
    if reasons:
        for r in reasons:
            print(f"  ✓ {r}")
    else:
        print("  (no automatic category matched from the words used)")

    print("\n--- SPOKEN ANSWER (English) ---")
    print(state.get("final_answer_en", "").strip())

    if state.get("lang") not in ("en", "unknown") and state.get("final_answer_local"):
        print(f"\n--- SPOKEN ANSWER ({state.get('lang')}) ---")
        print(state.get("final_answer_local").strip())

    print("\n--- SIGNAL PACKET (AI PC → UNO Q) ---")
    print(json.dumps(state.get("signal_packet", {}), indent=2, ensure_ascii=False))
    if state.get("error"):
        print("\n--- NON-FATAL NOTES ---")
        for e in state["error"]:
            print(f"  ! {e}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="the citizen's problem, in their words")
    ap.add_argument("--lang", default="en", help="output language: en | ta | hi | te")
    ap.add_argument("--audio", help="path to a .wav recording to transcribe (ASR test)")
    ap.add_argument("--image", action="append", default=[],
                    help="path to a document photo to OCR (Vision test); repeat for multiple documents")
    ap.add_argument("--quiet", action="store_true",
                    help="don't print step-by-step agent output to the console "
                         "(full detail is still written to the log file)")
    args = ap.parse_args()

    _setup_logging(verbose_console=not args.quiet)

    print(f"LLM backend configured : {config.LLM_BACKEND}  "
          f"(live={llm.is_live()})   [falls back to extractive if unavailable]")
    print(f"Vision backend configured : {config.VISION_BACKEND}  "
          f"(ASR: {config.ASR_BACKEND}, TTS: {config.TTS_BACKEND}, all offline)")

    if args.query or args.audio or args.image:
        query = " ".join(args.query) if args.query else None
        _print_report(run_once(query, args.lang, audio_path=args.audio,
                                image_paths=args.image, verbose=not args.quiet))
        return

    print("\nInteractive mode — type a legal problem (blank to quit).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        _print_report(run_once(q, args.lang, verbose=not args.quiet))


if __name__ == "__main__":
    main()
