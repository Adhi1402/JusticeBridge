"""
Corpus builder (v4) — built from the REAL Act PDFs, not the indiacode JSON's
scraped `sections` text array.

For each act configured under `config.ACTS_BY_VERTICAL`, this:
  1. Looks up the act's `pdf_url_en` in indiacode (3).json and downloads the
     PDF (cached under pdfs/ — re-run is instant once downloaded).
  2. Extracts the full text with pdfplumber.
  3. Chunks the ENTIRE raw text with LangChain's RecursiveCharacterTextSplitter
     (`add_start_index=True`) — every character of the PDF ends up in some
     chunk. Nothing is filtered or dropped before chunking.
  4. Separately locates section-heading positions with a loose regex ("17.
     Time limit for payment of wages.—") and uses them ONLY to TAG each chunk
     with its best-effort `section_no`/`title` (whichever heading precedes the
     chunk's start offset) — citation metadata, not a content filter. A chunk
     that starts before any detected heading (preamble, long title, table of
     contents) is kept and tagged section_no=None rather than being dropped.

  Earlier version of this file used the regex matches to carve up the text
  BEFORE chunking, which silently dropped everything before the first
  matched heading and any section whose heading didn't match the pattern.
  This version guarantees full-text coverage; tagging is best-effort only.

Scaling to a new vertical is a one-line change in config.py:

    ACTS_BY_VERTICAL = {
        "wages": ["The Code on Wages, 2019"],
        "tenancy": ["The Model Tenancy Act, 2021"],   # <- add here
        ...
    }

...then re-run this file. Nothing else in the codebase needs to change: the
Planner already routes on the vertical key, and retrieval.py already filters
Chroma by the `vertical` metadata field.

Run:  python -m justicebridge.build_corpus [--vertical wages]
Then: python -m justicebridge.build_index
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

PDF_DIR = config.PDF_DIR   # package-local pdfs/ (self-contained repo)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Matches headings like:  "17. Time limit for payment of wages.—"
# or "Section 17.Time limit..." — loose on purpose; PDF text extraction line
# wraps and spacing vary enough that exact-title anchoring drops sections.
SECTION_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:Section\s+)?(\d{1,3}[A-Z]?)\.\s*([A-Z][^\n.]{3,120})[.—-]",
)


# ---------------------------------------------------------------------------
# 1. Load act metadata (pdf_url_en, act_year) from indiacode (3).json
# ---------------------------------------------------------------------------

def _load_acts_meta():
    with open(config.INDIACODE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {a["short_title"]: a for a in data["acts"]}


# ---------------------------------------------------------------------------
# 2. Download (or reuse cached) PDF
# ---------------------------------------------------------------------------

def _download_pdf(act_meta: dict) -> Path | None:
    url = act_meta.get("pdf_url_en")
    short_title = act_meta["short_title"]
    if not url:
        print(f"  no pdf_url_en for {short_title}, skipping")
        return None

    PDF_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", short_title)[:60]
    dest = PDF_DIR / f"{safe_name}.pdf"
    if dest.exists():
        print(f"  using cached PDF: {dest.name}")
        return dest

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  downloaded: {short_title}")
        time.sleep(1)  # be polite to indiacode.nic.in
        return dest
    except Exception as e:
        print(f"  FAILED to download {short_title}: {e}")
        return None


def _extract_pdf_text(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 3. Find heading positions (for TAGGING only — not used to drop content)
# ---------------------------------------------------------------------------

def _find_heading_positions(full_text: str):
    """Returns [(char_offset, section_no, title), ...] sorted by offset."""
    out = []
    for m in SECTION_HEADING_RE.finditer(full_text):
        out.append((m.start(), m.group(1), m.group(2).strip()))
    out.sort(key=lambda x: x[0])
    return out


def _tag_for_offset(headings, offset):
    """Best-effort: the last heading at or before `offset`. None if the chunk
    starts before any detected heading (preamble/long title/contents) —
    such chunks are still kept, just without a section_no."""
    section_no, title = None, ""
    for h_offset, no, t in headings:
        if h_offset <= offset:
            section_no, title = no, t
        else:
            break
    return section_no, title


# ---------------------------------------------------------------------------
# 4. Chunk the ENTIRE raw text — nothing filtered out — then tag each chunk
#    with its best-effort section_no/title via the heading positions above.
# ---------------------------------------------------------------------------

def _chunk_full_text(full_text, act_title, act_year, vertical):
    headings = _find_heading_positions(full_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    docs = splitter.create_documents([full_text])

    out = []
    for j, d in enumerate(docs):
        offset = d.metadata["start_index"]
        section_no, title = _tag_for_offset(headings, offset)
        out.append({
            "act": act_title,
            "act_year": act_year,
            "vertical": vertical,
            "section_no": section_no,       # may be None (preamble/front-matter) — kept, not dropped
            "title": title,
            "chunk_id": f"{act_title}|c{j}|off{offset}",
            "text": d.page_content,
        })
    return out


# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------

def build(only_vertical: str | None = None):
    acts_meta = _load_acts_meta()
    verticals = config.ACTS_BY_VERTICAL
    if only_vertical:
        verticals = {only_vertical: verticals[only_vertical]}

    corpus = []
    for vertical, act_titles in verticals.items():
        for act_title in act_titles:
            print(f"Processing [{vertical}]: {act_title}")
            act_meta = acts_meta.get(act_title)
            if not act_meta:
                print(f"  WARNING: '{act_title}' not found in indiacode JSON — skipping")
                continue

            pdf_path = _download_pdf(act_meta)
            if not pdf_path:
                continue

            full_text = _extract_pdf_text(pdf_path)
            chunks = _chunk_full_text(full_text, act_title, act_meta.get("act_year"), vertical)
            untagged = sum(1 for c in chunks if c["section_no"] is None)
            corpus.extend(chunks)
            print(f"  -> {len(chunks)} chunks (full text, nothing dropped); "
                  f"{untagged} untagged (preamble/front-matter)")

    # Merge with any existing corpus for verticals we didn't touch this run,
    # so `--vertical wages` doesn't wipe out other verticals already built.
    existing = []
    if Path(config.CORPUS_FILE).exists() and only_vertical:
        with open(config.CORPUS_FILE, "r", encoding="utf-8") as f:
            existing = [c for c in json.load(f) if c.get("vertical") not in verticals]

    final_corpus = existing + corpus

    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    with open(config.CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_corpus, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(final_corpus)} total chunks to {config.CORPUS_FILE}")
    from collections import Counter
    for act, n in Counter(c["act"] for c in final_corpus).items():
        print(f"  {act}: {n} chunks")
    print("Next: python -m justicebridge.build_index")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertical", help="only (re)build this vertical, e.g. wages", default=None)
    args = ap.parse_args()
    build(only_vertical=args.vertical)
