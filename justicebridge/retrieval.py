"""
Hybrid retrieval engine — the Retrieval agent's engine, now backed by a
REGISTRY of per-topic KB stores.

Each KB store (kb_registry.KB_STORES) is its own Chroma vector collection
inside one persist dir, plus its own in-memory BM25 index. The Planner picks
which store ids to search; retrieve() queries exactly those stores and fuses
results. So "which vector DB does the next agent look at" is a real, explicit
routing decision, not a metadata filter afterthought.

Semantic (Chroma) + keyword (BM25) rankings are fused per store with
Reciprocal Rank Fusion (RRF), then merged across stores by fused score.

Public API:
    retrieve(query, kb_stores=[...], k=6) -> (sections, retrieval_sim)
    build_index()  -> (re)builds every store's collection from corpus.json
"""

import json
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from . import config
from .kb_registry import KB_STORES

_embeddings = None
_docs_by_store = None            # store_id -> [Document]
_vectordb_by_store = {}          # store_id -> Chroma
_bm25_by_store = {}              # store_id -> BM25Retriever


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def _store_id_for_vertical(vertical):
    """corpus.json tags chunks with `vertical` (== the store id in build)."""
    return vertical


def _load_docs_by_store():
    with open(config.CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    by_store = {sid: [] for sid in KB_STORES}
    for c in corpus:
        if not c.get("section_no"):
            continue  # skip preamble/front-matter — not a citable provision
        sid = c.get("vertical")
        if sid not in by_store:
            continue
        by_store[sid].append(Document(
            # prepend the section title so semantic search matches on what a
            # section is ABOUT, not only its dense statutory body.
            page_content=(f"{c.get('title','')}. {c['text']}" if c.get("title") else c["text"]),
            metadata={
                "act": c["act"],
                "vertical": sid,
                "section_no": c["section_no"],
                "title": c.get("title", ""),
                "chunk_id": c["chunk_id"],
            },
        ))
    return by_store


def _collection_dir(store_id):
    return str(Path(config.CHROMA_DIR) / KB_STORES[store_id]["collection"])


def build_index():
    """Build (or rebuild) one persistent Chroma collection per KB store."""
    by_store = _load_docs_by_store()
    emb = _get_embeddings()
    for sid, docs in by_store.items():
        if not docs:
            print(f"  [{sid}] no chunks in corpus — skipping (build its corpus first)")
            continue
        cdir = _collection_dir(sid)
        print(f"  [{sid}] embedding {len(docs)} chunks -> {cdir}")
        Chroma.from_documents(
            documents=docs,
            embedding=emb,
            collection_name=KB_STORES[sid]["collection"],
            persist_directory=cdir,
        )
    print("All KB store collections built.")


def _ensure_store_loaded(store_id):
    global _docs_by_store
    if _docs_by_store is None:
        _docs_by_store = _load_docs_by_store()

    if store_id not in _vectordb_by_store:
        cdir = _collection_dir(store_id)
        if not Path(cdir).exists():
            # build just this store's collection on demand
            docs = _docs_by_store.get(store_id, [])
            if not docs:
                _vectordb_by_store[store_id] = None
                _bm25_by_store[store_id] = None
                return
            _vectordb_by_store[store_id] = Chroma.from_documents(
                documents=docs,
                embedding=_get_embeddings(),
                collection_name=KB_STORES[store_id]["collection"],
                persist_directory=cdir,
            )
        else:
            _vectordb_by_store[store_id] = Chroma(
                collection_name=KB_STORES[store_id]["collection"],
                persist_directory=cdir,
                embedding_function=_get_embeddings(),
            )
        docs = _docs_by_store.get(store_id, [])
        if docs:
            bm = BM25Retriever.from_documents(docs)
            bm.k = config.RETRIEVAL_K
            _bm25_by_store[store_id] = bm
        else:
            _bm25_by_store[store_id] = None


def _search_store(store_id, query, k):
    """Return (list[(doc, fused_score)], max_semantic_sim) for one store."""
    _ensure_store_loaded(store_id)
    vdb = _vectordb_by_store.get(store_id)
    bm25 = _bm25_by_store.get(store_id)
    if vdb is None:
        return [], 0.0

    try:
        sem_scored = vdb.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        sem_scored = []
    semantic = [d for d, _s in sem_scored]
    sim = max((s for _d, s in sem_scored), default=0.0)
    sim = max(0.0, float(sim))

    keyword = bm25.invoke(query) if bm25 else []

    scores, lookup = {}, {}
    for rank, doc in enumerate(semantic):
        cid = doc.metadata["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        lookup[cid] = doc
    for rank, doc in enumerate(keyword):
        cid = doc.metadata["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        lookup[cid] = doc
    fused = [(lookup[cid], sc) for cid, sc in scores.items()]
    return fused, sim


def retrieve(query: str, kb_stores=None, k: int = None, vertical: str = None):
    """Hybrid retrieve across the Planner-selected KB stores.

    kb_stores: list of store ids to search (the Planner's decision). If None,
    falls back to [vertical] for back-compat, else all stores.
    Returns (sections, retrieval_sim).
    """
    k = k or config.RETRIEVAL_K
    if kb_stores is None:
        kb_stores = [vertical] if vertical else list(KB_STORES.keys())

    # Split the search set into the substantive topic store(s) and the
    # cross-cutting always-include store(s) (free_aid). The substantive topic
    # owns most citation slots; free_aid gets a small reserved quota so a wage
    # query cites wage law, not mostly Legal-Services-Act sections.
    primary = [s for s in kb_stores if s in KB_STORES and not KB_STORES[s].get("always_include")]
    aid = [s for s in kb_stores if s in KB_STORES and KB_STORES[s].get("always_include")]
    aid_quota = min(1, k // 3) if aid else 0

    def _merged(store_ids, want):
        fused = []
        sim = 0.0
        for sid in store_ids:
            f, s = _search_store(sid, query, k)
            fused.extend(f)
            sim = max(sim, s)
        fused.sort(key=lambda x: x[1], reverse=True)
        out, seen = [], set()
        for doc, score in fused:
            cid = doc.metadata["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            out.append((doc, score))
            if len(out) >= want:
                break
        return out, sim

    primary_hits, primary_sim = _merged(primary or aid, k - aid_quota)
    aid_hits, aid_sim = _merged(aid, aid_quota) if aid_quota else ([], 0.0)
    max_sim = max(primary_sim, aid_sim)

    sections, seen = [], set()
    for doc, score in primary_hits + aid_hits:
        cid = doc.metadata["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        sections.append({
            "act": doc.metadata["act"],
            "section_no": doc.metadata["section_no"],
            "title": doc.metadata["title"],
            "text": doc.page_content,
            "store": doc.metadata["vertical"],
            "score": round(score, 5),
        })
    return sections, max_sim


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        build_index()
    q = "my employer has not paid my wages for two months"
    secs, sim = retrieve(q, kb_stores=["wages", "free_aid"], k=3)
    print(f"query={q!r}  retrieval_sim={sim:.3f}")
    for s in secs:
        print(f"  [{s['store']}] {s['act']} s.{s['section_no']}: {s['title'][:45]}  score={s['score']}")
