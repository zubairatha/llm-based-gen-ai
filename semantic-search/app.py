# app.py
# Minimal Streamlit UI for Similarity · MMR · Hybrid (BM25 + Semantic)
# Requires: streamlit, langchain-chroma, langchain-community, sentence-transformers, chromadb, rank-bm25

import time
from typing import List, Tuple, Dict, Literal
from collections import defaultdict

import streamlit as st
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------
# Config
# ---------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZES = [500, 1000, 1500]
COLLECTION_PREFIXES = ("rag_eval", "rag_fresh", "rag_vis")
PERSIST_DIR_TPL = "chroma_db_{cs}"

# ---------------------------
# Utilities
# ---------------------------
def _simple_tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())

def _unified_id(md: dict) -> str:
    # Prefer preassigned chunk_id, else source::start_index
    if md.get("chunk_id"):
        return md["chunk_id"]
    return f"{md.get('source','N/A')}::{md.get('start_index','N/A')}"

# ---------------------------
# Cache resources
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def open_populated_collection(chunk_size: int):
    """Open first populated collection among preferred prefixes."""
    emb = get_embeddings()
    persist_dir = PERSIST_DIR_TPL.format(cs=chunk_size)
    for pref in COLLECTION_PREFIXES:
        name = f"{pref}_{chunk_size}"
        vs = Chroma(embedding_function=emb, persist_directory=persist_dir, collection_name=name)
        count = len(vs.get(include=[]).get("ids", []))
        if count > 0:
            return vs, name, count
    return None, None, 0

# Ignore unhashable vs in cache key by using a hashable string cache_key
@st.cache_resource(show_spinner=False)
def build_bm25_from_chroma(_vs, cache_key: str, limit: int = 1_000_000):
    raw = _vs.get(include=["documents", "metadatas"], limit=limit)
    docs   = raw.get("documents", [])
    metas  = raw.get("metadatas", [])
    ids_unified, id2meta, id2text = [], {}, {}
    for i in range(len(docs)):
        uid = _unified_id(metas[i])
        ids_unified.append(uid)
        id2meta[uid] = metas[i]
        id2text[uid] = docs[i]
    tokenized = [_simple_tokenize(id2text[uid]) for uid in ids_unified]
    bm25 = BM25Okapi(tokenized)
    return bm25, ids_unified, id2meta, id2text

# ---------------------------
# Search helpers
# ---------------------------
def semantic_search_with_scores(vs: Chroma, query: str, k: int = 10) -> List[Tuple[str, float]]:
    pairs = vs.similarity_search_with_score(query, k=k)
    out = []
    for doc, dist in pairs:
        uid = _unified_id(doc.metadata)
        out.append((uid, float(dist)))  # distance: lower is better
    return out

def bm25_search(bm25: BM25Okapi, unified_ids: List[str], query: str, k: int = 10) -> List[Tuple[str, float]]:
    toks = _simple_tokenize(query)
    scores = bm25.get_scores(toks)
    rank_idx = sorted(range(len(unified_ids)), key=lambda i: -scores[i])[:k]
    return [(unified_ids[i], float(scores[i])) for i in rank_idx if scores[i] > 0.0]

def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax <= vmin:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def fuse_weighted_sum(
    sem: List[Tuple[str, float]],  # (id, distance) lower better
    lex: List[Tuple[str, float]],  # (id, bm25)   higher better
    w_sem: float = 0.6,
    w_lex: float = 0.4
) -> Dict[str, float]:
    sem_ids, sem_dists = zip(*sem) if sem else ([], [])
    sem_sims = [1.0 - d for d in sem_dists]   # coarse cosine sim
    sem_norm = _minmax_norm(list(sem_sims))
    lex_ids, lex_scores = zip(*lex) if lex else ([], [])
    lex_norm = _minmax_norm(list(lex_scores))
    scores = defaultdict(float)
    for i, _id in enumerate(sem_ids):
        scores[_id] += w_sem * sem_norm[i]
    for i, _id in enumerate(lex_ids):
        scores[_id] += w_lex * lex_norm[i]
    return scores

def fuse_rrf(sem: List[Tuple[str, float]], lex: List[Tuple[str, float]], k_rrf: int = 60) -> Dict[str, float]:
    sem_sorted = sorted(sem, key=lambda x: x[1])   # distance asc
    lex_sorted = sorted(lex, key=lambda x: -x[1])  # bm25 desc
    scores = defaultdict(float)
    for rank, (id_, _) in enumerate(sem_sorted, start=1):
        scores[id_] += 1.0 / (k_rrf + rank)
    for rank, (id_, _) in enumerate(lex_sorted, start=1):
        scores[id_] += 1.0 / (k_rrf + rank)
    return scores

def run_search(
    vs: Chroma,
    bm25: BM25Okapi,
    unified_ids: List[str],
    id2meta: Dict[str, dict],
    id2text: Dict[str, str],
    query: str,
    method: Literal["similarity", "similarity+scores", "mmr", "hybrid_weighted", "hybrid_rrf"],
    k: int,
    lambda_mult: float,
    fetch_k: int,
    w_sem: float,
    w_lex: float,
):
    t0 = time.perf_counter()
    results, info = [], {}

    if method == "similarity":
        docs = vs.similarity_search(query, k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        for d in docs:
            uid = _unified_id(d.metadata)
            results.append((uid, None, id2meta.get(uid, d.metadata), id2text.get(uid, d.page_content)))

    elif method == "similarity+scores":
        pairs = vs.similarity_search_with_score(query, k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        for d, dist in pairs:
            uid = _unified_id(d.metadata)
            results.append((uid, float(dist), id2meta.get(uid, d.metadata), id2text.get(uid, d.page_content)))

    elif method == "mmr":
        docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        for d in docs:
            uid = _unified_id(d.metadata)
            results.append((uid, None, id2meta.get(uid, d.metadata), id2text.get(uid, d.page_content)))

    elif method == "hybrid_weighted":
        sem = semantic_search_with_scores(vs, query, k=max(fetch_k, k))
        lex = bm25_search(bm25, unified_ids, query, k=max(fetch_k * 2, k))
        fused = fuse_weighted_sum(sem, lex, w_sem=w_sem, w_lex=w_lex)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        for uid, score in top:
            results.append((uid, float(score), id2meta.get(uid, {}), id2text.get(uid, "")))
        info = {"sem_candidates": len(sem), "lex_candidates": len(lex)}

    elif method == "hybrid_rrf":
        sem = semantic_search_with_scores(vs, query, k=max(fetch_k, k))
        lex = bm25_search(bm25, unified_ids, query, k=max(fetch_k * 2, k))
        fused = fuse_rrf(sem, lex, k_rrf=60)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        for uid, score in top:
            results.append((uid, float(score), id2meta.get(uid, {}), id2text.get(uid, "")))
        info = {"sem_candidates": len(sem), "lex_candidates": len(lex)}
    else:
        elapsed_ms = 0.0

    return results, elapsed_ms, info

# ---------------------------
# Minimal UI
# ---------------------------
st.set_page_config(page_title="RAG Search", layout="wide")
st.markdown("<h2 style='margin-bottom:0.1rem;'>RAG Search</h2>", unsafe_allow_html=True)
st.caption("Similarity · MMR · Hybrid (BM25 + Semantic)")

# Controls row (main page)
c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.2, 1.2])
with c1:
    cs = st.selectbox("Chunk size", CHUNK_SIZES, index=1, help="Pick which Chroma index to use")
with c2:
    method = st.selectbox(
        "Method",
        ["similarity", "similarity+scores", "mmr", "hybrid_weighted", "hybrid_rrf"],
        index=2,
        help="Choose retrieval strategy",
    )
with c3:
    k = st.slider("k", 1, 15, 5, help="Top-k results")
with c4:
    search_btn = st.button("Search", type="primary", use_container_width=True)

# Advanced parameters in an expander
with st.expander("Advanced", expanded=False):
    cA, cB, cC = st.columns(3)
    with cA:
        lambda_mult = st.slider("MMR λ", 0.0, 1.0, 0.5, 0.05, help="Balance relevance vs diversity")
    with cB:
        fetch_k = st.slider("Candidate pool (fetch_k)", 10, 100, 20, 5, help="Initial candidates for MMR/Hybrid")
    with cC:
        w_sem = st.slider("Hybrid weight (semantic)", 0.0, 1.0, 0.6, 0.05, help="Weighted fusion: semantic vs BM25")
        w_lex = 1.0 - w_sem
        st.caption(f"BM25 weight auto-set to {w_lex:.2f}")

# Query input (main page, full width)
query = st.text_input(
    "Query",
    value="renewables: storage needs vs transmission expansion",
    placeholder="Type your question…",
)

# Open index + build BM25 once per collection
vs, collection_name, count = open_populated_collection(cs)
if count == 0:
    st.warning(f"No populated collection found for {cs}. Build your Chroma index first.")
    st.stop()

cache_key = f"cs={cs}|coll={collection_name}|items={count}"
bm25, doc_ids, id2meta, id2text = build_bm25_from_chroma(vs, cache_key=cache_key)

# Run
if search_btn and query.strip():
    results, elapsed_ms, info = run_search(
        vs, bm25, doc_ids, id2meta, id2text,
        query=query.strip(),
        method=method,
        k=k,
        lambda_mult=lambda_mult,
        fetch_k=fetch_k,
        w_sem=w_sem,
        w_lex=w_lex,
    )

    # Minimal header line
    meta_line = f"{PERSIST_DIR_TPL.format(cs=cs)} / {collection_name} · {count} chunks"
    st.caption(f"{meta_line}  •  latency {elapsed_ms:.2f} ms"
               + (f"  •  candidates semi:{info.get('sem_candidates','?')} bm25:{info.get('lex_candidates','?')}" if info else ""))

    if not results:
        st.info("No results.")
    else:
        # Minimal result cards (no theme/subtopic)
        for i, (uid, score, md, txt) in enumerate(results, 1):
            st.markdown(f"**{i}.** `{uid}`")
            st.write((txt or "")[:900] + ("..." if txt and len(txt) > 900 else ""))
            st.caption(
                f"source: {md.get('source','N/A')}  |  start_index: {md.get('start_index','N/A')}  |  "
                f"chunk_id: {md.get('chunk_id','N/A')}" + (f"  |  score: {score:.4f}" if score is not None else "")
            )
            st.divider()
