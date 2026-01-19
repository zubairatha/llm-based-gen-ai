# app.py
# Minimal Streamlit UI — Similarity · MMR · Hybrid (BM25 + Semantic)
# (+3) Compare different (free) embedding models inside the app
# Requirements: streamlit, langchain-chroma, langchain-community, sentence-transformers, chromadb, rank-bm25

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
CHUNK_SIZES = [500, 1000, 1500]
BASE_PREFIXES = ("rag_eval", "rag_fresh", "rag_vis")  # used to harvest texts if we need to (re)build
PERSIST_DIR_TPL = "chroma_db_{cs}_{modelkey}"        # per (chunk, model) store

# Embedding model options (free, HF sentence-transformers). Keys become folder-safe.
EMBED_OPTIONS = {
    "all-MiniLM-L6-v2 (384d)": "sentence-transformers/all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1 (384d)": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "bge-small-en-v1.5 (384d)": "BAAI/bge-small-en-v1.5",
    "e5-small-v2 (384d)": "intfloat/e5-small-v2",
}

def _model_key(name: str) -> str:
    # safe key for dirs/collections
    return name.lower().replace("/", "_").replace("-", "_").replace(".", "_")

# ---------------------------
# Utilities
# ---------------------------
def _simple_tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())

def _unified_id(md: dict) -> str:
    if md.get("chunk_id"):
        return md["chunk_id"]
    return f"{md.get('source','N/A')}::{md.get('start_index','N/A')}"

# ---------------------------
# Cache resources
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource(show_spinner=True)
def _open_collection(persist_dir: str, collection_name: str, model_name: str):
    emb = get_embeddings(model_name)
    vs = Chroma(embedding_function=emb, persist_directory=persist_dir, collection_name=collection_name)
    count = len(vs.get(include=[]).get("ids", []))
    return vs, count

@st.cache_resource(show_spinner=True)
def _harvest_any_existing_texts(cs: int) -> Tuple[List[str], List[dict]]:
    """
    Open any populated base collection for this chunk size (regardless of model),
    and harvest (documents, metadatas). Used to seed new model collections.
    """
    # try without model suffix for backward compatibility
    # look for default dirs chroma_db_{cs}/*; if not found, try our new pattern for some default model key guesses
    candidates = [f"chroma_db_{cs}"]  # legacy
    # also check any per-model dirs that might exist:
    # not strictly necessary to enumerate; the legacy collection should exist from your earlier runs.
    for persist_dir in candidates:
        for pref in BASE_PREFIXES:
            name = f"{pref}_{cs}"
            try:
                # try with a default embedding just to access documents; embedding dim doesn't matter for get()
                emb = get_embeddings(EMBED_OPTIONS["all-MiniLM-L6-v2 (384d)"])
                vs = Chroma(embedding_function=emb, persist_directory=persist_dir, collection_name=name)
                raw = vs.get(include=["documents", "metadatas"], limit=1_000_000)
                docs = raw.get("documents", [])
                metas = raw.get("metadatas", [])
                if docs:
                    return docs, metas
            except Exception:
                continue
    # If nothing found, return empty (the UI will inform the user)
    return [], []

@st.cache_resource(show_spinner=True)
def build_or_open_model_collection(cs: int, model_name: str):
    """
    Try to open a per-(chunk, model) collection. If absent, harvest texts from any base collection,
    then (re)embed with the chosen model and persist.
    """
    modelkey = _model_key(model_name)
    persist_dir = PERSIST_DIR_TPL.format(cs=cs, modelkey=modelkey)
    coll_name = f"rag_{modelkey}_{cs}"

    # 1) Try open if already built
    vs, count = _open_collection(persist_dir, coll_name, model_name)
    if count > 0:
        return vs, persist_dir, coll_name, count, False  # not freshly built

    # 2) Harvest texts from any existing base collection
    docs, metas = _harvest_any_existing_texts(cs)
    if not docs:
        # Nothing to build from
        return vs, persist_dir, coll_name, 0, False

    # 3) Re-embed with selected model into a new per-model store
    texts = docs
    metadatas = metas
    emb = get_embeddings(model_name)
    vs_new = Chroma.from_texts(
        texts=texts,
        embedding=emb,
        metadatas=metadatas,
        persist_directory=persist_dir,
        collection_name=coll_name,
    )
    count_new = len(vs_new.get(include=[]).get("ids", []))
    return vs_new, persist_dir, coll_name, count_new, True  # freshly built

# NOTE: Unhashable param fix — prefix _vs and use cache_key
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
    return [(_unified_id(d.metadata), float(dist)) for d, dist in pairs]

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

def fuse_weighted_sum(sem: List[Tuple[str, float]], lex: List[Tuple[str, float]], w_sem: float = 0.6, w_lex: float = 0.4) -> Dict[str, float]:
    sem_ids, sem_dists = zip(*sem) if sem else ([], [])
    sem_sims = [1.0 - d for d in sem_dists]  # crude sim
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
    sem_sorted = sorted(sem, key=lambda x: x[1])   # dist asc
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
# Minimal UI w/ model selection + compare
# ---------------------------
st.set_page_config(page_title="RAG Search (Models)", layout="wide")
st.markdown("<h2 style='margin-bottom:0.1rem;'>RAG Search</h2>", unsafe_allow_html=True)
st.caption("Similarity · MMR · Hybrid (BM25 + Semantic) · Model comparison")

top1, top2, top3, top4, top5 = st.columns([1.1, 1.5, 1.3, 1.1, 1.0])
with top1:
    cs = st.selectbox("Chunk size", CHUNK_SIZES, index=1)
with top2:
    model_label = st.selectbox("Embedding model", list(EMBED_OPTIONS.keys()), index=0)
    model_name = EMBED_OPTIONS[model_label]
with top3:
    method = st.selectbox("Method", ["similarity", "similarity+scores", "mmr", "hybrid_weighted", "hybrid_rrf"], index=2)
with top4:
    k = st.slider("k", 1, 15, 5)
with top5:
    compare_toggle = st.toggle("Compare 2 models", value=False, help="Run the same query against two models")

# Optional comparison model
if compare_toggle:
    _, cmod, _ = st.columns([0.2, 1.6, 0.2])
    with cmod:
        model_label_2 = st.selectbox("Second model", list(EMBED_OPTIONS.keys()), index=2, key="second_model")
        model_name_2 = EMBED_OPTIONS[model_label_2]
        if model_name_2 == model_name:
            st.warning("Pick a different model for comparison.")

with st.expander("Advanced", expanded=False):
    colA, colB, colC = st.columns(3)
    with colA:
        lambda_mult = st.slider("MMR λ", 0.0, 1.0, 0.5, 0.05)
    with colB:
        fetch_k = st.slider("Candidate pool (fetch_k)", 10, 100, 20, 5)
    with colC:
        w_sem = st.slider("Hybrid weight (semantic)", 0.0, 1.0, 0.6, 0.05)
        w_lex = 1.0 - w_sem
        st.caption(f"BM25 weight auto-set to {w_lex:.2f}")

query = st.text_input("Query", value="renewables: storage needs vs transmission expansion", placeholder="Type your question…")
go = st.button("Search", type="primary", use_container_width=True)

def _run_pipeline_for_model(cs: int, model_name: str, model_label: str):
    vs, persist_dir, coll_name, count, built = build_or_open_model_collection(cs, model_name)
    if count == 0:
        return None, None, None, f"No data for {cs} / {model_label}. Build your base collection first."
    cache_key = f"cs={cs}|model={model_label}|coll={coll_name}|items={count}"
    bm25, doc_ids, id2meta, id2text = build_bm25_from_chroma(vs, cache_key=cache_key)
    return (vs, bm25, doc_ids, id2meta, id2text, persist_dir, coll_name, count, built), None

if go and query.strip():
    # Primary model
    pack1, err1 = _run_pipeline_for_model(cs, model_name, model_label)
    if err1:
        st.error(err1)
        st.stop()

    (vs1, bm251, ids1, md1, txt1, dir1, coll1, cnt1, built1) = pack1
    res1, t1, info1 = run_search(vs1, bm251, ids1, md1, txt1, query.strip(), method, k, lambda_mult, fetch_k, w_sem, w_lex)

    st.caption(f"{dir1} / {coll1} · {cnt1} chunks · latency {t1:.2f} ms"
               + (f" · built now with {model_label}" if built1 else f" · cached {model_label}"))
    if not res1:
        st.info("No results.")
    else:
        for i, (uid, score, md, txt) in enumerate(res1, 1):
            st.markdown(f"**{i}.** `{uid}`")
            st.write((txt or "")[:900] + ("..." if txt and len(txt) > 900 else ""))
            st.caption(
                f"source: {md.get('source','N/A')}  |  start_index: {md.get('start_index','N/A')}  |  "
                f"chunk_id: {md.get('chunk_id','N/A')}" + (f"  |  score: {score:.4f}" if score is not None else "")
            )
            st.divider()

    # Comparison model
    if compare_toggle and (not err1) and model_name_2 != model_name:
        pack2, err2 = _run_pipeline_for_model(cs, model_name_2, model_label_2)
        if err2:
            st.error(err2)
        else:
            (vs2, bm252, ids2, md2, txt2, dir2, coll2, cnt2, built2) = pack2
            res2, t2, info2 = run_search(vs2, bm252, ids2, md2, txt2, query.strip(), method, k, lambda_mult, fetch_k, w_sem, w_lex)
            st.subheader("Comparison")
            st.caption(f"{dir2} / {coll2} · {cnt2} chunks · latency {t2:.2f} ms"
                       + (f" · built now with {model_label_2}" if built2 else f" · cached {model_label_2}"))
            if not res2:
                st.info("No results for comparison model.")
            else:
                for i, (uid, score, md, txt) in enumerate(res2, 1):
                    st.markdown(f"**{i}.** `{uid}`")
                    st.write((txt or "")[:900] + ("..." if txt and len(txt) > 900 else ""))
                    st.caption(
                        f"source: {md.get('source','N/A')}  |  start_index: {md.get('start_index','N/A')}  |  "
                        f"chunk_id: {md.get('chunk_id','N/A')}" + (f"  |  score: {score:.4f}" if score is not None else "")
                    )
                    st.divider()
