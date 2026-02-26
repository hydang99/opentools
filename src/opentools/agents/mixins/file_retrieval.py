"""
Hybrid retrieval over long text (e.g. large tool outputs).

Provides BM25 + dense (embedding) search with optional reranking. Used when a tool
returns very long text so the agent can retrieve only relevant snippets for the query.
"""
from __future__ import annotations
import re, time, faiss, torch, numpy as np, time
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Preprocessing functions

def tokenize(text: str):
    """Split text into lowercase word tokens (alphanumeric sequences)."""
    pattern = re.compile(r"\w+")
    return pattern.findall(text.lower())

def tokenize_set(text: str) -> set:
    """Return the set of word tokens from text (for overlap scoring)."""
    return set(tokenize(text))

def split_sentences(text: str) -> List[str]:
    """Split text into sentences on period/question/exclamation followed by space."""
    pieces = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in pieces if p.strip()]

def sentence_window(query: str, text: str, radius: int = 4) -> str:
    """Return a window of sentences around the one most overlapping the query tokens."""
    sents = split_sentences(text)
    if not sents:
        return text[:400]
    q = tokenize_set(query)
    scored = [(len(q & tokenize_set(s)), i) for i, s in enumerate(sents)]
    scored.sort(reverse=True)
    c = scored[0][1] if scored else 0
    lo, hi = max(0, c - radius), min(len(sents), c + radius + 1)
    return " ".join(sents[lo:hi]).strip()

def check_cuda_capability(device_id: int):
    """Return a GPU device id with enough free memory (<80% used), or try the next device."""
    props = torch.cuda.get_device_properties(device_id)
    total = props.total_memory // (1024 * 1024)
    used = torch.cuda.memory_allocated(device_id) // (1024 * 1024)
    if used / total > 0.8:
        return check_cuda_capability(device_id + 1)
    return device_id

def select_device() -> str:
    """Return the best available device string: cuda (with memory check), mps, or cpu."""
    if torch.cuda.is_available():
        return f"cuda:{check_cuda_capability(0)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def chunk_by_tokens(text: str, tok: AutoTokenizer, target: int = 480, overlap: int = 80):
    """Split text into overlapping token-based chunks; returns list of dicts with id and text."""
    # Silence long-sequence warnings for chunking only
    try:
        tok.model_max_length = 10**9
    except Exception:
        pass
    encode = tok(text, add_special_tokens=False)
    ids = encode.input_ids
    out = []
    start = 0
    i = 0
    while start < len(ids):
        end = min(start + target, len(ids))
        window_ids = ids[start:end]
        chunk_text = tok.decode(window_ids, skip_special_tokens=True)
        out.append({"id": i, "text": chunk_text})
        i += 1
        if end == len(ids):
            break
        start = max(end - overlap, start + 1)
    return out

def build_flat_ip(dim: int) -> faiss.Index:
    """Build a FAISS flat index for inner-product (cosine when vectors are normalized)."""
    return faiss.IndexFlatIP(dim)

def build_ivf_index(mat: np.ndarray, nlist: int = 4096, nprobe: int = 16) -> faiss.Index:
    """Build a FAISS IVF index for approximate inner-product search on large matrices."""
    dim = mat.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    sample_size = min(200_000, mat.shape[0])
    sample = mat[np.random.choice(mat.shape[0], sample_size, replace=False)] if mat.shape[0] > sample_size else mat
    index.train(sample)
    index.add(mat)
    index.nprobe = nprobe
    return index

def rrf_score(rank_positions: List[int], k: int = 60) -> float:
    """Compute Reciprocal Rank Fusion score from a list of rank positions."""
    return sum(1.0 / (k + r) for r in rank_positions)

def fuse_rrf(bm_ids: List[int], dense_ids: List[int], k: int) -> List[int]:
    """Fuse BM25 and dense rank lists using Reciprocal Rank Fusion; return top-k chunk indices."""
    pos_b = {i: r for r, i in enumerate(bm_ids)}
    pos_d = {i: r for r, i in enumerate(dense_ids)}
    seen = set(pos_b) | set(pos_d)
    fused = []
    for i in seen:
        ranks = []
        if i in pos_b: ranks.append(pos_b[i])
        if i in pos_d: ranks.append(pos_d[i])
        fused.append((i, rrf_score(ranks)))
    fused.sort(key=lambda x: x[1], reverse=True)
    if not fused:
        if dense_ids:
            fused = [(i, 1.0 / (60 + r)) for r, i in enumerate(dense_ids)]
        elif bm_ids:
            fused = [(i, 1.0 / (60 + r)) for r, i in enumerate(bm_ids)]
    return [i for i, _ in fused[:k]]

@dataclass
class RetrieverConfig:
    """Configuration for encoder, reranker, chunk size, and IVF threshold."""
    encoder_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chunk_tokens: int = 480
    overlap_tokens: int = 80
    use_ivf_threshold: int = 100000

class TextHybridRetriever:
    """Hybrid retriever over long text: BM25 + dense embeddings, optional reranker, returns relevant snippets."""

    def __init__(self, text: Optional[str] = None, encoder_model: str = "BAAI/bge-small-en-v1.5", reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        """Initialize retriever; if text is provided, build the index (chunk, BM25, FAISS) from it."""
        self.cfg = RetrieverConfig(encoder_model=encoder_model, reranker_model=reranker_model)
        self.device = select_device()

        tok_name = self.cfg.encoder_model if "bge" in self.cfg.encoder_model.lower() else "sentence-transformers/all-MiniLM-L6-v2"
        self.chunk_tok = AutoTokenizer.from_pretrained(tok_name)
        try:
            self.chunk_tok.model_max_length = 10**9
        except Exception:
            pass

        self.chunks = []
        self.corpus_tokens = []
        self.bm25= None

        self.embedder = None
        self.prepare_text = None
        self.faiss_index = None

        self.tok_cache = None
        self.mdl_cache = None
        self.reranker_model_name = None

        if text is not None:
            self.build(text)

    def build(self, text: str):
        """Chunk text, build BM25 and FAISS index (flat or IVF for large corpora)."""
        t0 = time.time()
        self.chunks = chunk_by_tokens(text, self.chunk_tok, self.cfg.chunk_tokens, self.cfg.overlap_tokens)
        self.corpus_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        self.embedder = SentenceTransformer(self.cfg.encoder_model, device=self.device)
        def _prep(texts: List[str], is_query: bool):
            if "bge" in self.cfg.encoder_model.lower():
                pref = "query: " if is_query else "passage: "
                return [pref + t for t in texts]
            return texts
        self.prepare_text = _prep

        texts = [c["text"] for c in self.chunks]
        mat = self.embedder.encode(self.prepare_text(texts, is_query=False), normalize_embeddings=True)
        mat = np.asarray(mat, dtype="float32")
        N = len(self.chunks)
        dim = mat.shape[1]
        if N >= 100000:
            nlist = max(1024, min(65536, int((N ** 0.5) * 16)))  
            nprobe = min(64, max(8, int(nlist * 0.02)))          

            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

            train_sz = min(200_000, N)
            train_idx = np.random.choice(N, train_sz, replace=False)
            index.train(mat[train_idx])

            index.add(mat)
            index.nprobe = nprobe
            self.faiss_index = index
        else:
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(mat)

    def bm25_search(self, query: str, k: int) -> List[int]:
        """Return top-k chunk indices from BM25 search for the query."""
        if self.bm25 is None or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return order.tolist()

    def dense_search(self, query: str, k: int) -> List[int]:
        """Return top-k chunk indices from FAISS dense (embedding) search for the query."""
        qv = self.embedder.encode(self.prepare_text([query], is_query=True), normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        D, I = self.faiss_index.search(qv, k)
        return [i for i in I[0] if i >= 0]

    def ensure_reranker(self):
        """Lazily load the reranker tokenizer and model if not already loaded."""
        if self.tok_cache is not None:
            return
        model_name = self.cfg.reranker_model
        try:
            self.tok_cache = AutoTokenizer.from_pretrained(model_name)
            load_kwargs = {"dtype": torch.float16} if "cuda" in self.device else {}
            self.mdl_cache = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs).to(self.device)
            self.reranker_model_name = model_name
            print(f"[reranker] loaded {model_name} on {self.device}")
        except Exception as exc:
            print(f"[reranker] failed to load {model_name}: {exc}")
            self.tok_cache = None
            self.mdl_cache = None
            self.reranker_model_name = None

    def rerank(self, query: str, cand_ids: List[int], top_n: int = 40, batch_size: int = 16) -> List[int]:
        """Rerank candidate chunk indices by relevance to the query; returns reordered indices."""
        self.ensure_reranker()
        if self.mdl_cache is None or not cand_ids:
            return cand_ids
        top_n = min(top_n, len(cand_ids))
        passages = [self.chunks[i]["text"] for i in cand_ids[:top_n]]
        scores: List[float] = []
        for b in range(0, top_n, batch_size):
            p = passages[b:b+batch_size]
            batch = self.tok_cache([query]*len(p), p, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                s = self.mdl_cache(**batch).logits[:, 0].tolist()
            scores.extend(s)
        ordered = [x for _, x in sorted(zip(scores, cand_ids[:top_n]), key=lambda z: z[0], reverse=True)]
        return ordered + cand_ids[top_n:]

    def retrieve(self, query: str, top_k: int = 6, candidates: int = 120, use_dense: bool = True, use_reranker: bool = True, window: int = 4):
        """Run hybrid retrieval (BM25 + dense, optional rerank) and return top_k snippets as {idx: {chunk_id, snippet}}."""
        bm_ids = self.bm25_search(query, k=candidates)
        dense_ids = self.dense_search(query, k=candidates) if use_dense and self.faiss_index is not None else []
        fused_ids = fuse_rrf(bm_ids, dense_ids, k=candidates)
        if use_reranker and fused_ids:
            fused_ids = self.rerank(query, fused_ids, top_n=max(30, top_k*4), batch_size=16)
        out = {}
        k = 0
        for i in fused_ids[:top_k]:
            txt = self.chunks[i]["text"]
            snip = sentence_window(query, txt, radius=window)
            out[k] = {"chunk_id": int(i), "snippet": snip}
            k += 1
        return out