# app/rag.py
from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from rank_bm25 import BM25Okapi
from app.utils import ensure_dirs, glob_docs, load_text_from_path, chunk_text

INDEX_DIR = "data/index"
META_PATH = os.path.join(INDEX_DIR, "metadata.jsonl")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.json")

DEFAULT_EMBEDDINGS_MODEL = os.environ.get(
    "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class RAGIndex:
    def __init__(self, embed_model: str = DEFAULT_EMBEDDINGS_MODEL):
        self.embedder = SentenceTransformer(embed_model)
        self.index = None  # type: ignore
        self.metadata: List[Dict[str, Any]] = []
        self.bm25 = None
        self._bm25_docs = None

    # ---------- Build ----------
    def build(self, paths: List[str], max_tokens=900, overlap_tokens=180) -> None:
        ensure_dirs()
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []

        for path in paths:
            raw = load_text_from_path(path)
            if not raw:
                continue
            chs = chunk_text(raw, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            for i, ch in enumerate(chs):
                chunks.append(ch)
                metas.append({
                     "source": path,
                    "source_name": os.path.basename(path),   # <— add
                    "chunk_id": i,
                    # Optional: quick & dirty page guess from chunk index
                    "page_hint": i + 1
                })

        if not chunks:
            raise RuntimeError("No text found. Add documents to data/raw/")

        X = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.metadata = metas

        # Save index + metadata
        faiss.write_index(self.index, FAISS_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")

        # BM25 for lexical fallback / quick text access
        tokenized = [c.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        with open(BM25_PATH, "w", encoding="utf-8") as f:
            json.dump({"docs": chunks}, f)

    # ---------- Load ----------
    def load(self) -> None:
        ensure_dirs()
        if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
            raise FileNotFoundError("Index not found. Run: python -m app.rag --reindex")

        self.index = faiss.read_index(FAISS_PATH)
        self.metadata = [json.loads(line) for line in open(META_PATH, "r", encoding="utf-8")]

        if os.path.exists(BM25_PATH):
            data = json.load(open(BM25_PATH, "r", encoding="utf-8"))
            tokenized = [d.split() for d in data["docs"]]
            self.bm25 = BM25Okapi(tokenized)
            self._bm25_docs = data["docs"]

    # ---------- Retrieve ----------
    def retrieve(self, query: str, k: int = 5, rerank: bool = False) -> List[Dict[str, Any]]:
        assert self.index is not None, "Index not loaded. Call load() first."
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q, k)
        idxs = idxs[0]
        scores = scores[0]

        docs = self._bm25_docs  # may be None if BM25 not loaded
        results: List[Dict[str, Any]] = []

        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            meta = self.metadata[int(i)]
            text = docs[int(i)] if docs else ""

            if not text:
                # Fallback: re-read source and re-chunk, then pick chunk_id
                raw = load_text_from_path(meta["source"]) or ""
                chs = chunk_text(raw)
                text = chs[meta["chunk_id"]] if meta["chunk_id"] < len(chs) else raw[:1200]

            results.append(
                {
                    "rank": rank,
                    "score": float(s),
                    "text": text,
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"],
                }
            )

        # Optional cross-encoder re-rank
        if rerank and len(results) > 1:
            try:
                ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                pairs = [(query, r["text"]) for r in results]
                rr = ce.predict(pairs)
                for r, sc in zip(results, rr):
                    r["rerank_score"] = float(sc)
                results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            except Exception:
                pass

        # Add simple citation id
        for i, r in enumerate(results, start=1):
            r["cite_id"] = f"[{i}]"
        return results


# ---------- CLI ----------
def _reindex():
    ensure_dirs()
    paths = glob_docs("data/raw")
    idx = RAGIndex()
    idx.build(paths)
    print(f"Indexed {len(idx.metadata)} chunks from {len(paths)} files → {FAISS_PATH}")


def _test(query: str, k: int = 5):
    idx = RAGIndex()
    idx.load()
    res = idx.retrieve(query, k=k)
    for r in res:
        print(r["cite_id"], r["source"], f"(chunk {r['chunk_id']})", "score=", round(r["score"], 3))
        print(r["text"][:200].replace("\n", " "), "...\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--test", type=str, default="")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if args.reindex:
        _reindex()
    elif args.test:
        _test(args.test, k=args.k)
    else:
        ap.print_help()
