# app/utils.py
from __future__ import annotations
import os, re, json, math, glob
from typing import List


# Try importing tiktoken, fallback to None if not installed
try:
    import tiktoken
except ImportError:
    tiktoken = None


def getenv(name: str, default: str | None = None) -> str:
    """Safe environment variable getter with default fallback."""
    return os.environ.get(name, default) if os.environ.get(name) is not None else (default or "")


def normalise_ws(text: str) -> str:
    """Collapse multiple whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, normalise whitespace."""
    paras = re.split(r"\n\s*\n", text)
    return [normalise_ws(p) for p in paras if normalise_ws(p)]


def count_tokens(s: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken if available, else fallback heuristic."""
    if tiktoken is None:
        # crude fallback: ~4 chars per token
        return max(1, math.ceil(len(s) / 4))
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(s))


def chunk_text(text: str, max_tokens: int = 900, overlap_tokens: int = 180) -> List[str]:
    """
    Chunk text into overlapping segments by token count.
    """
    paras = split_into_paragraphs(text)
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    for p in paras:
        ptok = count_tokens(p)
        if ptok > max_tokens:  # split very long paragraphs
            words = p.split()
            piece = []
            piece_tok = 0
            for w in words:
                wt = max(1, len(w) // 4)
                if piece_tok + wt > max_tokens:
                    chunks.append(" ".join(piece))
                    # overlap from end of last chunk
                    overlap = chunks[-1].split()
                    keep = max(0, int(len(overlap) * overlap_tokens / max_tokens))
                    piece = overlap[-keep:]
                    piece_tok = sum(max(1, len(x) // 4) for x in piece)
                piece.append(w)
                piece_tok += wt
            if piece:
                chunks.append(" ".join(piece))
            continue

        if buf_tokens + ptok <= max_tokens:
            buf.append(p)
            buf_tokens += ptok
        else:
            if buf:
                chunks.append("\n\n".join(buf))
                # overlap
                tail = buf[-1]
                tail_words = tail.split()
                keep = max(0, int(len(tail_words) * overlap_tokens / max_tokens))
                buf = [" ".join(tail_words[-keep:])] if keep else []
                buf_tokens = count_tokens(buf[0]) if buf else 0
            buf.append(p)
            buf_tokens += ptok

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks


def load_text_from_path(path: str) -> str:
    """Load text from PDF, Markdown or TXT."""
    from pypdf import PdfReader

    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages)

    elif path.lower().endswith((".md", ".txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        return ""


def glob_docs(root: str = "data/raw") -> List[str]:
    """Find all supported docs in a folder (PDF, MD, TXT)."""
    patterns = ["*.pdf", "*.md", "*.txt"]
    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(root, p)))
    return sorted(paths)


def ensure_dirs():
    """Ensure index directory exists."""
    os.makedirs("data/index", exist_ok=True)
