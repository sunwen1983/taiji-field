"""Grounding encoder v1 (Phase 1): situation text -> hexagram by resonance.

Understanding = resonance: a situation is encoded as a vector and matched
against one document per hexagram by cosine similarity. The documents are
the authored one-line readings in :mod:`taiji_field.documents` (plus the
hexagram name). There is no token prediction anywhere here — the encoder
only *retrieves* the closest pre-existing state.

Backend, in order of preference:

1. ``sentence-transformers`` with ``BAAI/bge-small-zh-v1.5`` (Chinese-first,
   CPU-only, ~24M parameters). If that model cannot be downloaded, fall
   back to ``paraphrase-multilingual-MiniLM-L12-v2``.
2. If ``sentence-transformers`` is unavailable at all (or both models fail),
   a dependency-free character-bigram TF-IDF encoder
   (:class:`TfidfEncoder`) is used instead. It is much weaker — it matches
   surface characters, not meaning — and the encoder reports which backend
   is in use so results are never mistaken for embedding resonance.

The 64 hexagram document embeddings are cached on disk
(``taiji_field/data/embeddings.npz``) with the model name and a hash of the
document texts, so the demo and the exam harness run fast after the first
build. Rebuild with ``python scripts/build_embeddings.py`` (or
:meth:`GroundingEncoder.rebuild_cache`).

bge usage note: for symmetric similarity (situation <-> short document)
the texts are encoded plain, with no retrieval instruction prefix — the
standard setup for sentence-similarity with this model family.
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from pathlib import Path

import numpy as np

from taiji_field.data import load_hexagrams
from taiji_field.documents import hexagram_documents

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
FALLBACK_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CACHE = Path(__file__).parent / "data" / "embeddings.npz"

_BY_KW = None


def _by_kw():
    global _BY_KW
    if _BY_KW is None:
        _BY_KW = {h.king_wen: h for h in load_hexagrams()}
    return _BY_KW


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of ``a`` and vector ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b)
    denom = np.where(denom == 0, 1.0, denom)
    return (a @ b) / denom


# ---------------------------------------------------------------------------
# TF-IDF fallback (no dependencies, no model download)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+")


def _char_ngrams(text: str) -> list[str]:
    """Character unigrams + bigrams (Chinese) plus lowercase word tokens."""
    text = text.lower()
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    grams = list(chars)
    grams += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    grams += _TOKEN_RE.findall(text)
    return grams


class TfidfEncoder:
    """Dependency-free fallback: character n-gram TF-IDF + cosine.

    Honest limit: this matches shared characters, not meaning. A situation
    that paraphrases a reading without reusing its characters will score
    near zero. It exists so the pipeline still runs (and tests still pass)
    on a machine with no model download — not as a rival to embeddings.
    """

    backend = "tfidf-char-ngram"

    def __init__(self, documents: list[dict] | None = None):
        self.documents = documents if documents is not None else hexagram_documents()
        vocab: dict[str, int] = {}
        rows = []
        for doc in self.documents:
            counts: dict[str, int] = {}
            for g in _char_ngrams(doc["text"]):
                counts[g] = counts.get(g, 0) + 1
            rows.append(counts)
            for g in counts:
                if g not in vocab:
                    vocab[g] = len(vocab)
        self._vocab = vocab
        n_docs = len(self.documents)
        df = np.zeros(len(vocab), dtype=np.float64)
        for counts in rows:
            for g in counts:
                df[vocab[g]] += 1
        idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        mat = np.zeros((n_docs, len(vocab)), dtype=np.float64)
        for i, counts in enumerate(rows):
            for g, c in counts.items():
                mat[i, vocab[g]] = (1 + math.log(c)) * idf[vocab[g]]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._doc_matrix = mat / np.where(norms == 0, 1.0, norms)
        self._idf = idf

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(len(self._vocab), dtype=np.float64)
            counts: dict[str, int] = {}
            for g in _char_ngrams(text):
                counts[g] = counts.get(g, 0) + 1
            for g, c in counts.items():
                if g in self._vocab:
                    vec[self._vocab[g]] = (1 + math.log(c)) * self._idf[self._vocab[g]]
            n = np.linalg.norm(vec)
            rows.append(vec / n if n > 0 else vec)
        return np.array(rows, dtype=np.float64)

    def doc_embeddings(self) -> np.ndarray:
        return self._doc_matrix

    def similarities(self, text: str) -> list[dict]:
        vec = self.encode([text])[0]
        sims = cosine_similarity(self._doc_matrix, vec)
        return _ranked(self.documents, sims)

    def match(self, text: str, top_k: int = 3) -> list[dict]:
        return self.similarities(text)[:top_k]


# ---------------------------------------------------------------------------
# Sentence-transformer backend
# ---------------------------------------------------------------------------


class SentenceTransformerEncoder:
    """``sentence-transformers`` backend (bge-small-zh-v1.5 preferred)."""

    def __init__(self, model_name: str, documents: list[dict] | None = None,
                 cache_path: str | Path | None = DEFAULT_CACHE):
        from sentence_transformers import SentenceTransformer

        self.backend = f"sentence-transformers:{model_name}"
        self.model_name = model_name
        self.documents = documents if documents is not None else hexagram_documents()
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._model = SentenceTransformer(model_name, device="cpu")
        self._doc_embeddings = self._load_or_build_doc_embeddings()

    @property
    def dim(self) -> int:
        return int(self._doc_embeddings.shape[1])

    def _doc_text_hash(self) -> str:
        joined = "\n".join(d["text"] for d in self.documents)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def _load_or_build_doc_embeddings(self) -> np.ndarray:
        if self.cache_path is not None and self.cache_path.exists():
            try:
                data = np.load(self.cache_path, allow_pickle=False)
                if (
                    str(data["model"]) == self.model_name
                    and str(data["doc_hash"]) == self._doc_text_hash()
                    and data["embeddings"].shape[0] == len(self.documents)
                ):
                    return np.asarray(data["embeddings"], dtype=np.float64)
            except Exception:
                pass  # corrupt/stale cache: rebuild below
        emb = self.encode([d["text"] for d in self.documents])
        if self.cache_path is not None:
            self._save_cache(emb)
        return emb

    def _save_cache(self, emb: np.ndarray) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            embeddings=emb.astype(np.float32),
            model=np.array(self.model_name),
            doc_hash=np.array(self._doc_text_hash()),
            king_wen=np.array([d["king_wen"] for d in self.documents]),
        )

    def rebuild_cache(self) -> np.ndarray:
        emb = self.encode([d["text"] for d in self.documents])
        if self.cache_path is not None:
            self._save_cache(emb)
        self._doc_embeddings = emb
        return emb

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(emb, dtype=np.float64)

    def doc_embeddings(self) -> np.ndarray:
        return self._doc_embeddings

    def similarities(self, text: str) -> list[dict]:
        vec = self.encode([text])[0]
        sims = cosine_similarity(self._doc_embeddings, vec)
        return _ranked(self.documents, sims)

    def match(self, text: str, top_k: int = 3) -> list[dict]:
        return self.similarities(text)[:top_k]


def _ranked(documents: list[dict], sims: np.ndarray) -> list[dict]:
    order = np.argsort(-sims)
    out = []
    for idx in order:
        doc = documents[int(idx)]
        h = _by_kw()[doc["king_wen"]]
        out.append(
            {
                "king_wen": doc["king_wen"],
                "bits": doc["bits"],
                "name_zh": doc["name_zh"],
                "unicode": h.unicode,
                "pinyin": h.pinyin,
                "english": h.english,
                "score": float(sims[int(idx)]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


class GroundingEncoder:
    """Situation text -> hexagram by resonance.

    Tries ``BAAI/bge-small-zh-v1.5`` first, then
    ``paraphrase-multilingual-MiniLM-L12-v2``; if ``sentence-transformers``
    is unavailable or both models fail, falls back to
    :class:`TfidfEncoder` and says so (``.backend``).
    """

    def __init__(self, model_name: str | None = None,
                 cache_path: str | Path | None = DEFAULT_CACHE,
                 backend: str = "auto"):
        self.documents = hexagram_documents()
        self.backend = "tfidf-char-ngram"  # overwritten on success
        self._inner = None
        if backend in ("auto", "sentence-transformers"):
            candidates = (
                [model_name] if model_name else [DEFAULT_MODEL, FALLBACK_MODEL]
            )
            last_error: Exception | None = None
            for name in candidates:
                try:
                    self._inner = SentenceTransformerEncoder(
                        name, self.documents, cache_path
                    )
                    self.backend = self._inner.backend
                    self.model_name = name
                    return
                except Exception as e:  # noqa: BLE001 — try next candidate
                    last_error = e
                    warnings.warn(
                        f"sentence-transformers model {name!r} failed "
                        f"({e}); trying next backend"
                    )
            warnings.warn(
                "all sentence-transformer models failed "
                f"(last error: {last_error}); using TF-IDF fallback"
            )
        self._inner = TfidfEncoder(self.documents)
        self.backend = self._inner.backend
        self.model_name = None

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._inner.encode(texts)

    def doc_embeddings(self) -> np.ndarray:
        return self._inner.doc_embeddings()

    def similarities(self, text: str) -> list[dict]:
        """All 64 hexagrams ranked by cosine similarity to ``text``."""
        if not text or not text.strip():
            raise ValueError("situation text must not be empty")
        return self._inner.similarities(text)

    def match(self, text: str, top_k: int = 3) -> list[dict]:
        return self.similarities(text)[:top_k]

    def rebuild_cache(self) -> np.ndarray | None:
        if hasattr(self._inner, "rebuild_cache"):
            return self._inner.rebuild_cache()
        return None
