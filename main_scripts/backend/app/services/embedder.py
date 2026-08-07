"""Shared SentenceTransformer singleton for the whole backend.

All RAG / NLU / nutrition paths must use this — loading MiniLM three times
per request floods the logs and wastes RAM/CPU.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

_EMBED_MODEL = None
_LOAD_FAILED = False


def get_shared_embed_model():
    global _EMBED_MODEL, _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            print("[Embed] Loaded shared all-MiniLM-L6-v2 once")
        except Exception as e:
            print(f"[Embed] Unavailable: {e}")
            _LOAD_FAILED = True
            return None
    return _EMBED_MODEL


def encode_texts(texts: List[str], *, normalize: bool = True) -> np.ndarray:
    model = get_shared_embed_model()
    if model is None:
        raise RuntimeError("Embedding model unavailable")
    vecs = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def encode_one(text: str, *, normalize: bool = True) -> List[float]:
    return encode_texts([text or ""], normalize=normalize)[0].tolist()
