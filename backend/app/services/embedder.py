"""Shared embedding model singletons for the backend.

Two models, two vector spaces — do not mix them:

  all-MiniLM-L6-v2 (384-d)
      Text↔text: fitness_guidelines, exercise_semantic, NLU prototypes,
      caption re-rank. Use encode_texts / encode_one.

  clip-ViT-B-32 (512-d)
      Text↔image: guideline_images only. Ingest uses encode_images_clip*;
      query time uses encode_text_clip. CLIP text vectors are weaker than
      MiniLM for pure text search — never use CLIP for fitness_guidelines.

Models load once per process; callers must handle None / RuntimeError if
download or init fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

_EMBED_MODEL = None
_LOAD_FAILED = False

_CLIP_MODEL = None
_CLIP_LOAD_FAILED = False
CLIP_MODEL_NAME = "clip-ViT-B-32"   # open source (OpenAI CLIP weights via sentence-transformers)
CLIP_DIM = 512


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


# ── CLIP (multimodal) ─────────────────────────────────────────────────────────

def get_shared_clip_model():
    """Loads once, cached. Returns None if unavailable (caller must handle)."""
    global _CLIP_MODEL, _CLIP_LOAD_FAILED
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL
    # Do not permanently latch forever — a first-boot HF download / processor
    # glitch used to set _CLIP_LOAD_FAILED and kill multimodal for the whole
    # uvicorn process. Allow a retry on the next call.
    if _CLIP_LOAD_FAILED:
        _CLIP_LOAD_FAILED = False
    try:
        from sentence_transformers import SentenceTransformer
        # trust_remote_code helps with CLIP processor configs on newer transformers
        try:
            _CLIP_MODEL = SentenceTransformer(CLIP_MODEL_NAME, trust_remote_code=True)
        except TypeError:
            _CLIP_MODEL = SentenceTransformer(CLIP_MODEL_NAME)
        print(f"[Embed] Loaded shared {CLIP_MODEL_NAME} once ({CLIP_DIM}-dim)")
    except Exception as e:
        print(f"[Embed] CLIP unavailable: {e}")
        _CLIP_LOAD_FAILED = True
        _CLIP_MODEL = None
        return None
    return _CLIP_MODEL


def encode_text_clip(text: str, *, normalize: bool = True) -> List[float]:
    """Embed a query/caption string into CLIP's joint text-image space."""
    model = get_shared_clip_model()
    if model is None:
        raise RuntimeError(
            "CLIP model unavailable — cannot run multimodal RAG. "
            "Install sentence-transformers / check network for model download "
            "(first run pulls ~600MB of weights from HuggingFace)."
        )
    vec = model.encode([text or ""], normalize_embeddings=normalize, show_progress_bar=False)[0]
    return np.asarray(vec, dtype=np.float32).tolist()


def encode_image_clip(image_path: Union[str, Path], *, normalize: bool = True) -> List[float]:
    """Embed a single image file into CLIP's joint text-image space."""
    model = get_shared_clip_model()
    if model is None:
        raise RuntimeError("CLIP model unavailable — cannot embed images")
    from PIL import Image
    img = Image.open(str(image_path)).convert("RGB")
    vec = model.encode(img, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32).tolist()


def encode_images_clip_batch(image_paths: List[Union[str, Path]], *, normalize: bool = True) -> List[List[float]]:
    model = get_shared_clip_model()
    if model is None:
        raise RuntimeError("CLIP model unavailable — cannot embed images")
    from PIL import Image
    imgs = [Image.open(str(p)).convert("RGB") for p in image_paths]
    vecs = model.encode(imgs, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32).tolist()
