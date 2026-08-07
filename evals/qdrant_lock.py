"""Open local-disk Qdrant for evals even when uvicorn already holds the lock."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "rag" / "qdrant_local"


def ensure_qdrant_for_eval() -> Path:
    """
    Prefer the live index. If another process has the flock (typical when
    uvicorn is running), copy to a temp dir and point QDRANT_PATH there.
    """
    path = Path(os.getenv("QDRANT_PATH", str(DEFAULT)))
    if not path.exists():
        raise FileNotFoundError(f"Qdrant path missing: {path}")

    # Fast path: try exclusive open briefly via qdrant; on lock, fall back.
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(path=str(path))
        client.close()
        os.environ["QDRANT_PATH"] = str(path)
        return path
    except RuntimeError as e:
        if "already accessed" not in str(e).lower() and "locked" not in str(e).lower():
            raise
    except Exception as e:
        # portalocker / AlreadyLocked often wrapped as RuntimeError; also catch broadly
        msg = str(e).lower()
        if "already" not in msg and "lock" not in msg:
            raise

    tmp = Path(tempfile.mkdtemp(prefix="qdrant_eval_"))
    print(f"[eval] Qdrant locked by another process — using copy at {tmp}")
    shutil.copytree(path, tmp, dirs_exist_ok=True)
    os.environ["QDRANT_PATH"] = str(tmp)

    try:
        from app.services.rag_retrieval import reset_qdrant_client
        reset_qdrant_client()
    except Exception:
        pass

    return tmp
