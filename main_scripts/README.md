# main_scripts — curated Adaptive Fitness Planner sources

This folder is a **snapshot of the important application scripts** for reading, review, handoff, and documentation. It is **not** a runnable second copy of the product by itself (no Qdrant index, no SQLite DB, no `node_modules`, no media dumps).

## What’s here

| Path | Contents |
|------|----------|
| `backend/` | FastAPI entry, conversation agent, plan engine, RAG/NLU services, MCP tools, routers, schemas, models, ingest scripts |
| `frontend/src/` | Chat / Plan / Progress UI + API client + key components |
| `evals/` | RAG, exercise, and agent eval runners + gold JSONL datasets |
| `data/download_sources.py` | Guideline PDF downloader |
| `MODULE_OUTLINE.md` | High-level outline of the module |
| `MODULE_DOCUMENTATION.md` | **Full in-depth documentation** (start here for understanding) |
| `MANIFEST.txt` | Exact file list copied into this folder |
| `SETUP_AND_TEST_GUIDE.md` | Original setup / pipeline runbook |
| `BUGFIX_CHANGELOG.md` | Historical bugfix notes |

## How to use

1. Read **`MODULE_OUTLINE.md`** for the map.
2. Read **`MODULE_DOCUMENTATION.md`** for every subsystem in depth.
3. Open matching files under `backend/` / `frontend/` / `evals/` for the code itself.
4. For a live run, use the **repo root** project (`backend/`, `frontend/`), not this mirror alone.

## Regenerating this folder

From the repo root (after code changes you want mirrored):

```bash
# Re-copy is done by the agent / maintainer; see MANIFEST.txt for the file set.
# Prefer editing the live tree under backend/ and frontend/, then re-sync here.
```

**Do not** treat `main_scripts/` as the source of truth for runtime — the live tree under the repo root is.
