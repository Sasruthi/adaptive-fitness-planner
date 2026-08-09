"""
Guideline PDF demo-photo extraction
===================================
Crops demonstration photos from page-templated guideline PDFs and writes a
page → URL map consumed by chunking and CLIP ingest.

Why a custom extractor
----------------------
These booklets repeat a full-page watermark/background on every page. A naive
"export all embedded images" pass returns that artwork twice per page. This
script keeps only placements that look like real demo photos.

Heuristic (per image placement: xref + bbox)
--------------------------------------------
1. Prefer placements mostly inside the visible page canvas.
2. Drop placements covering >45% of page width AND height (watermark).
3. Remaining on-canvas placements are demo photos for that page.
4. Fallback: if none remain, accept off-canvas placements that still fail the
   watermark size test (spread layouts where the only demo thumb is off-page).

Pipeline position
-----------------
  extract_pdf_images.py  →  pdf_images_map.json + PNG crops
       ├─→ chunk_all_sources.py     (stamps chunk.image_urls by same page)
       └─→ embed_images_clip.py     (CLIP vectors in guideline_images)

Inputs
------
- PDFs under backend/data/adaptive-fitness-planner-data/raw/** (skips structured/)
- Filename → source_id via chunk_all_sources.FILENAME_TO_SOURCE_ID
  (unknown PDFs get FILE:<stem>)

Outputs
-------
- PNG: backend/static/guideline_images/{source_id}/p{page:03d}_{n}.png
- Map: .../chunks/pdf_images_map.json
    { "<source_id>": { "<page_number>": ["/static/guideline_images/...png", ...] } }
  page_number is 1-indexed (fitz index + 1), matching chunk page_number.
- If all_chunks.json exists, rewrites each row's image_urls from the map.

Run:
    python backend/rag/extract_pdf_images.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # backend/
DATA_ROOT    = PROJECT_ROOT / "data" / "adaptive-fitness-planner-data"
RAW_DIR      = DATA_ROOT / "raw"
CHUNK_DIR    = DATA_ROOT / "chunks"
STATIC_DIR   = PROJECT_ROOT / "static" / "guideline_images"
MAP_FILE     = CHUNK_DIR / "pdf_images_map.json"
CHUNK_FILE   = CHUNK_DIR / "all_chunks.json"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Placement covering more than this fraction of the page in BOTH dimensions
# is treated as full-page background art, not a demo photo.
BACKGROUND_FRACTION = 0.45

# Minimum placement bbox area (PDF points²). Filters bullet icons / noise.
# Placement size ≠ source pixel size — thumbs can still pass this bar.
MIN_PLACEMENT_AREA = 40.0

# Same folders chunk_all_sources treats as DB-only (not guideline PDF RAG).
SKIP_FOLDERS = {"structured"}


def _visible_rect(rect: "fitz.Rect", page_rect: "fitz.Rect", tol: float = 1.0) -> bool:
    """True if the placement is mostly within the visible page canvas."""
    if rect.x1 < -tol or rect.x0 > page_rect.width + tol:
        return False
    if rect.y1 < -tol or rect.y0 > page_rect.height + tol:
        return False
    return True


def _is_watermark_placement(rect: "fitz.Rect", page_rect: "fitz.Rect") -> bool:
    frac_w = rect.width / page_rect.width if page_rect.width else 0.0
    frac_h = rect.height / page_rect.height if page_rect.height else 0.0
    return frac_w > BACKGROUND_FRACTION and frac_h > BACKGROUND_FRACTION


def _discover_pdfs() -> Dict[str, Path]:
    """
    Discover source_id → pdf_path under data/raw/**.

    Reuses chunk_all_sources.FILENAME_TO_SOURCE_ID so IDs match text chunks.
    PDFs absent from that map get FILE:<stem> instead of being skipped.
    """
    try:
        from chunk_all_sources import FILENAME_TO_SOURCE_ID  # sibling module
    except Exception:
        FILENAME_TO_SOURCE_ID = {}

    found: Dict[str, Path] = {}
    for folder in RAW_DIR.iterdir() if RAW_DIR.exists() else []:
        if not folder.is_dir() or folder.name in SKIP_FOLDERS:
            continue
        for pdf_path in folder.rglob("*.pdf"):
            source_id = FILENAME_TO_SOURCE_ID.get(pdf_path.name.lower(), None)
            if source_id is None:
                source_id = f"FILE:{pdf_path.stem}"
            found[source_id] = pdf_path
    return found


def _collect_candidates(
    page: "fitz.Page",
    page_rect: "fitz.Rect",
    *,
    allow_off_canvas: bool,
) -> List[Tuple[int, "fitz.Rect", float]]:
    """Return (xref, rect, area) candidates for one page pass."""
    keep: List[Tuple[int, "fitz.Rect", float]] = []
    seen_xrefs = set()
    for img in page.get_images():
        xref = img[0]
        for rect in page.get_image_rects(xref):
            if not allow_off_canvas and not _visible_rect(rect, page_rect):
                continue
            # Watermark test uses placement size relative to page — works for
            # both on-canvas logos and their off-canvas spread duplicates.
            if _is_watermark_placement(rect, page_rect):
                continue
            area = rect.width * rect.height
            if area < MIN_PLACEMENT_AREA:
                continue
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            keep.append((xref, rect, area))
    return keep


def extract_from_pdf(source_id: str, pdf_path: Path) -> Dict[str, List[str]]:
    """
    Extract demo photos from one PDF.

    Returns: {page_number(str): [image_url, ...]} with 1-indexed page keys.
    """
    out: Dict[str, List[str]] = {}
    doc = fitz.open(str(pdf_path))
    out_dir = STATIC_DIR / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for pno in range(len(doc)):
        page = doc[pno]
        page_rect = page.rect
        page_number = pno + 1  # matches chunk_all_sources 1-indexed page_number

        keep = _collect_candidates(page, page_rect, allow_off_canvas=False)
        if not keep:
            # Spread-layout fallback: only demo xref is off-canvas.
            keep = _collect_candidates(page, page_rect, allow_off_canvas=True)

        if not keep:
            continue

        keep.sort(key=lambda t: t[2], reverse=True)  # largest first

        urls = []
        for idx, (xref, rect, area) in enumerate(keep):
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK etc -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                fname = f"p{page_number:03d}_{idx}.png"
                fpath = out_dir / fname
                pix.save(str(fpath))
                urls.append(f"/static/guideline_images/{source_id}/{fname}")
            except Exception as e:
                print(f"    [WARN] failed to extract xref={xref} on page {page_number}: {e}")

        if urls:
            out[str(page_number)] = urls

    doc.close()
    return out


def attach_image_urls_to_chunks(
    images_map: Dict[str, Dict[str, List[str]]],
    chunk_file: Path = CHUNK_FILE,
) -> int:
    """
    Set image_urls on all_chunks.json rows that share (source_id, page_number).

    Returns how many chunk rows changed. Safe after extract without re-chunking.
    Chat UI prefers CLIP retrieval; image_urls is same-page metadata only.
    """
    if not chunk_file.exists() or not images_map:
        return 0
    try:
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] could not load {chunk_file} to attach image_urls: {e}")
        return 0

    updated = 0
    for c in chunks:
        sid = c.get("source_id")
        page = c.get("page_number")
        urls: List[str] = []
        if sid and page is not None:
            urls = list(images_map.get(sid, {}).get(str(int(page)), []) or [])
        prev = c.get("image_urls") or []
        if urls != prev:
            updated += 1
        c["image_urls"] = urls

    chunk_file.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return updated


def run() -> Dict[str, Dict[str, List[str]]]:
    """Scan all PDFs, rewrite pdf_images_map.json, stamp chunk image_urls."""
    full_map: Dict[str, Dict[str, List[str]]] = {}
    if MAP_FILE.exists():
        try:
            full_map = json.loads(MAP_FILE.read_text(encoding="utf-8"))
        except Exception:
            full_map = {}

    pdfs = _discover_pdfs()
    if not pdfs:
        print(f"[WARN] No PDFs found under {RAW_DIR}")

    for source_id, pdf_path in sorted(pdfs.items()):
        print(f"[{source_id}] scanning {pdf_path.name} for demo photos ...")
        page_map = extract_from_pdf(source_id, pdf_path)
        n_pages = len(page_map)
        n_imgs = sum(len(v) for v in page_map.values())
        if n_imgs == 0:
            print("    -> no qualifying photos (prose-only doc) — skipping")
            full_map.pop(source_id, None)
            continue
        print(f"    -> {n_imgs} image(s) across {n_pages} page(s)")
        full_map[source_id] = page_map

    MAP_FILE.write_text(json.dumps(full_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved image map -> {MAP_FILE}")

    n_attached = attach_image_urls_to_chunks(full_map)
    if n_attached:
        print(f"Attached image_urls on {n_attached} chunk row(s) in {CHUNK_FILE}")
    elif CHUNK_FILE.exists():
        print(f"image_urls stamp complete (no row diffs) -> {CHUNK_FILE}")
    else:
        print(f"[WARN] {CHUNK_FILE} missing — run chunk_all_sources.py then re-extract")

    return full_map


if __name__ == "__main__":
    run()
