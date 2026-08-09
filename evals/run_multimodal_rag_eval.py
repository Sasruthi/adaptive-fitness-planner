"""
Multimodal RAG Eval — CLIP image retrieval
=============================================
Small precision@k check for backend/app/services/rag_retrieval.py's
retrieve_guideline_images(). Run this AFTER:
  1. python backend/rag/extract_pdf_images.py
  2. python backend/rag/embed_images_clip.py

Usage:
    python evals/run_multimodal_rag_eval.py

For each test case, checks whether the expected (source_id, page_number)
shows up in the top-k CLIP image results, and prints the score so you can
tune IMAGE_SCORE_THRESHOLD in rag_retrieval.py.

Note: stop the API server first if it holds backend/rag/qdrant_local/.lock
(local Qdrant is single-writer).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.rag_retrieval import retrieve_guideline_images  # noqa: E402

# Ground truth for common_yoga_protocol.pdf (SRC009) — 1-indexed page numbers
# verified against PDF text + pdf_images_map.json after extract.
TEST_CASES = [
    {
        # p.36/p.37 share the same alternate-nostril demo photo (spread layout).
        "query": "how to do pranayama alternate nostril breathing",
        "source_id": "SRC009",
        "page_number": (36, 37),
    },
    {
        "query": "nadi shodhana anuloma viloma technique",
        "source_id": "SRC009",
        "page_number": (36, 37),
    },
    {
        "query": "tadasana mountain pose standing",
        "source_id": "SRC009",
        "page_number": 21,
    },
    {
        "query": "vrksasana tree pose how to balance",
        "source_id": "SRC009",
        "page_number": 22,
    },
    {
        # p.38 seated mudra; p.39 starts DHYĀNA; p.40 meditation close-up.
        "query": "how to sit in meditation dhyana sambhavi",
        "source_id": "SRC009",
        "page_number": (38, 39, 40),
    },
]


def _expected_pages(page_number) -> set:
    if isinstance(page_number, (list, tuple, set)):
        return {int(p) for p in page_number}
    return {int(page_number)}


def run(top_k: int = 3):
    hits, total = 0, 0
    for case in TEST_CASES:
        total += 1
        expect_pages = _expected_pages(case["page_number"])
        # score_threshold=0.0 so we can see ranking; pass/fail uses page match.
        results = retrieve_guideline_images(
            case["query"],
            top_k=top_k,
            source_ids=[case["source_id"]],
            score_threshold=0.0,
        )
        matched_page = any(
            r["source_id"] == case["source_id"]
            and r["page_number"] in expect_pages
            for r in results
        )
        hits += int(matched_page)
        pages_lbl = "/".join(str(p) for p in sorted(expect_pages))
        print(f"\nQuery: {case['query']!r}  (expect {case['source_id']} p.{pages_lbl})")
        if not results:
            print("  (no results — did you run embed_images_clip.py? is qdrant unlocked?)")
        for r in results:
            flag = (
                "✓"
                if r["source_id"] == case["source_id"]
                and r["page_number"] in expect_pages
                else " "
            )
            print(
                f"  [{flag}] score={r['score']:.4f} "
                f"{r['source_id']} p.{r['page_number']} {r['image_url']}"
            )

    print(f"\n{'='*50}\nPrecision@{top_k}: {hits}/{total} ({hits/total*100:.0f}%)")
    print(
        "Tune IMAGE_SCORE_THRESHOLD in rag_retrieval.py based on the score "
        "gap between correct (✓) and incorrect hits above."
    )


if __name__ == "__main__":
    run()
