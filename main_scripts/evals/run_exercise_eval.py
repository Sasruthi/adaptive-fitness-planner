#!/usr/bin/env python3
"""
Exercise semantic retrieval evaluation (+ diet-intent gate).

Retrieval metrics printed:
  Name Hit@k, Equipment compliance, Apparatus compliance,
  Instruction coverage, Media coverage, Mean score

Usage:
  cd backend && source ../env/bin/activate
  python ../evals/run_exercise_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from evals.metrics import mean  # noqa: E402
from evals.qdrant_lock import ensure_qdrant_for_eval  # noqa: E402

ensure_qdrant_for_eval()
from app.conversation.agent import _query_is_diet_focused  # noqa: E402
from app.services.exercise_rag import (  # noqa: E402
    _requires_apparatus,
    retrieve_exercise_semantic,
)

DATA = Path(__file__).resolve().parent / "datasets" / "exercise_rag.jsonl"
K = 4


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def name_hit(hits, must_any) -> float:
    if not must_any:
        return 1.0
    names = " ".join(h.get("name", "") for h in hits).lower()
    return 1.0 if any(m.lower() in names for m in must_any) else 0.0


def main():
    cases = load_jsonl(DATA)
    name_hits, equip_ok, apparatus_ok, instr_cov, media_cov, scores, diet_ok = (
        [], [], [], [], [], [], []
    )

    print("=" * 64)
    print(f"  EXERCISE RETRIEVAL EVAL  —  {len(cases)} cases, k={K}")
    print("=" * 64)

    for case in cases:
        q = case["query"]

        if case.get("expect_zero_exercises") or case.get("intent") == "diet":
            is_diet = _query_is_diet_focused(q)
            # Also assert product path would skip exercise retrieval
            hits = [] if is_diet else retrieve_exercise_semantic(
                q, top_k=K, equipment=case.get("equipment"), prefer_media=True,
                relax_filters_on_empty=False,
            )
            ok = is_diet and len(hits) == 0
            if not is_diet:
                ok = False
            diet_ok.append(1.0 if ok else 0.0)
            print(
                f"[{'PASS' if ok else 'FAIL'}] {case['id']} diet-intent={is_diet} "
                f"ex_hits={len(hits)} | {q[:50]}"
            )
            continue

        hits = retrieve_exercise_semantic(
            q,
            top_k=K,
            equipment=case.get("equipment"),
            prefer_media=True,
            prefer_difficulty=case.get("prefer_difficulty"),
            relax_filters_on_empty=False,
        )

        nh = name_hit(hits, case.get("must_name_any") or [])
        name_hits.append(nh)

        forbid_eq = [s.lower() for s in (case.get("forbid_equipment_substrings") or [])]
        viol_eq = 0
        viol_app = 0
        with_instr = 0
        with_media = 0
        for h in hits:
            eq = (h.get("equipment") or "").lower()
            if any(f in eq for f in forbid_eq):
                viol_eq += 1
            if case.get("forbid_apparatus") and _requires_apparatus(h.get("name", ""), h.get("description", "")):
                viol_app += 1
            if (h.get("description") or "").strip():
                with_instr += 1
            if h.get("gif_url") or h.get("image_url") or h.get("has_media"):
                with_media += 1

        # Empty retrieval = FAIL compliance (do not inflate metrics)
        eo = 1.0 if hits and viol_eq == 0 else 0.0
        ao = 1.0 if hits and viol_app == 0 else 0.0
        ic = (with_instr / len(hits)) if hits else 0.0
        mc = (with_media / len(hits)) if hits else 0.0
        ms = mean([float(h.get("score") or 0.0) for h in hits]) if hits else 0.0

        equip_ok.append(eo)
        apparatus_ok.append(ao)
        instr_cov.append(ic)
        media_cov.append(mc)
        scores.append(ms)

        status = "PASS" if hits and nh and eo and ao else "FAIL"
        print(
            f"[{status}] {case['id']}: n={len(hits)} name_hit={nh:.0f} "
            f"equip={eo:.0f} apparatus={ao:.0f} instr={ic:.2f} media={mc:.2f}"
        )
        for h in hits[:3]:
            media = "gif/img" if (h.get("gif_url") or h.get("image_url")) else "no-media"
            print(f"         - {h.get('name')} | {h.get('equipment')} | {media}")

    print("\n--- Exercise retrieval metrics ---")
    if name_hits:
        print(f"  Name Hit@{K}:              {mean(name_hits):.3f}  ({sum(name_hits):.0f}/{len(name_hits)})")
        print(f"  Equipment compliance:    {mean(equip_ok):.3f}")
        print(f"  No-apparatus compliance: {mean(apparatus_ok):.3f}")
        print(f"  Instruction coverage:    {mean(instr_cov):.3f}")
        print(f"  Media coverage:          {mean(media_cov):.3f}")
        print(f"  Mean retrieval score:    {mean(scores):.3f}")
    if diet_ok:
        print(f"  Diet-intent accuracy:    {mean(diet_ok):.3f}  ({sum(diet_ok):.0f}/{len(diet_ok)})")
    print()
    fail = False
    if name_hits and mean(name_hits) < 0.5:
        fail = True
    if diet_ok and mean(diet_ok) < 0.5:
        fail = True
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
