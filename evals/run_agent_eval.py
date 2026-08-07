#!/usr/bin/env python3
"""
Evaluate profile auto-ingest + intent heuristics (semantic NLU).

Does NOT call the LLM by default — measures embedding-based gates that
protect slot accuracy (age/gender/diabetes, diet vs exercise intent).

Usage:
  cd backend && source ../env/bin/activate
  python ../evals/run_agent_eval.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from evals.metrics import mean  # noqa: E402
from app.conversation.agent import (  # noqa: E402
    _auto_ingest_profile_hints,
    _query_is_diet_focused,
)
from app.conversation.profile_store import session_store  # noqa: E402

DATA = Path(__file__).resolve().parent / "datasets" / "agent_slots.jsonl"


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    cases = load_jsonl(DATA)
    scores = []

    print("=" * 64)
    print(f"  AGENT SLOT / INTENT EVAL (semantic)  —  {len(cases)} cases")
    print("=" * 64)

    for case in cases:
        tid = str(uuid.uuid4())
        msg = case["message"]
        ok = True
        notes = []

        if "expect_diet_intent" in case:
            got = _query_is_diet_focused(msg)
            if got != case["expect_diet_intent"]:
                ok = False
                notes.append(f"diet_intent got={got}")
        else:
            # Only production auto-ingest — no harness injection of body parts
            _auto_ingest_profile_hints(msg, tid)
            p = session_store.get_profile(tid)

            if case.get("expect_age") is not None and p.age != case["expect_age"]:
                ok = False
                notes.append(f"age got={p.age}")
            if case.get("expect_gender") and p.gender != case["expect_gender"]:
                ok = False
                notes.append(f"gender got={p.gender}")
            if case.get("expect_health_flags_any"):
                if not any(f in p.health_flags for f in case["expect_health_flags_any"]):
                    ok = False
                    notes.append(f"flags got={p.health_flags}")
            if case.get("expect_equipment_any"):
                if not any(e in p.available_equipment for e in case["expect_equipment_any"]):
                    ok = False
                    notes.append(f"equip got={p.available_equipment}")
            if case.get("expect_body_parts_count_min"):
                if len(p.target_body_parts) < case["expect_body_parts_count_min"]:
                    ok = False
                    notes.append(f"parts={len(p.target_body_parts)}")
            if case.get("expect_diet_only"):
                if p.plan_mode != "diet_only":
                    ok = False
                    notes.append(f"plan_mode got={p.plan_mode}")
            if case.get("expect_body_parts_any"):
                if not any(b in p.target_body_parts for b in case["expect_body_parts_any"]):
                    ok = False
                    notes.append(f"parts got={p.target_body_parts}")

        scores.append(1.0 if ok else 0.0)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['id']}: {msg[:55]}" + (f" | {notes}" if notes else ""))

    print("\n--- Agent metrics ---")
    print(f"  Slot/intent accuracy: {mean(scores):.3f}  ({sum(scores):.0f}/{len(scores)})")
    print()
    # Require stronger than coin-flip for green
    return 0 if mean(scores) >= 0.75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
