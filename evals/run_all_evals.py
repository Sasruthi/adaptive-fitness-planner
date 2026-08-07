#!/usr/bin/env python3
"""Run all eval suites and print a combined metric table to the console."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
EVALS = Path(__file__).resolve().parent

RUNNERS = [
    ("Guideline RAG (retrieval + generation)", "run_rag_eval.py"),
    ("Exercise retrieval", "run_exercise_eval.py"),
    ("Agent slots / intent", "run_agent_eval.py"),
]


def main() -> int:
    env = {**dict(**__import__("os").environ), "PYTHONPATH": f"{ROOT}:{BACKEND}"}
    # Ensure dotenv-loaded keys from backend cwd
    rc = 0
    print("\n" + "#" * 64)
    print("#  ADAPTIVE FITNESS PLANNER — FULL EVAL SUITE")
    print("#" * 64 + "\n")

    for title, script in RUNNERS:
        print(f"\n>>> {title}\n")
        proc = subprocess.run(
            [sys.executable, str(EVALS / script)],
            cwd=str(BACKEND),
            env=env,
        )
        if proc.returncode != 0:
            rc = proc.returncode
            print(f"[ERROR] {script} exited {proc.returncode}")

    print("\n" + "#" * 64)
    print("#  DONE")
    print("#" * 64 + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
