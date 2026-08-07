"""
FILE LOCATION: backend/scripts/fetch_external_sources.py

Downloads the two external exercise datasets that add real incremental
value (see build_exercise_corpus.py docstring for why wrkout is excluded)
into data/adaptive-fitness-planner-data/raw/external/.

Run once during setup, or on a schedule in CI if you want to pick up
upstream additions later — NOT at request time. requires `requests`.

    python scripts/fetch_external_sources.py
"""
from pathlib import Path
import requests

RAW_EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "adaptive-fitness-planner-data" / "raw" / "external"
RAW_EXTERNAL.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "exercemus_exercises.json":
        "https://raw.githubusercontent.com/exercemus/exercises/main/exercises.json",
    "longhaul_cardio.json":
        "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/cardio.json",
    "longhaul_flexibility.json":
        "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/flexibility.json",
    "longhaul_strength.json":
        "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/strength.json",
}


def run():
    for filename, url in SOURCES.items():
        dest = RAW_EXTERNAL / filename
        print(f"Fetching {url} -> {dest}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  {len(resp.content):,} bytes saved")
    print(f"\nDone. Files saved to {RAW_EXTERNAL}")
    print("NOTE: wrkout/exercises.json deliberately NOT fetched — it is the "
          "verbatim upstream source that free-exercise-db (873 records, already "
          "in raw/structured/) was restructured from. Diffed directly: 873/873 "
          "wrkout exercises match free-exercise-db exercises by name, byte-for-byte "
          "identical instructions. Zero incremental value, so it's skipped rather "
          "than doubling ingestion time for nothing.")


if __name__ == "__main__":
    run()
