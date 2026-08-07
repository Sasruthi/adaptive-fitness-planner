"""
Adaptive Fitness Planner — GitHub Exercise Repo Puller
=======================================================
Pulls structured exercise data from the 4 GitHub repos in the manifest
(SRC011-SRC015) and loads them into the existing SQLite exercise catalog.

free-exercise-db (SRC011) is already downloaded as a JSON — loaded directly.
The 3 repo-URL-only sources are pulled via GitHub raw content API.

Run: python pull_github_exercises.py
"""

import sys, json, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.models import Base, Exercise

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "fitness.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
Session = sessionmaker(bind=engine)
session = Session()

HEADERS = {"User-Agent": "AdaptiveFitnessPlanner/1.0 (portfolio project)"}

# ── SRC011: free-exercise-db (already downloaded as JSON, public domain) ──────

def load_free_exercise_db():
    path = Path("/home/claude/rag_sources/structured/free_exercise_db_exercises.json")
    with open(path) as f:
        data = json.load(f)

    count = 0
    for ex in data:
        # Check for duplicate by name
        existing = session.query(Exercise).filter(
            Exercise.name.ilike(ex.get("name", "").strip())
        ).first()
        if existing:
            continue

        session.add(Exercise(
            name=ex.get("name", ""),
            description=" ".join(ex.get("instructions", [])) or None,
            category=ex.get("category", "").title() or None,
            body_part=", ".join(ex.get("primaryMuscles", [])) or None,
            target_muscle=", ".join(ex.get("primaryMuscles", [])) or None,
            equipment=ex.get("equipment", "") or None,
            difficulty_level=ex.get("level", "").title() or None,
            rating=None,
            gif_url=None,   # free-exercise-db images need GitHub raw URL prefix to serve
            has_media=False,
            match_confidence=1.0,
            source="free_exercise_db_public_domain",
        ))
        count += 1

    session.commit()
    return count

# ── SRC012-015: Try to pull raw JSON from GitHub repos ───────────────────────

GITHUB_SOURCES = [
    {
        "source_id": "SRC012",
        "name": "hasaneyldrm/exercises-dataset",
        "raw_url": "https://raw.githubusercontent.com/hasaneyldrm/exercises-dataset/main/exercises.json",
        "parser": "hasaneyldrm",
    },
    {
        "source_id": "SRC013",
        "name": "wrkout/exercises.json",
        "raw_url": "https://raw.githubusercontent.com/wrkout/exercises.json/master/exercises.json",
        "parser": "wrkout",
    },
    {
        "source_id": "SRC014",
        "name": "exercemus/exercises",
        "raw_url": "https://raw.githubusercontent.com/exercemus/exercises/main/exercises.json",
        "parser": "exercemus",
    },
    {
        "source_id": "SRC015",
        "name": "longhaul-fitness/exercises",
        "raw_url": "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/exercises.json",
        "parser": "generic",
    },
]

def parse_and_load(data, source_name, parser_type) -> int:
    """
    Different repos use different schemas — normalize to our Exercise model.
    Returns count of new records added.
    """
    if not isinstance(data, list):
        data = data.get("exercises", data.get("data", []))

    count = 0
    for ex in data:
        if not isinstance(ex, dict):
            continue
        name = (ex.get("name") or ex.get("title") or "").strip()
        if not name:
            continue

        # Dedup by name
        existing = session.query(Exercise).filter(
            Exercise.name.ilike(name)
        ).first()
        if existing:
            continue

        # Normalize fields across different schemas
        description = (
            ex.get("description") or
            ex.get("overview") or
            " ".join(ex.get("instructions", [])) or
            None
        )
        body_part = (
            ex.get("body_part") or
            ex.get("bodyPart") or
            ", ".join(ex.get("primaryMuscles", [])) or
            ex.get("muscle_group") or
            None
        )
        equipment = (
            ex.get("equipment") or
            ex.get("equipments") or
            None
        )
        if isinstance(equipment, list):
            equipment = ", ".join(equipment)

        difficulty = (
            ex.get("level") or
            ex.get("difficulty") or
            ex.get("difficulty_level") or
            None
        )
        gif_url = (
            ex.get("gifUrl") or
            ex.get("gif_url") or
            ex.get("video_url") or
            None
        )

        session.add(Exercise(
            name=name,
            description=description,
            category=ex.get("category", "").title() if ex.get("category") else None,
            body_part=body_part,
            target_muscle=body_part,
            equipment=str(equipment) if equipment else None,
            difficulty_level=str(difficulty).title() if difficulty else None,
            rating=None,
            gif_url=gif_url,
            has_media=bool(gif_url),
            match_confidence=1.0,
            source=source_name,
        ))
        count += 1

    session.commit()
    return count


def pull_github_source(source: dict) -> int:
    print(f"\n  Pulling {source['name']} ...")
    try:
        resp = requests.get(source["raw_url"], headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            count = parse_and_load(data, source["name"], source["parser"])
            print(f"  Added {count} new exercises from {source['name']}")
            return count
        else:
            print(f"  [skip] HTTP {resp.status_code} — repo may be private or URL changed")
            return 0
    except Exception as e:
        print(f"  [error] {e.__class__.__name__}: {e}")
        return 0


# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 60)
print("GitHub Exercise Repo Puller")
print("=" * 60)

before = session.query(Exercise).count()
print(f"\nExisting exercises in DB: {before}")

# Load free-exercise-db (local JSON, public domain)
print("\n[SRC011] free-exercise-db (local, public domain)")
n1 = load_free_exercise_db()
print(f"  Added {n1} new exercises")

# Pull from GitHub repos
total_github = 0
for source in GITHUB_SOURCES:
    total_github += pull_github_source(source)

after = session.query(Exercise).count()
with_media = session.query(Exercise).filter(Exercise.has_media == True).count()

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"  Before : {before} exercises")
print(f"  Added  : {after - before} new exercises")
print(f"  Total  : {after} exercises in DB")
print(f"  With media (gif/video): {with_media}")
