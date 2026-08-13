"""
FILE LOCATION: backend/scripts/load_db.py

Load the merged exercise corpus + derived taxonomy into the database.
Run in order:
    python scripts/fetch_external_sources.py    
    python scripts/build_exercise_corpus.py
    python scripts/load_db.py
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `app.*` imports

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.models import Base, Exercise, Taxonomy

BACKEND_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BACKEND_DIR / "data" / "fitness.db"
PROCESSED_DIR = BACKEND_DIR / "data" / "adaptive-fitness-planner-data" / "processed"

engine = create_engine(f"sqlite:///{DB_PATH}")
Base.metadata.drop_all(engine)   # clean rebuild each run during dev
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# ---------- Load merged exercises ----------

exercises_path = PROCESSED_DIR / "exercises_merged.json"
if not exercises_path.exists():
    raise FileNotFoundError(
        f"{exercises_path} not found. Run scripts/build_exercise_corpus.py first "
        f"(and scripts/fetch_external_sources.py before that)."
    )

with open(exercises_path, encoding="utf-8") as f:
    merged = json.load(f)

n_exercises = 0
for row in merged:
    ex = Exercise(
        name=row["name"],
        description=row.get("description") or None,
        category=row.get("category") or None,
        body_part=row.get("body_part") or None,
        target_muscle=row.get("target_muscle") or None,
        equipment=row.get("equipment") or None,
        difficulty_level=row.get("difficulty_level") or None,
        rating=row.get("rating"),
        gif_url=row.get("gif_url") or None,
        image_url=row.get("image_url") or None,
        video_url=row.get("video_url") or None,
        has_media=bool(row.get("has_media")),
        match_confidence=1.0,
        source=row.get("source"),
    )
    session.add(ex)
    n_exercises += 1

session.commit()

# ---------- Load derived taxonomy (NOT separate files — see docstring) ----------

taxonomy_path = PROCESSED_DIR / "taxonomy.json"
if not taxonomy_path.exists():
    raise FileNotFoundError(f"{taxonomy_path} not found. Run scripts/build_exercise_corpus.py first.")

with open(taxonomy_path, encoding="utf-8") as f:
    taxonomy = json.load(f)

n_taxonomy = 0
for kind, names in taxonomy.items():  # kind in {"body_part", "equipment", "muscle"}
    for name in names:
        session.add(Taxonomy(kind=kind, name=name))
        n_taxonomy += 1

session.commit()

# ---------- Verify ----------

total_ex = session.query(Exercise).count()
with_media = session.query(Exercise).filter(Exercise.has_media == True).count()
total_tax = session.query(Taxonomy).count()

print(f"Loaded exercises: {n_exercises}")
print(f"TOTAL exercises in DB: {total_ex}  (with media: {with_media})")
print(f"Taxonomy loaded: {n_taxonomy} entries across {len(taxonomy)} kinds (total={total_tax})")
with_gif = session.query(Exercise).filter(Exercise.gif_url.isnot(None), Exercise.gif_url != "").count()
print(f"Exercises with gif_url: {with_gif}")
if with_media == 0:
    print("\nNOTE: 0 exercises have media. Clone hasaneyldrm/exercises-dataset and "
          "link images/ + videos/ under backend/static/exercises/, then rebuild.")
print(f"DB file: {DB_PATH}")
