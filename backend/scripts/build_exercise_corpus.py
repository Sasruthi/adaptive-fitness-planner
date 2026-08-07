"""
FILE LOCATION: backend/scripts/build_exercise_corpus.py

Merges every exercise data source we evaluated into one deduplicated
corpus, ready for scripts/load_db.py to load into SQLite.

SOURCES USED (and why):
  - free_exercise_db_exercises.json (873 records, yuhonas/free-exercise-db)
    Your upload. Base strength/stretching/plyometrics coverage.
  - exercises.json (1,318 unique records, hasaneyldrm/exercises-dataset)
    Your upload. Gives us the exact 10-value body_part taxonomy
    (upper arms/upper legs/back/waist/chest/shoulders/lower legs/
    lower arms/cardio/neck) PLUS multi-language instructions (including
    Hindi) and gif media — used as the preferred record whenever a name
    collides with another source.
  - exercemus/exercises.json's top-level `exercises` list (872 records)
    98% identical to free-exercise-db (852/872 name overlap, confirmed
    by direct diff) — but contributes real value on the *shared* records
    via extra metadata (video URLs, tempo, tips) and 20 genuinely new
    exercises.
  - exercemus/exercises.json's `exercises_to_merge` list (190 records)
    Real wger.de-sourced exercises exercemus hasn't reviewed into their
    main list yet, but the underlying data is legitimate — worth taking.
  - longhaul-fitness cardio.json (9) + flexibility.json (46)
    Small, but plugs a real gap: your other sources have almost no
    dedicated cardio/flexibility content, and your Profile.goal field
    supports improve_flexibility/improve_endurance as goals with little
    inventory to serve them.

SOURCE DELIBERATELY EXCLUDED:
  - wrkout/exercises.json — diffed directly against free-exercise-db:
    873/873 names match, byte-identical instructions. This is the
    verbatim upstream free-exercise-db was restructured from. Zero
    incremental value; including it would only slow down ingestion.
  - longhaul-fitness strength.json (349) — EXCLUDED from this pass
    despite being available, because it has no `equipment` field at
    all (only pk/name/slug/primaryMuscles/secondaryMuscles/steps), and
    349 strength exercises with no equipment data would be unusable for
    your hard equipment filtering in exercise_retrieval.py without
    manual re-tagging. cardio.json/flexibility.json are kept because
    that gap matters more than the equipment gap for those categories.
    Flip INCLUDE_LONGHAUL_STRENGTH below if you'd rather have them with
    equipment="unknown" than not have them at all.

Run after fetch_external_sources.py:
    python scripts/build_exercise_corpus.py
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

RAW_STRUCTURED = Path(__file__).resolve().parents[1] / "data" / "adaptive-fitness-planner-data" / "raw" / "structured"
RAW_EXTERNAL   = Path(__file__).resolve().parents[1] / "data" / "adaptive-fitness-planner-data" / "raw" / "external"
PROCESSED_DIR  = Path(__file__).resolve().parents[1] / "data" / "adaptive-fitness-planner-data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INCLUDE_LONGHAUL_STRENGTH = False  # see docstring above

# The 10-value taxonomy hasaneyldrm already uses, matching Profile.BODY_PARTS
BODY_PARTS = ["neck", "shoulders", "chest", "back", "upper arms", "lower arms",
              "waist", "upper legs", "lower legs", "cardio"]

# Best-effort muscle -> body_part mapping for sources that only give muscles
# (free-exercise-db, exercemus, longhaul), calibrated against hasaneyldrm's
# own category distribution (upper arms=292, upper legs=227, back=203,
# waist=169, chest=163, shoulders=143, lower legs=59, lower arms=37, cardio=29,
# neck=2) so the derived split lands in the same ballpark.
MUSCLE_TO_BODYPART = {
    "biceps": "upper arms", "triceps": "upper arms", "brachialis": "upper arms",
    "forearms": "lower arms",
    "chest": "chest", "pectorals": "chest",
    "lats": "back", "middle back": "back", "lower back": "back", "traps": "back",
    "serratus anterior": "back", "upper back": "back",
    "shoulders": "shoulders", "delts": "shoulders", "rotator cuff": "shoulders",
    "abs": "waist", "obliques": "waist",
    "quads": "upper legs", "hamstrings": "upper legs", "glutes": "upper legs",
    "adductors": "upper legs", "abductors": "upper legs", "hip flexors": "upper legs",
    "calves": "lower legs", "soleus": "lower legs",
    "neck": "neck",
    "cardiovascular system": "cardio", "heart": "cardio",
}


def muscles_to_body_part(primary: List[str], secondary: List[str]) -> str:
    """Substring match so longhaul's singular forms ('bicep', 'lat',
    'glute', 'trap') and exercemus's plural forms both resolve."""
    for muscle in (primary or []) + (secondary or []):
        m = (muscle or "").lower().strip()
        for key, bp in MUSCLE_TO_BODYPART.items():
            if key in m or m in key:
                return bp
    return "waist"  # safest generic fallback — abs/core work is rarely wrong


def normalize_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return re.sub(r"\s+", " ", n)


def normalize_equipment(eq) -> str:
    if isinstance(eq, list):
        eq = ", ".join(eq) if eq else "none"
    eq = (eq or "none").lower().strip()
    if eq in ("body only", "body weight", "bodyweight", "none", ""):
        return "body only"
    return eq


# ══════════════════════════════════════════════════════════════════════════
# PARSERS — each returns a list of records in one canonical shape
# ══════════════════════════════════════════════════════════════════════════

def parse_free_exercise_db(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for x in data:
        out.append({
            "name": x["name"],
            "description": " ".join(x.get("instructions", []) or []),
            "category": x.get("category") or "strength",
            "body_part": None,  # derived from muscles below
            "primary_muscles": x.get("primaryMuscles", []),
            "secondary_muscles": x.get("secondaryMuscles", []),
            "equipment": normalize_equipment(x.get("equipment")),
            "difficulty_level": (x.get("level") or "").title() or None,
            "gif_url": None,
            "image_url": None,
            "video_url": None,
            "media_id": None,
            "rating": None,
            "source": "free_exercise_db",
        })
    return out


def parse_hasaneyldrm(path: Path) -> List[Dict]:
    """
    Parse hasaneyldrm/exercises-dataset records.

    Media paths in the JSON are repo-relative (e.g. videos/0001-xxx.gif,
    images/0001-xxx.jpg). We rewrite them to FastAPI static URLs so the
    frontend can load local GIFs/thumbnails from /static/exercises/...
    after the dataset is cloned/linked under backend/static/exercises/.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for x in data:
        instr = x.get("instructions", {})
        steps = x.get("instruction_steps", {})
        if isinstance(instr, dict):
            description = instr.get("en", "") or ""
        else:
            description = str(instr or "")
        if isinstance(steps, dict) and steps.get("en"):
            # Prefer step list for clearer RAG embedding text
            step_text = " ".join(steps["en"])
            if step_text:
                description = step_text

        gif_rel = (x.get("gif_url") or "").lstrip("/")
        img_rel = (x.get("image") or "").lstrip("/")
        # Normalize: videos/*.gif → /static/exercises/videos/...
        #            images/*.jpg → /static/exercises/images/...
        gif_url = f"/static/exercises/{gif_rel}" if gif_rel else None
        image_url = f"/static/exercises/{img_rel}" if img_rel else None

        target = x.get("target")
        secondary = list(x.get("secondary_muscles") or [])
        if x.get("muscle_group") and x["muscle_group"] not in secondary:
            secondary.append(x["muscle_group"])

        out.append({
            "name": x["name"],
            "description": description,
            "category": x.get("category") or x.get("body_part") or "strength",
            "body_part": x.get("body_part") or x.get("category"),  # trusted directly
            "primary_muscles": [target] if target else [],
            "secondary_muscles": secondary,
            "equipment": normalize_equipment(x.get("equipment")),
            "difficulty_level": None,
            "gif_url": gif_url,
            "image_url": image_url,
            "video_url": None,
            "media_id": x.get("media_id"),
            "rating": None,
            "source": "hasaneyldrm",
            "languages": instr if isinstance(instr, dict) else None,
            "attribution": x.get("attribution"),
        })
    return out


def parse_exercemus(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for x in data.get("exercises", []):
        out.append({
            "name": x["name"],
            "description": (x.get("description", "") + " " + " ".join(x.get("instructions", []) or [])).strip(),
            "category": x.get("category") or "strength",
            "body_part": None,
            "primary_muscles": x.get("primary_muscles", []),
            "secondary_muscles": x.get("secondary_muscles", []),
            "equipment": normalize_equipment(x.get("equipment")),
            "difficulty_level": None,
            "gif_url": None,
            "image_url": None,
            "video_url": x.get("video"),
            "media_id": None,
            "rating": None,
            "source": "exercemus",
        })
    # the 190 wger.de exercises exercemus hasn't merged into `exercises` yet
    for x in data.get("exercises_to_merge", []):
        out.append({
            "name": x["name"],
            "description": x.get("description", ""),
            "category": "strength",  # not provided; wger.de skews strength/rehab
            "body_part": None,
            "primary_muscles": x.get("primary_muscles", []),
            "secondary_muscles": x.get("secondary_muscles", []),
            "equipment": normalize_equipment(x.get("equipment")),
            "difficulty_level": None,
            "gif_url": None,
            "image_url": None,
            "video_url": None,
            "media_id": None,
            "rating": None,
            "source": "exercemus_unmerged_wger",
        })
    return out


def parse_longhaul(cardio_path: Path, flexibility_path: Path, strength_path: Optional[Path]) -> List[Dict]:
    out = []

    def _parse_file(path, forced_category):
        data = json.loads(path.read_text(encoding="utf-8"))
        for x in data:
            steps = x.get("steps") or x.get("instructions") or []
            out.append({
                "name": x["name"],
                "description": " ".join(steps),
                "category": forced_category,
                "body_part": None,
                "primary_muscles": x.get("primaryMuscles", []),
                "secondary_muscles": x.get("secondaryMuscles", []),
                "equipment": normalize_equipment(None),  # not provided by this source
                "difficulty_level": None,
                "gif_url": None,
                "image_url": None,
                "video_url": None,
                "media_id": None,
                "rating": None,
                "source": "longhaul_fitness",
            })

    _parse_file(cardio_path, "cardio")
    _parse_file(flexibility_path, "stretching")
    if strength_path and INCLUDE_LONGHAUL_STRENGTH:
        _parse_file(strength_path, "strength")
    return out


# ══════════════════════════════════════════════════════════════════════════
# MERGE + DEDUP
# ══════════════════════════════════════════════════════════════════════════

# Preference order when the same exercise name appears in multiple sources —
# hasaneyldrm wins (richest: media + multi-language + trusted body_part),
# then free_exercise_db (largest, well-established), then exercemus, then
# the unmerged wger.de batch, then longhaul.
SOURCE_PRIORITY = {
    "hasaneyldrm": 0, "free_exercise_db": 1, "exercemus": 2,
    "exercemus_unmerged_wger": 3, "longhaul_fitness": 4,
}


def merge_and_dedupe(all_records: List[Dict]) -> List[Dict]:
    by_name: Dict[str, Dict] = {}
    for rec in all_records:
        key = normalize_name(rec["name"])
        if not key:
            continue
        if key not in by_name or SOURCE_PRIORITY[rec["source"]] < SOURCE_PRIORITY[by_name[key]["source"]]:
            by_name[key] = rec
        else:
            # keep the winner's core fields, but backfill anything it's missing
            winner = by_name[key]
            if not winner.get("gif_url") and rec.get("gif_url"):
                winner["gif_url"] = rec["gif_url"]
            if not winner.get("image_url") and rec.get("image_url"):
                winner["image_url"] = rec["image_url"]
            if not winner.get("video_url") and rec.get("video_url"):
                winner["video_url"] = rec["video_url"]
            if not winner.get("description") and rec.get("description"):
                winner["description"] = rec["description"]
            # Prefer richer instruction text when winner has a thin description
            if rec.get("source") == "hasaneyldrm" and rec.get("description") and (
                not winner.get("description") or len(rec["description"]) > len(winner.get("description") or "")
            ) and SOURCE_PRIORITY[rec["source"]] <= SOURCE_PRIORITY[winner["source"]]:
                pass  # winner already hasaneyldrm or equal; keep winner description
    return list(by_name.values())


def finalize(records: List[Dict]) -> List[Dict]:
    for rec in records:
        if not rec.get("body_part"):
            rec["body_part"] = muscles_to_body_part(rec.get("primary_muscles", []), rec.get("secondary_muscles", []))
        rec["body_part"] = rec["body_part"].lower().strip()
        if rec["body_part"] not in BODY_PARTS:
            # e.g. hasaneyldrm's own category strings should already match,
            # but guard against anything unexpected slipping through
            rec["body_part"] = muscles_to_body_part(rec.get("primary_muscles", []), rec.get("secondary_muscles", []))
        rec["target_muscle"] = ", ".join((rec.get("primary_muscles") or [])[:2]) or None
        rec["has_media"] = bool(rec.get("gif_url") or rec.get("image_url") or rec.get("video_url"))
        if rec.get("secondary_muscles") and rec.get("description"):
            secs = ", ".join(rec["secondary_muscles"][:5])
            if secs and secs.lower() not in (rec["description"] or "").lower():
                rec["description"] = f"{rec['description']} Secondary muscles: {secs}."
    return records


def run():
    all_records: List[Dict] = []

    fe_path = RAW_STRUCTURED / "free_exercise_db_exercises.json"
    ha_path = RAW_STRUCTURED / "exercises.json"
    ex_path = RAW_EXTERNAL / "exercemus_exercises.json"
    lh_cardio = RAW_EXTERNAL / "longhaul_cardio.json"
    lh_flex = RAW_EXTERNAL / "longhaul_flexibility.json"
    lh_strength = RAW_EXTERNAL / "longhaul_strength.json"

    if fe_path.exists():
        recs = parse_free_exercise_db(fe_path)
        print(f"free_exercise_db: {len(recs)} records")
        all_records += recs
    else:
        print(f"[SKIP] {fe_path} not found")

    if ha_path.exists():
        recs = parse_hasaneyldrm(ha_path)
        print(f"hasaneyldrm: {len(recs)} records")
        all_records += recs
    else:
        print(f"[SKIP] {ha_path} not found")

    if ex_path.exists():
        recs = parse_exercemus(ex_path)
        print(f"exercemus (incl. unmerged wger.de): {len(recs)} records")
        all_records += recs
    else:
        print(f"[SKIP] {ex_path} not found — run fetch_external_sources.py first")

    if lh_cardio.exists() and lh_flex.exists():
        recs = parse_longhaul(lh_cardio, lh_flex, lh_strength if lh_strength.exists() else None)
        print(f"longhaul-fitness: {len(recs)} records")
        all_records += recs
    else:
        print(f"[SKIP] longhaul files not found — run fetch_external_sources.py first")

    print(f"\nTotal raw records before dedup: {len(all_records)}")
    merged = merge_and_dedupe(all_records)
    print(f"After name-dedup: {len(merged)}")
    finalized = finalize(merged)

    out_path = PROCESSED_DIR / "exercises_merged.json"
    out_path.write_text(json.dumps(finalized, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(finalized)} exercises to {out_path}")

    from collections import Counter
    print("\nBy body_part:", Counter(r["body_part"] for r in finalized).most_common())
    print("\nBy source (winning record):", Counter(r["source"] for r in finalized).most_common())
    print("\nWith media:", sum(1 for r in finalized if r["has_media"]))

    # Derived taxonomy — this replaces bodyParts.json/equipments.json/muscles.json
    equipment_tokens = set()
    for r in finalized:
        for token in r["equipment"].split(","):
            token = token.strip()
            if token:
                equipment_tokens.add(token)

    taxonomy = {
        "body_part": sorted(set(r["body_part"] for r in finalized)),
        "equipment": sorted(equipment_tokens),
        "muscle": sorted(set(
            m for r in finalized for m in (r.get("primary_muscles") or []) + (r.get("secondary_muscles") or []) if m
        )),
    }
    tax_path = PROCESSED_DIR / "taxonomy.json"
    tax_path.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    print(f"Wrote derived taxonomy to {tax_path}")
    print(f"  body_part ({len(taxonomy['body_part'])}): {taxonomy['body_part']}")
    print(f"  equipment ({len(taxonomy['equipment'])}): {taxonomy['equipment']}")
    print(f"  muscle ({len(taxonomy['muscle'])} total)")


if __name__ == "__main__":
    run()
