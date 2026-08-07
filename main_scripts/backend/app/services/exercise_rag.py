"""
FILE LOCATION: backend/app/services/exercise_rag.py

Thin semantic layer over exercise descriptions.
=================================================
Deliberately SEPARATE from `fitness_guidelines` (the RAG collection for
PDFs). Plan generation still uses SQL (`exercise_retrieval.py`) for exact
filtering by body_part/equipment/difficulty — that's correct and unchanged.

This collection exists for conversational Q&A like "what's a good
exercise for lower back pain", where we:
  1. Semantically match the question to exercise instructions/name
  2. Soft-filter by known profile constraints (equipment, body parts)
  3. Prefer hits that have gif_url / image_url so the UI can show media

Uses the shared Qdrant client from rag_retrieval — local-disk Qdrant only
allows one open client per path.
"""

from typing import Dict, List, Optional

from app.services.rag_retrieval import get_qdrant

EXERCISE_COLLECTION = "exercise_semantic"


def _get_embed_model():
    from app.services.embedder import get_shared_embed_model
    return get_shared_embed_model()


def _norm(s: str) -> str:
    return (s or "").lower().strip()


_BODYWEIGHT_EQ = {"none", "body only", "body weight", "bodyweight", "body-only", ""}

# Catalog often tags pull-ups / hanging leg raises as "body only" because you
# only use your bodyweight — but the user who said "no equipment" means
# floor work at home, not a pull-up bar / rings / dip station.
_APPARATUS_PHRASES = (
    "pull-up bar", "pull up bar", "pullup bar",
    "chin-up bar", "chin up bar", "chinup bar",
    "hanging from", "hang from a", "hang from the",
    "dip bar", "dip station", "dip-pull-up", "dip pull-up",
    "gymnastic ring", " rings", "on the rings",
    "captain's chair", "captains chair", "roman chair",
    "suspension trainer", "trx ",
    "stall bar", "wall bar", "wall bars", "swedish ladder",
    "lat pulldown", "cable",
)

_APPARATUS_NAME_TOKENS = (
    "pull-up", "pull up", "pullup", "pullups",
    "chin-up", "chin up", "chinup", "chinups",
    "muscle up", "muscle-up",
    "front lever", "back lever",
    "ring dip", "rings",
    "hanging",
    "planche",
)

_ADVANCED_SKILL_TOKENS = (
    "planche", "muscle up", "muscle-up", "front lever", "back lever",
    "human flag", "one arm chin", "one-arm chin", "one arm pull",
    "handstand push", "pistol squat", "pistol", "dragon flag", "skin the cat",
    "kipping", "clapping push", "explosive",
)

_BODY_PART_SYNONYMS = {
    "neck": ["neck"],
    "shoulders": ["shoulder", "deltoid"],
    "chest": ["chest", "pectoral", "pec"],
    "back": ["back", "lat", "trapezius", "rhomboid"],
    "upper arms": ["upper arm", "bicep", "tricep", "arm"],
    "lower arms": ["forearm", "wrist", "lower arm"],
    "waist": ["waist", "abs", "core", "oblique", "abdominal", "midsection"],
    "upper legs": ["upper leg", "quad", "hamstring", "thigh", "glute", "hip"],
    "lower legs": ["lower leg", "calf", "calves", "shin"],
    "cardio": ["cardio", "conditioning", "endurance"],
}


def _requires_apparatus(name: str, description: str = "") -> bool:
    """True if the move needs a bar/rings/station — not pure floor bodyweight."""
    hay = f"{_norm(name)} {_norm(description)}"
    if any(p in hay for p in _APPARATUS_PHRASES):
        return True
    name_n = _norm(name)
    return any(t in name_n for t in _APPARATUS_NAME_TOKENS)


def _is_advanced_skill(name: str, description: str = "") -> bool:
    hay = f"{_norm(name)} {_norm(description)}"
    return any(t in hay for t in _ADVANCED_SKILL_TOKENS)


def _difficulty_allowed(diff: str, prefer_difficulty: Optional[str]) -> bool:
    """Hard gate: beginners never get Expert (or known advanced skills)."""
    want = _norm(prefer_difficulty or "")
    if not want:
        return True
    diff_n = _norm(diff)
    if want == "beginner":
        if "expert" in diff_n or "advanced" in diff_n:
            return False
        return True
    if want == "intermediate":
        if "expert" in diff_n:
            return False
    return True


def _is_bodyweight_only_filter(allowed: Optional[List[str]]) -> bool:
    if not allowed:
        return False
    allowed_n = {_norm(a) for a in allowed}
    return bool(allowed_n) and allowed_n <= _BODYWEIGHT_EQ


def _body_part_ok(ex_bp: str, ex_target: str, wanted: Optional[List[str]]) -> bool:
    """Match catalog region to wanted parts via synonym sets (controlled vocab)."""
    if not wanted:
        return True
    hay = f"{_norm(ex_bp)} {_norm(ex_target)}"
    for w in wanted:
        wn = _norm(w)
        syns = _BODY_PART_SYNONYMS.get(wn, [wn])
        if any(s in hay for s in syns if s):
            return True
    return False


def _equipment_ok(ex_eq: str, allowed: Optional[List[str]], *, name: str = "", description: str = "") -> bool:
    """Hard filter. Bodyweight-only = floor/no-gear, not 'bodyweight on a bar'."""
    if allowed is None:
        return True
    if len(allowed) == 0:
        allowed = ["body only", "none"]
    eq = _norm(ex_eq)
    if _is_bodyweight_only_filter(allowed):
        if eq not in _BODYWEIGHT_EQ:
            return False
        if _requires_apparatus(name, description):
            return False
        return True
    for a in allowed:
        an = _norm(a)
        if not an:
            continue
        if an in _BODYWEIGHT_EQ:
            if eq in _BODYWEIGHT_EQ and not _requires_apparatus(name, description):
                return True
            continue
        if an in eq or eq in an:
            return True
    return False


def retrieve_exercise_semantic(
    query: str,
    top_k: int = 5,
    *,
    equipment: Optional[List[str]] = None,
    body_parts: Optional[List[str]] = None,
    prefer_media: bool = True,
    prefer_difficulty: Optional[str] = None,
    relax_filters_on_empty: bool = False,
) -> List[Dict]:
    """
    Semantic search over exercise name+description+muscle, then apply
    profile constraints (equipment / body parts). Prefers rows with media
    so the frontend can render GIFs/images for the chosen exercises.

    Equipment is a HARD filter by default. Difficulty is a SOFT preference:
    `prefer_difficulty="beginner"` re-ranks matching / blank difficulty
    above expert moves — it does not hard-exclude (many catalog rows have
    null difficulty).
    """
    try:
        client = get_qdrant()
        if EXERCISE_COLLECTION not in [c.name for c in client.get_collections().collections]:
            return []
        from app.services.embedder import encode_one
        vec = encode_one(query)
        fetch_n = max(top_k * 12, 60)
        results = client.query_points(
            collection_name=EXERCISE_COLLECTION,
            query=vec,
            limit=fetch_n,
            with_payload=True,
        )

        want_diff = _norm(prefer_difficulty) if prefer_difficulty else ""

        hits = []
        for p in results.points:
            payload = p.payload or {}
            eq = payload.get("equipment", "")
            bp = payload.get("body_part", "")
            target = payload.get("target_muscle", "")
            diff = payload.get("difficulty", "") or ""
            name = payload.get("name", "") or ""
            description = payload.get("description", "") or ""
            if not _equipment_ok(eq, equipment, name=name, description=description):
                continue
            if not _body_part_ok(bp, target, body_parts):
                continue
            if want_diff == "beginner" and _is_advanced_skill(name, description):
                continue
            if not _difficulty_allowed(diff, prefer_difficulty):
                continue
            gif = payload.get("gif_url", "") or ""
            image = payload.get("image_url", "") or ""
            video = payload.get("video_url", "") or ""
            diff_n = _norm(diff)
            if not want_diff:
                diff_rank = 0
            elif want_diff in diff_n:
                diff_rank = 0
            elif not diff_n:
                diff_rank = 1
            elif want_diff == "beginner" and "intermediate" in diff_n:
                diff_rank = 3
            else:
                diff_rank = 2
            hits.append({
                "name": name,
                "body_part": bp,
                "target_muscle": target,
                "equipment": eq,
                "difficulty": diff,
                "description": description,
                "gif_url": gif,
                "image_url": image,
                "video_url": video,
                "has_media": bool(gif or image or video),
                "score": round(p.score, 4),
                "_diff_rank": diff_rank,
            })

        # Difficulty first, then media
        if prefer_media:
            hits.sort(key=lambda h: (h["_diff_rank"], not h["has_media"], -h["score"]))
        else:
            hits.sort(key=lambda h: (h["_diff_rank"], -h["score"]))

        if (
            not hits
            and relax_filters_on_empty
            and (equipment or body_parts)
            and not _is_bodyweight_only_filter(equipment)
        ):
            return retrieve_exercise_semantic(
                query,
                top_k=top_k,
                equipment=None,
                body_parts=None,
                prefer_media=prefer_media,
                prefer_difficulty=prefer_difficulty,
                relax_filters_on_empty=False,
            )

        out = hits[:top_k]
        for h in out:
            h.pop("_diff_rank", None)
        return out
    except Exception as e:
        print(f"[ExerciseRAG] retrieval failed, returning empty: {e}")
        return []
