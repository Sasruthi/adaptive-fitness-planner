"""
Adaptive Fitness Planner — Exercise Catalog Merge Script
==========================================================
Merges exercise sources by SEMANTIC title similarity, not just exact-string match.

Sources merged:
  A. fitness_data.json  (megaGymDataset, 2909 rows) - rich metadata, NO media
  B. gifs_data.json      (fitnessprogramer.com, 1411 rows) - media + target muscle, NO description/equipment
  C. exercises.json      (ExerciseDB sample, 30 rows) - richest schema + local gif, used as gold-standard reference

Strategy:
  1. Normalize titles (lowercase, strip punctuation, remove filler words like
     "exercise", parenthetical equipment notes, plural/singular variance).
  2. Embed normalized titles with a sentence-transformer (semantic, catches
     "Barbell Squat" == "Back Squat (Barbell)"). Falls back to TF-IDF char
     n-gram cosine similarity if no model/network is available (still catches
     lexical near-duplicates, just not true paraphrase-level matches).
  3. For each fitness_data (A) row, find best candidate in gifs_data (B) by
     cosine similarity. Accept match only above a confidence threshold.
  4. Output:
       - merged_exercises.csv      -> matched rows (rich metadata + media)
       - unmatched_no_media.csv    -> rich metadata, no media found (flag for scraping)
       - unmatched_no_metadata.csv -> media exists, no rich metadata matched
       - match_audit.csv           -> every match with similarity score, for manual QA
"""

import json
import re
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "collected_data"
OUT_DIR = PROJECT_ROOT / "data" / "merged_output"
OUT_DIR.mkdir(exist_ok=True)

MATCH_THRESHOLD = 0.82  # cosine similarity cutoff for accepting a semantic match
REVIEW_THRESHOLD = 0.68  # below MATCH but above this -> "needs human review" bucket

# ---------- 1. Load ----------

def load_json(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)

fitness_data = load_json("fitness_data.json")     # rich metadata, no media (2909)
gifs_data = load_json("gifs_data.json")           # media + muscle, no metadata (1411)
exercisedb_sample = load_json("exercises.json")   # gold-standard schema (30)

print(f"Loaded: fitness_data={len(fitness_data)}, gifs_data={len(gifs_data)}, exercisedb_sample={len(exercisedb_sample)}")

# ---------- 2. Normalize titles ----------

FILLER_WORDS = {"exercise", "workout", "version", "variation"}

def normalize_title(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"\(.*?\)", " ", t)          # drop parenthetical notes e.g. "(barbell)"
    t = re.sub(r"[^a-z0-9\s]", " ", t)      # strip punctuation
    tokens = [w for w in t.split() if w not in FILLER_WORDS]
    # naive singularize (strip trailing 's' on words >3 chars, avoid 'press'->'pres')
    tokens = [w[:-1] if w.endswith("s") and len(w) > 4 and not w.endswith("ss") else w for w in tokens]
    return " ".join(sorted(tokens))  # sort tokens so word order doesn't matter for TF-IDF/exact pass

def normalize_title_ordered(title: str) -> str:
    """Same cleaning but keep word order — used for embedding model (order matters semantically)."""
    t = title.lower().strip()
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    tokens = [w for w in t.split() if w not in FILLER_WORDS]
    tokens = [w[:-1] if w.endswith("s") and len(w) > 4 and not w.endswith("ss") else w for w in tokens]
    return " ".join(tokens)

for row in fitness_data:
    row["_norm"] = normalize_title_ordered(row["title"])
for row in gifs_data:
    row["_norm"] = normalize_title_ordered(row["title"])

# ---------- 3. Embed (semantic) with fallback ----------

EMBED_METHOD = None
embed_fn = None

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    EMBED_METHOD = "sentence-transformers (all-MiniLM-L6-v2)"

    def embed_fn(texts):
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

except Exception as e:
    print(f"[fallback] sentence-transformers unavailable ({e.__class__.__name__}); "
          f"using TF-IDF char n-gram cosine similarity instead.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    EMBED_METHOD = "TF-IDF char(3-5) n-grams (fallback, no model download required)"

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    all_texts = [r["_norm"] for r in fitness_data] + [r["_norm"] for r in gifs_data]
    vectorizer.fit(all_texts)

    def embed_fn(texts):
        return vectorizer.transform(texts).toarray()

print(f"Embedding method in use: {EMBED_METHOD}")

A_texts = [r["_norm"] for r in fitness_data]
B_texts = [r["_norm"] for r in gifs_data]

A_emb = embed_fn(A_texts)
B_emb = embed_fn(B_texts)

# normalize for cosine sim via dot product (sentence-transformers already normalized;
# TF-IDF vectors need normalizing too)
def l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms

A_emb = l2_normalize(np.asarray(A_emb, dtype=np.float32))
B_emb = l2_normalize(np.asarray(B_emb, dtype=np.float32))

# ---------- 4. Greedy best-match (A -> B), one-to-one ----------

sim_matrix = A_emb @ B_emb.T  # (len(A), len(B))

best_b_idx = sim_matrix.argmax(axis=1)
best_b_score = sim_matrix.max(axis=1)

used_b = set()
matches = []          # confident matches
review_matches = []   # borderline, needs human eyeballing
unmatched_a = []      # no media found at all

# process in descending score order so the best matches claim their B row first
order = np.argsort(-best_b_score)

assigned_b_for_a = {}
for a_idx in order:
    b_idx = int(best_b_idx[a_idx])
    score = float(best_b_score[a_idx])
    if b_idx in used_b:
        # that B row already claimed by a better-scoring A row; try next best for this A
        row_scores = sim_matrix[a_idx].copy()
        row_scores[list(used_b)] = -1
        b_idx = int(row_scores.argmax())
        score = float(row_scores[b_idx])
    assigned_b_for_a[a_idx] = (b_idx, score)
    if score >= MATCH_THRESHOLD:
        used_b.add(b_idx)

for a_idx in range(len(fitness_data)):
    b_idx, score = assigned_b_for_a[a_idx]
    a_row = fitness_data[a_idx]
    b_row = gifs_data[b_idx]
    record = {
        "title": a_row["title"],
        "matched_gif_title": b_row["title"],
        "similarity": round(score, 4),
        "description": a_row.get("description", ""),
        "category": a_row.get("category", ""),
        "body_part": a_row.get("body_part", ""),
        "target_muscle_gifsrc": b_row.get("body_part", ""),  # gifs_data calls it body_part but it's target muscle
        "equipment": a_row.get("equipment", ""),
        "difficulty_level": a_row.get("difficulty_level", ""),
        "rating": a_row.get("rating", ""),
        "gif_url": b_row.get("gif_url", ""),
        "source_a_id": a_row.get("id", ""),
        "source_b_id": b_row.get("id", ""),
    }
    if score >= MATCH_THRESHOLD:
        matches.append(record)
    elif score >= REVIEW_THRESHOLD:
        review_matches.append(record)
    else:
        unmatched_a.append(record)

matched_b_indices = {assigned_b_for_a[a_idx][0] for a_idx in range(len(fitness_data))
                      if assigned_b_for_a[a_idx][1] >= MATCH_THRESHOLD}
unmatched_b = [gifs_data[i] for i in range(len(gifs_data)) if i not in matched_b_indices]

# ---------- 5. Write outputs ----------

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

match_fields = ["title", "matched_gif_title", "similarity", "description", "category",
                "body_part", "target_muscle_gifsrc", "equipment", "difficulty_level",
                "rating", "gif_url", "source_a_id", "source_b_id"]

write_csv(OUT_DIR / "merged_exercises_confident.csv", matches, match_fields)
write_csv(OUT_DIR / "merged_exercises_needs_review.csv", review_matches, match_fields)
write_csv(OUT_DIR / "unmatched_metadata_no_media.csv", unmatched_a, match_fields)

write_csv(OUT_DIR / "unmatched_media_no_metadata.csv", unmatched_b,
          ["id", "title", "body_part", "gif_url", "_norm"])

print("\n=== RESULTS ===")
print(f"Embedding method: {EMBED_METHOD}")
print(f"Confident matches (>= {MATCH_THRESHOLD}):           {len(matches)}")
print(f"Needs human review ({REVIEW_THRESHOLD}-{MATCH_THRESHOLD}):    {len(review_matches)}")
print(f"No match found for rich-metadata row:    {len(unmatched_a)}")
print(f"Gif/media rows with no metadata match:    {len(unmatched_b)}")
print(f"\nOutputs written to: {OUT_DIR}")
