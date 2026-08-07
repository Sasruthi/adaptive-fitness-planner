"""
Adaptive Fitness Planner — Unified Chunking Pipeline
======================================================
Single script that walks the entire project folder structure,
detects each file type, applies the correct chunking strategy,
attaches rich metadata from source_manifest.csv, and writes
all chunks to chunks/all_chunks.json ready for embedding.

Folder structure handled:
  RAW/india_guidelines/     → Tier 1 India PDFs  (primary RAG source)
  RAW/global_generic/       → Tier 2 WHO PDFs    (fallback RAG source)
  RAW/structured/           → JSON/TXT exercises  (DB only, NOT embedded)
  RAW/authored/             → custom notes        (Tier 4, embedded if present)
  MANIFESTS/                → CSV metadata        (skipped, used internally)

Chunking strategies:
  PDF short structured  (≤30 pages, >60% text)  → page-level + sub-split
  PDF long prose        (>30 pages, >60% text)  → section-aware sliding window
  PDF image-heavy       (<40% text pages)       → page-level (text-only pages)
  JSON exercise records                         → one chunk per exercise record
  TXT (repo URLs)                               → skipped (not RAG content)
  CSV                                           → row-grouped chunks with headers
"""

import csv, json, re, os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
from collections import defaultdict

import fitz  # PyMuPDF

# ── Paths (relative to project root) ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "data" / "adaptive-fitness-planner-data"
RAW          = DATA_ROOT / "raw"
MANIFESTS    = DATA_ROOT / "manifests"
CHUNK_DIR    = DATA_ROOT / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

folders = {
    "india_guidelines": RAW / "india_guidelines",
    "global_generic":   RAW / "global_generic",
    "structured":       RAW / "structured",
    "authored":         RAW / "authored",
}

# ── Chunking constants ────────────────────────────────────────────────────────
CHUNK_TARGET  = 800    # target chars per chunk
CHUNK_OVERLAP = 150    # sentence overlap between consecutive chunks
MIN_CHUNK     = 80     # discard anything shorter (headers, page numbers)
MAX_CHUNK     = 1200   # hard ceiling — oversized chunks get sub-split

# ── Chunk dataclass ───────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:      str
    text:          str
    # Source provenance
    source_id:     str
    source_name:   str
    trust_tier:    str
    intended_use:  str
    country:       str
    folder:        str          # which subfolder this came from
    filename:      str
    # Retrieval metadata
    page_number:   Optional[int]
    section_title: Optional[str]
    content_type:  str          # nutrition | exercise | lifestyle | safety_medical | general
    doc_type:      str          # pdf_structured | pdf_prose | pdf_image | json_exercise | csv | txt
    char_count:    int
    # For JSON exercise records
    exercise_name: Optional[str] = None
    tags:          List[str]    = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 1. MANIFEST LOADER — resolves filename → source metadata
# ══════════════════════════════════════════════════════════════════════════════

# Filename → source_id mapping (accounts for naming differences between
# what was downloaded vs what's in the manifest)
FILENAME_TO_SOURCE_ID: Dict[str, str] = {
    # india_guidelines
    "icmr_nin_dietary_guidelines_indians_2024.pdf":  "SRC001",
    "nin_dietary_guidelines_website_copy.pdf":        "SRC002",
    "fit_india_fitness_protocols_18_65.pdf":          "SRC003",
    "nin_dgi_booklet_english.pdf":                   "SRC004",
    "icmr_nutrient_requirements_press_release.pdf":   "SRC005",
    "fssai_eat_right_india_handbook.pdf":             "SRC006",
    "fssai_do_you_eat_right.pdf":                    "SRC007",
    "nin_nutrition_lifestyle_immunity.pdf":           "SRC008",
    "common_yoga_protocol.pdf":                      "SRC009",
    # global_generic
    "who_physical_activity_sedentary_guidelines_2020.pdf": "SRC010",
    # structured
    "free_exercise_db_exercises.json":               "SRC011",
    "hasaneyldrm_exercises_dataset_repo.txt":        "SRC012",
    "wrkout_exercises_repo.txt":                     "SRC013",
    "exercemus_exercises_repo.txt":                  "SRC014",
    "longhaul_fitness_exercises_repo.txt":           "SRC015",
}

def load_manifest(manifests_dir: Path) -> Dict[str, dict]:
    """Returns source_id → manifest row dict."""
    manifest_file = manifests_dir / "source_manifest.csv"
    if not manifest_file.exists():
        print(f"[WARN] Manifest not found at {manifest_file}, using defaults")
        return {}
    result = {}
    with open(manifest_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["source_id"]] = row
    return result

def resolve_meta(filename: str, folder_name: str, manifest: Dict[str, dict]) -> dict:
    """Returns metadata dict for a given file."""
    source_id = FILENAME_TO_SOURCE_ID.get(filename, f"UNKNOWN_{filename}")
    row = manifest.get(source_id, {})

    # Infer country from folder
    country = "India" if folder_name == "india_guidelines" else \
              "Global" if folder_name == "global_generic" else "Unknown"

    # Infer category from intended_use
    use = row.get("intended_use", "")
    category = (
        "exercise"       if use in ("activity_guidance", "structured_exercises") else
        "nutrition"      if use in ("grounding", "backup_grounding", "diet_behavior",
                                    "nutrient_reference") else
        "lifestyle"      if use == "lifestyle" else
        "general"
    )

    return {
        "source_id":    source_id,
        "source_name":  row.get("source_name", filename),
        "trust_tier":   row.get("trust_tier", "Unknown"),
        "intended_use": use,
        "country":      country,
        "category":     category,
        "status":       row.get("status", "unknown"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTENT-TYPE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

CONTENT_SIGNALS = {
    "exercise":       ["exercise", "physical activity", "yoga", "walk", "strength",
                       "muscle", "workout", "stretch", "aerobic", "cardio", "posture",
                       "asana", "rep", "set ", "squat", "push", "pull"],
    "nutrition":      ["diet", "food", "eat", "protein", "carbohydrate", "fat",
                       "calorie", "nutrient", "vitamin", "mineral", "fibre", "sugar",
                       "vegetable", "fruit", "grain", "pulse", "dal", "roti", "rice"],
    "safety_medical": ["bp", "blood pressure", "diabetes", "heart", "chronic",
                       "disease", "injury", "hypertension", "obesity", "bmi",
                       "medication", "condition", "avoid if", "contraindicated"],
    "lifestyle":      ["sleep", "stress", "habit", "hydration", "water intake",
                       "mental health", "wellbeing", "immunity", "lifestyle"],
    "guideline":      ["guideline", "recommendation", "should", "must", "limit",
                       "avoid", "restrict", "ensure", "daily intake", "per day"],
}

def classify_content(text: str) -> str:
    t = text.lower()
    scores = {ctype: sum(1 for kw in kws if kw in t)
              for ctype, kws in CONTENT_SIGNALS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

HEADING_PATTERNS = [
    re.compile(r"^(Guideline\s+\d+)", re.IGNORECASE),
    re.compile(r"^(Chapter\s+\d+[\.:]\s*\S)", re.IGNORECASE),
    re.compile(r"^(Section\s+\d+)", re.IGNORECASE),
    re.compile(r"^\d+\.\d*\s+[A-Z][A-Za-z\s]{4,60}$"),
    re.compile(r"^[A-Z][A-Z\s\-]{8,50}$"),           # ALL CAPS headings
    re.compile(r"^(Introduction|Summary|Background|Recommendations|"
               r"Physical Activity|Nutrition|Diet|Exercise|References)", re.IGNORECASE),
]

def detect_heading(text: str) -> Optional[str]:
    for line in text.split("\n")[:8]:
        line = line.strip()
        if 5 < len(line) < 130:
            for pat in HEADING_PATTERNS:
                if pat.match(line):
                    return line
    return None

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\f', '\n', text)
    return text.strip()

def sliding_window(text: str,
                   size: int = CHUNK_TARGET,
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sentence-boundary sliding window chunker."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf, buf_len = [], [], 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if buf_len + len(sent) > size and buf:
            chunk = " ".join(buf).strip()
            if len(chunk) >= MIN_CHUNK:
                chunks.append(chunk)
            # keep overlap
            dropped = 0
            while buf and dropped < overlap:
                dropped += len(buf.pop(0))
            buf_len = sum(len(s) for s in buf)
        buf.append(sent)
        buf_len += len(sent)
    if buf:
        chunk = " ".join(buf).strip()
        if len(chunk) >= MIN_CHUNK:
            chunks.append(chunk)
    return chunks

def safe_chunks(text: str) -> List[str]:
    """Ensure no chunk exceeds MAX_CHUNK; sub-split if needed."""
    if len(text) <= MAX_CHUNK:
        return [text] if len(text) >= MIN_CHUNK else []
    return sliding_window(text)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PDF CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

def analyse_pdf(doc: fitz.Document):
    """Returns (total_pages, text_page_count, text_ratio)."""
    total = len(doc)
    text_pgs = sum(1 for pg in doc if len(pg.get_text().strip()) > 100)
    return total, text_pgs, text_pgs / total if total else 0

def choose_pdf_strategy(total_pages: int, text_ratio: float) -> str:
    if text_ratio < 0.4:
        return "pdf_image"          # Mostly image — extract what text exists, page-level
    if total_pages <= 30:
        return "pdf_structured"     # Short structured (NIN booklet, press release, yoga)
    return "pdf_prose"              # Long prose (FSSAI 136p, WHO, ICMR full guidelines)

def chunk_pdf(pdf_path: Path, meta: dict, folder_name: str) -> List[Chunk]:
    doc = fitz.open(str(pdf_path))
    total_pages, text_pages, text_ratio = analyse_pdf(doc)
    strategy = choose_pdf_strategy(total_pages, text_ratio)

    print(f"    pages={total_pages} | text_ratio={text_ratio:.0%} | strategy={strategy}")

    chunks: List[Chunk] = []
    chunk_counter = [0]  # mutable for inner fn

    def make_chunk(text: str, page_num: int,
                   section_title: Optional[str] = None,
                   sub_idx: int = 0) -> Chunk:
        suffix = f"_s{sub_idx:02d}" if sub_idx else ""
        cid = f"{meta['source_id']}_p{page_num:03d}{suffix}"
        chunk_counter[0] += 1
        return Chunk(
            chunk_id=cid,
            text=text,
            source_id=meta["source_id"],
            source_name=meta["source_name"],
            trust_tier=meta["trust_tier"],
            intended_use=meta["intended_use"],
            country=meta["country"],
            folder=folder_name,
            filename=pdf_path.name,
            page_number=page_num,
            section_title=section_title,
            content_type=classify_content(text),
            doc_type=strategy,
            char_count=len(text),
        )

    # ── pdf_structured / pdf_image: page-level, then sub-split if oversized ──
    if strategy in ("pdf_structured", "pdf_image"):
        for i, page in enumerate(doc):
            raw = clean_text(page.get_text())
            if len(raw) < MIN_CHUNK:
                continue
            heading = detect_heading(raw)
            sub_texts = safe_chunks(raw)
            for si, st in enumerate(sub_texts):
                chunks.append(make_chunk(st, i + 1, heading, si if len(sub_texts) > 1 else 0))

    # ── pdf_prose: section-aware — accumulate pages between headings ──────────
    elif strategy == "pdf_prose":
        section_buf: List[str]  = []
        section_title: Optional[str] = None
        section_page: int = 1

        def flush_section():
            if not section_buf:
                return
            full = " ".join(section_buf)
            for si, st in enumerate(sliding_window(full)):
                chunks.append(make_chunk(st, section_page, section_title, si))

        for i, page in enumerate(doc):
            raw = clean_text(page.get_text())
            if len(raw) < MIN_CHUNK:
                continue
            heading = detect_heading(raw)
            if heading and section_buf:
                flush_section()
                section_buf = [raw]
                section_title = heading
                section_page = i + 1
            else:
                if heading and not section_buf:
                    section_title = heading
                    section_page = i + 1
                section_buf.append(raw)

        flush_section()

    doc.close()
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 5. JSON CHUNKER (exercise records)
# ══════════════════════════════════════════════════════════════════════════════

def chunk_json(json_path: Path, meta: dict, folder_name: str) -> List[Chunk]:
    """One chunk per exercise record — preserves structure for retrieval."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"    [SKIP] JSON is not a list: {json_path.name}")
        return []

    chunks: List[Chunk] = []
    for i, record in enumerate(data):
        name = record.get("name", record.get("title", f"exercise_{i}"))

        # Build a readable text blob from the record fields
        parts = []
        if record.get("name"):
            parts.append(f"Exercise: {record['name']}")
        if record.get("instructions"):
            instr = record["instructions"]
            if isinstance(instr, list):
                instr = " ".join(instr)
            parts.append(f"Instructions: {instr}")
        for field in ["description", "category", "level", "equipment",
                      "primaryMuscles", "secondaryMuscles", "bodyParts",
                      "targetMuscles", "force", "mechanic"]:
            val = record.get(field)
            if val:
                if isinstance(val, list):
                    val = ", ".join(val)
                parts.append(f"{field.title()}: {val}")

        text = "\n".join(parts)
        if len(text) < MIN_CHUNK:
            continue

        tags = []
        for f2 in ["primaryMuscles", "bodyParts", "targetMuscles", "category",
                   "equipment", "level"]:
            v = record.get(f2)
            if v:
                tags += v if isinstance(v, list) else [v]

        sub_texts = safe_chunks(text) if len(text) > MAX_CHUNK else [text]
        for si, sub_text in enumerate(sub_texts):
            suffix = f"_s{si:02d}" if len(sub_texts) > 1 else ""
            chunks.append(Chunk(
                chunk_id=f"{meta['source_id']}_ex{i:04d}{suffix}",
                text=sub_text,
                source_id=meta["source_id"],
                source_name=meta["source_name"],
                trust_tier=meta["trust_tier"],
                intended_use=meta["intended_use"],
                country=meta["country"],
                folder=folder_name,
                filename=json_path.name,
                page_number=None,
                section_title=name,   # exercise name as section title for retrieval
                content_type="exercise",
                doc_type="json_exercise",
                char_count=len(sub_text),
                exercise_name=name,
                tags=list(set(t.lower() for t in tags if t)),
            ))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 6. FILE ROUTER — picks the right chunker per file
# ══════════════════════════════════════════════════════════════════════════════

SKIP_EXTENSIONS = {".txt"}    # repo URL placeholder files — no RAG value
SKIP_FILENAMES  = {"source_manifest.csv", ".~lock.source_manifest.csv#"}

def should_skip(path: Path, meta: dict) -> Optional[str]:
    if path.name in SKIP_FILENAMES:
        return "manifest file"
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return "placeholder txt (repo URL only)"
    if meta.get("status") == "failed":
        return "download failed"
    if meta.get("intended_use") == "structured_exercises":
        # JSON exercise files still get chunked (for exercise-specific RAG)
        # TXT repo URLs already caught above
        if path.suffix.lower() != ".json":
            return "structured exercise repo placeholder"
    return None

def process_file(path: Path, folder_name: str,
                 manifest: Dict[str, dict]) -> List[Chunk]:
    meta = resolve_meta(path.name, folder_name, manifest)
    skip_reason = should_skip(path, meta)
    if skip_reason:
        print(f"  [SKIP] {path.name} — {skip_reason}")
        return []

    ext = path.suffix.lower()
    print(f"  [{ext.upper()[1:]}] {path.name}")
    print(f"    source={meta['source_id']} | tier={meta['trust_tier']} "
          f"| use={meta['intended_use']}")

    if ext == ".pdf":
        return chunk_pdf(path, meta, folder_name)
    elif ext == ".json":
        chunks = chunk_json(path, meta, folder_name)
        print(f"    → {len(chunks)} exercise chunks")
        return chunks
    elif ext == ".csv":
        # CSV in RAW folders (not manifest) — treat as authored notes
        return chunk_csv(path, meta, folder_name)
    else:
        print(f"    [WARN] Unhandled extension {ext}")
        return []


def chunk_csv(csv_path: Path, meta: dict, folder_name: str) -> List[Chunk]:
    """Row-grouped CSV chunker for authored notes / custom data."""
    chunks = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        buf, buf_len, group_idx = [], 0, 0
        for row in reader:
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            buf.append(row_text)
            buf_len += len(row_text)
            if buf_len >= CHUNK_TARGET:
                text = "\n".join(buf)
                chunks.append(Chunk(
                    chunk_id=f"{meta['source_id']}_csv_g{group_idx:04d}",
                    text=text,
                    source_id=meta["source_id"],
                    source_name=meta["source_name"],
                    trust_tier=meta["trust_tier"],
                    intended_use=meta["intended_use"],
                    country=meta["country"],
                    folder=folder_name,
                    filename=csv_path.name,
                    page_number=None,
                    section_title=", ".join(headers[:4]) if headers else None,
                    content_type=classify_content(text),
                    doc_type="csv",
                    char_count=len(text),
                ))
                buf, buf_len, group_idx = [], 0, group_idx + 1
        if buf:
            text = "\n".join(buf)
            if len(text) >= MIN_CHUNK:
                chunks.append(Chunk(
                    chunk_id=f"{meta['source_id']}_csv_g{group_idx:04d}",
                    text=text,
                    source_id=meta["source_id"],
                    source_name=meta["source_name"],
                    trust_tier=meta["trust_tier"],
                    intended_use=meta["intended_use"],
                    country=meta["country"],
                    folder=folder_name,
                    filename=csv_path.name,
                    page_number=None,
                    section_title=", ".join(headers[:4]) if headers else None,
                    content_type=classify_content(text),
                    doc_type="csv",
                    char_count=len(text),
                ))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN WALK — processes all folders in order
# ══════════════════════════════════════════════════════════════════════════════

def run():
    manifest = load_manifest(MANIFESTS)
    print(f"Manifest loaded: {len(manifest)} entries\n")

    all_chunks: List[Chunk] = []
    stats: Dict[str, dict] = defaultdict(lambda: {"files": 0, "chunks": 0})

    for folder_name, folder_path in folders.items():
        if not folder_path.exists():
            print(f"[WARN] Folder not found, skipping: {folder_path}")
            continue

        files = sorted(folder_path.iterdir())
        print(f"\n{'='*60}")
        print(f"FOLDER: {folder_name}  ({len(files)} files)")
        print('='*60)

        for fpath in files:
            if not fpath.is_file():
                continue
            chunks = process_file(fpath, folder_name, manifest)
            all_chunks.extend(chunks)
            if chunks:
                stats[folder_name]["files"] += 1
                stats[folder_name]["chunks"] += len(chunks)
                print(f"    → {len(chunks)} chunks\n")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = CHUNK_DIR / "all_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f,
                  indent=2, ensure_ascii=False)

    # ── Summary ─────────────────────────────────────────────────────────────
    from collections import Counter
    print(f"\n{'='*60}")
    print(f"CHUNKING COMPLETE")
    print(f"{'='*60}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output:        {out_path}\n")

    print("By folder:")
    for fn, s in stats.items():
        print(f"  {fn}: {s['files']} files → {s['chunks']} chunks")

    print("\nBy trust tier:")
    for tier, n in Counter(c.trust_tier for c in all_chunks).most_common():
        print(f"  {tier}: {n}")

    print("\nBy content_type:")
    for ct, n in Counter(c.content_type for c in all_chunks).most_common():
        print(f"  {ct}: {n}")

    print("\nBy doc_type:")
    for dt, n in Counter(c.doc_type for c in all_chunks).most_common():
        print(f"  {dt}: {n}")

    sizes = [c.char_count for c in all_chunks]
    if sizes:
        import statistics
        print(f"\nChunk size — min:{min(sizes)} | "
              f"max:{max(sizes)} | "
              f"median:{statistics.median(sizes):.0f} | "
              f"avg:{sum(sizes)/len(sizes):.0f}")

    return all_chunks

if __name__ == "__main__":
    run()