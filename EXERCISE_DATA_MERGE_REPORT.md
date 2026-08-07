# Exercise Data Pipeline — Merge Report

Ran the full pipeline against your real uploaded files + the actual
GitHub repos (cloned, not guessed from READMEs). Numbers below are from
a real execution, not estimates.

## What went in, what didn't, and why

| Source | Records contributed | Decision |
|---|---|---|
| `free_exercise_db_exercises.json` (your upload) | 742 winning records | **Included** — base strength/stretching coverage |
| `exercises.json` / hasaneyldrm (your upload) | 1,316 winning records | **Included, highest priority** — gives the exact 10-value body_part taxonomy, multi-language instructions (incl. Hindi) |
| `exercemus/exercises` main list | 12 winning records | **Included** — diffed directly: 852/872 names identical to free-exercise-db, only 12 won the dedup after also checking `exercises_to_merge` |
| `exercemus/exercises` `exercises_to_merge` (unreviewed wger.de) | 161 winning records | **Included** — real wger.de data exercemus hasn't merged into their main list yet, but legitimate |
| `longhaul-fitness` cardio.json + flexibility.json | 52 winning records | **Included** — small, but fills a real gap (your other sources have almost no dedicated cardio/flexibility) |
| `longhaul-fitness` strength.json (349 records) | 0 | **Excluded by default** — has no `equipment` field at all, would be unusable for your hard equipment filtering. Flip `INCLUDE_LONGHAUL_STRENGTH = True` in `build_exercise_corpus.py` if you'd rather have them tagged `equipment="body only"`-by-default than not have them. |
| `wrkout/exercises.json` | 0 | **Excluded entirely** — cloned and diffed directly: 873/873 names match `free-exercise-db`, byte-identical instructions text. This is confirmed to be the literal upstream source free-exercise-db was restructured from. Zero incremental value. |

## Real numbers from running it

```
Total raw records before dedup: 3314
After name-dedup: 2283

By body_part: upper arms=441, upper legs=414, back=336, shoulders=290,
              waist=281, chest=260, lower legs=160, lower arms=62,
              cardio=29, neck=10

By source (winning record): hasaneyldrm=1316, free_exercise_db=742,
              exercemus_unmerged_wger=161, longhaul_fitness=52, exercemus=12

Taxonomy derived (no separate files needed): 10 body_parts, 41 equipment
types, 70 muscles — 121 Taxonomy rows total
```

I also ran the resulting DB against your actual `exercise_retrieval.py`
query pattern (body_part + equipment ILIKE filters) and confirmed it
returns correct results.

## One real data gap surfaced during testing

**0 exercises have a working `gif_url`.** I checked: hasaneyldrm's export
has a populated `media_id` field (e.g. `"ila4NZS"`) on every record, but
`gif_url` itself is `null` throughout — the actual media isn't in the
JSON export, just an ID that presumably resolves against a CDN the
LogPress app uses internally that isn't documented in the repo. Your
`has_media`/`gif_url` frontend rendering (`ExerciseCard`, `DayCard`) will
show no images until this is resolved. Options: find the CDN URL
pattern (may require reverse-engineering their app), or plan on a
different media source for now.

## How to run it

```bash
python scripts/fetch_external_sources.py     # downloads exercemus + longhaul-fitness
python scripts/build_exercise_corpus.py      # merges everything, writes exercises_merged.json + taxonomy.json
python scripts/load_db.py                    # loads into fitness.db (replaces old broken version)
```

Your two uploaded files are already placed at
`data/adaptive-fitness-planner-data/raw/structured/` in this package.

## Files in this package (this delivery only — see previous zip for the agent refactor)

```
backend/
  scripts/
    fetch_external_sources.py   NEW
    build_exercise_corpus.py    NEW — the merge/dedup engine
    load_db.py                   REWRITTEN — no more hardcoded sandbox paths,
                                  no more nonexistent bodyParts.json/etc.
  data/adaptive-fitness-planner-data/
    raw/structured/               your 2 uploaded files, included
    raw/external/                 fetch_external_sources.py populates this
    processed/                    build_exercise_corpus.py writes here
```

## Known rough edges, honestly flagged

- Equipment taxonomy has some near-duplicate variants across sources
  (`"bands"` vs `"band"`, `"kettlebell"` vs `"kettlebells"`, `"ez curl
  bar"` vs `"e-z curl bar"` vs `"ez barbell"`) — a canonicalization pass
  would tighten this further. Didn't do it now to avoid guessing at
  merges you might disagree with; happy to add if you want a specific
  canonical list.
- `muscles_to_body_part()`'s fallback (any muscle string that matches
  nothing) defaults to `"waist"` on the reasoning that ab/core work is
  the least likely to be wrong — but it's a heuristic, worth spot-checking
  the `neck` count (only 10) since that's the thinnest bucket.
