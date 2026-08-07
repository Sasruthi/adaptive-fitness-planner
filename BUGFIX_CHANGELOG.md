# Fixes from your test transcript

## Bug #3 — the deadlock (most severe, fixed)
**Root cause**: `BODY_PARTS` only listed specific regions; `"full body"`
was never valid. `merge()` silently dropped it with zero explanation,
so the agent had no signal to self-correct — it looped, invented fake
numbered menus that don't exist in the schema, and eventually
**fabricated a full plan in chat without ever calling `generate_plan`**.
That last part is the one that should never happen in an agentic
system — a tool failing should never be papered over with an
improvised chat answer.

Fixed in `profile_store.py`:
- `"full body"` / `"full_body"` / `"whole body"` / `"total body"` now
  auto-expand to all 10 regions (tested — see transcript below)
- `merge()` now returns *why* something was rejected + the valid
  options, instead of silently dropping it
- Verified with real test:
  ```
  merge(target_body_parts=["full body"]) → all 10 regions stored ✓
  merge(target_body_parts=["bicep zone"]) → rejected, with valid_options listed ✓
  ```

Fixed in `agent.py`:
- `update_profile`'s docstring now lists every valid enum value
  explicitly, so the LLM stops guessing (this is what led it to invent
  the fake "1. full_body 2. upper_body..." menu — none of those
  ever existed in your schema)
- System prompt now has an explicit **NEVER FABRICATE A PLAN** rule,
  and an explicit instruction to read the `rejected` field and correct
  itself instead of resending the same bad value

## Bug #2 — re-asking gender after she said "exercises for women" (fixed)
**Root cause**: `answer_fitness_question` and `update_profile` were
architecturally disconnected — nothing told the agent that facts
revealed while asking a question still count as profile info.

Fixed: system prompt now has an explicit **INFER, DON'T JUST WAIT TO BE
ASKED** rule with this exact example.

## Bug #1 — recommending hill-climbing/squats despite a stated knee injury (fixed, partially)
**Root cause**: `answer_fitness_question(query)` had no access to
already-known health flags at all — pure text in, text out, no
session context.

Fixed: it now pulls `health_flags`/`custom_health_notes` from the
session profile automatically, folds them into the retrieval query, and
explicitly instructs the LLM to *screen out or flag* contraindicated
suggestions rather than append a generic caution note.

**Honest limitation**: this improves grounding and instruction, but if
the underlying guideline PDF content itself doesn't have injury-aware
exercise recommendations, the model can still only work with what's
retrieved. Worth spot-checking this exact scenario again after
re-ingesting, and telling me if "hill walking for someone with a knee
injury" still comes up — that would point to the guideline source
content itself needing review, not the pipeline.

## Video/image gap (backend wired up, frontend needs your file)
Added `video_url` end-to-end: `build_exercise_corpus.py` → `models.py`
→ `load_db.py` → `exercise_retrieval.py` → API schema → the
`answer_fitness_question` tool output → `exercise_rag.py`. Tested
against real data: **24 of 2,283 exercises have a working YouTube URL**
(corrected from my earlier — wrong — guess of "~700"; I hadn't actually
checked before saying that number, checked it this time).

`frontend_ExerciseMedia_snippet.jsx` is a drop-in block for wherever
your `ExerciseCard` renders media — I don't have that file (it wasn't
in the scripts you uploaded, only backend/rag was), so this is a
snippet to adapt, not a patched file.

## What to re-test
Re-run the exact transcript scenario: knee injury → "exercises for
women" → build a plan → say "full body" for target body parts. Confirm:
1. It doesn't re-ask your gender
2. "full body" registers on the first try
3. It actually calls `generate_plan` (check server logs for the tool
   call) rather than writing out a plan in plain chat text
