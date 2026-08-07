# Adaptive Fitness Planner — Full Module Documentation

**Audience:** engineers, reviewers, and handoff partners who need to understand *everything* this module does — architecture, data flow, contracts, scripts, APIs, evals, and operational knobs.

**Companion files in this folder:**
- `MODULE_OUTLINE.md` — short map
- `MANIFEST.txt` — exact scripts copied here
- Live runnable tree lives at the **repo root** (`backend/`, `frontend/`, `evals/`), not only under `main_scripts/`

---

# 1. Overview & goals

## 1.1 What this module is

An **India-first Adaptive Fitness Planner**: a conversational AI product that:

1. Talks to the user in natural language (chat).
2. Collects a structured, validated fitness profile.
3. Answers questions using **retrieval-augmented generation (RAG)** over Indian / WHO / Fit India style guidelines.
4. Retrieves **real exercises** from a catalog (SQLite + semantic Qdrant), with GIF/image/video when available.
5. Builds a **personalised weekly workout + diet plan** (or diet-only), with safety notes and citations.
6. Persists plans and workout logs; optionally emails the plan; shows progress / streak.

## 1.2 Design principles (non-negotiable contracts)

| Principle | Meaning in code |
|-----------|-----------------|
| **Never fabricate a plan in chat prose** | Plans come from `generate_plan` / `plan_generator`, not invented markdown schedules |
| **Hard safety gate in code** | `Profile.is_safe_to_plan()` — health flags + required slots; not left to LLM judgement alone |
| **Filters before synthesis** | Injury → excluded body parts; equipment → SQL/semantic filters |
| **Deterministic plan path by default** | Agentic LLM JSON is never returned raw; always re-grounded through `generate_plan` |
| **Verify numbers** | BMR/TDEE via Mifflin–St Jeor in code; meal macros grounded in INDB when possible |
| **Honest thin programs** | If too few distinct exercises match filters, `stats.thin_program` + UI/agent warning |
| **Intent-aware Q&A** | `info` vs `exercise_qa` vs `plan` so water/nutrition Qs don’t spam exercise GIFs |

## 1.3 Tech stack (summary)

| Layer | Tech |
|-------|------|
| API | FastAPI, Uvicorn |
| Agent | LangGraph + LangChain tools (ReAct) |
| LLM | Groq (default), Azure OpenAI failover, optional Ollama |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Vector DB | Qdrant (local filesystem) |
| Relational | SQLite via SQLAlchemy |
| Frontend | React + Vite + Tailwind |
| Email | SMTP (optional) |

---

# 2. Repository layout

## 2.1 Top-level (live project)

```
adaptive-fitness-planner/
├── backend/                 # FastAPI application (source of truth for runtime)
│   ├── main.py              # App entry, CORS, /health, static media
│   ├── app/
│   │   ├── config.py        # DB_PATH, QDRANT_PATH
│   │   ├── llm.py           # Shared LLM factory + failover
│   │   ├── conversation/    # Chat agent, profile, taxonomies
│   │   ├── services/        # Plan, RAG, NLU, exercises, nutrition
│   │   ├── routers/         # HTTP endpoints
│   │   ├── mcp_server/      # Action tools (save, calories, log, …)
│   │   ├── models/          # SQLAlchemy ORM
│   │   └── schemas/         # Pydantic API models
│   ├── rag/                 # Chunk/embed scripts + qdrant_local/
│   ├── scripts/             # Exercise corpus build + load_db
│   ├── data/                # fitness.db, processed JSON
│   └── static/exercises/    # Symlinked images/videos
├── frontend/                # Vite React UI
├── evals/                   # RAG / exercise / agent eval harness
├── data/                    # Guideline PDFs, hasaneyldrm media, downloads
├── main_scripts/            # THIS folder — curated script + docs snapshot
└── *.md                     # Setup, changelog, merge reports
```

## 2.2 What `main_scripts/` contains

A curated copy of important **code** (and small datasets for evals), plus this documentation. It excludes:

- `node_modules`, Python `env/` / venv  
- Qdrant storage directories  
- Large dumps (`exercises.json`, `INDB.xlsx`, GIF trees)  
- `__pycache__`

Use it for reading and handoff; **run** from the live `backend/` + `frontend/` trees.

---

# 3. End-to-end user journeys

## 3.1 Pure Q&A (no plan)

1. User opens Chat → frontend calls `POST /api/conversation/start`.
2. User asks e.g. “How much water should I drink?” or “Carbs for diabetes in India?”
3. Agent calls `answer_fitness_question`.
4. `classify_turn_intent` → typically `info` (guidelines only) or `exercise_qa` (guidelines + demos).
5. Guideline passages retrieved from Qdrant; for `exercise_qa`, exercises from `exercise_semantic` (+ filters).
6. LLM answers with citations; frontend may show exercise cards with GIF/image.

## 3.2 Profile collection → plan

1. User states goals, body parts, equipment, age, gender, height, weight, health.
2. Agent calls `update_profile` (and/or semantic auto-ingest for some slots).
3. `get_profile_status` / `missing_fields()` until complete.
4. Health flags must be explicit (`["none"]` or real flags) — `custom_health_notes` alone do not unlock planning.
5. Agent calls `generate_plan` → `generate_plan_agentic` → **`generate_plan`** (deterministic).
6. Response stage becomes `plan_ready`; plan held in session; UI can open Plan tab / offer save.

## 3.3 Diet-only mode

1. User asks for diet / nutrition plan without workouts.
2. `plan_mode=diet_only` → exercise slots (body parts, equipment, fitness level, time) **not** required.
3. Height + weight + age + gender + health still required (BMR).
4. Plan has meals / tips; `week_plan` workout days empty or omitted per mode.

## 3.4 Save, log, progress

1. Save with email/name → MCP `save_plan` → SQLite + optional SMTP.
2. Log completed exercises → `log_workout`.
3. Progress page → streak / sessions via `get_workout_progress`.
4. Reminder email → `send_reminder`.

---

# 4. API surface

Entry: `backend/main.py`  
Docs: `http://localhost:8000/docs`

## 4.1 Health

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Name, version, docs links |
| GET | `/health` | SQLite exercise count, Qdrant vector count, Groq key presence |
| GET | `/static/...` | Exercise media (follow_symlink) |

## 4.2 Conversation — `/api/conversation`

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/start` | New `thread_id` + greeting |
| POST | `/message` | User turn → agent reply; may include `plan`, `exercises`, `profile`, filters |

Router: `backend/app/routers/conversation.py`  
Implementation: `app.conversation.agent` (`start_conversation`, `process_user_message`).

**Important:** The chat path can produce a full plan via the agent’s `generate_plan` tool.  
`POST /api/plan/generate` remains for programmatic / non-chat use.

## 4.3 Plan — `/api/plan`

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/generate` | Build plan from profile + filters |
| POST | `/save` | Persist plan + email |
| GET | `/{user_email}` | Latest active plan |
| POST | `/calories` | Mifflin calorie helper |

## 4.4 Workout — `/api/workout`

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/log` | Log completed exercises |
| GET | `/progress/{user_email}` | Streak / stats |
| POST | `/reminder` | Email reminder |

## 4.5 Exercises — `/api/exercises`

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/search` | Filtered / text search |
| GET | `/taxonomy` | Body parts / equipment taxonomy |
| GET | `/{exercise_id}` | Single exercise |

---

# 5. Conversation agent & tools

**File:** `backend/app/conversation/agent.py`  
**Pattern:** LangGraph ReAct agent with tools bound to an LLM.

## 5.1 Session model

- Each chat has a `thread_id` (UUID).
- `session_store` (in `profile_store.py`) holds per-thread:
  - `Profile`
  - latest `plan`
  - latest exercise hit list for UI cards
- ContextVar `_current_thread` routes tool calls to the right session.

## 5.2 Tools (what the LLM can call)

| Tool | Role |
|------|------|
| `answer_fitness_question` | RAG Q&A; intent-gated; may attach exercise demos |
| `update_profile` | Store validated slots; returns changed + **rejected** with valid options |
| `get_profile_status` | Dump profile + missing fields |
| `generate_plan` | Safety-gated plan build; surfaces thin-program / enrichment failures |
| `save_generated_plan` | Persist session plan + email |
| `send_reminder_email` | Reminder |
| `log_completed_workout` | Log session |
| `get_progress` | Streak / stats |

## 5.3 System prompt rules (behavioural)

Documented in the agent system prompt, including:

- Infer profile facts from how the user asks (e.g. “exercises for women” → gender).
- Never invent numbered menus of invalid enum tokens.
- Never fabricate a full plan in chat without `generate_plan`.
- Read `rejected` from `update_profile` and self-correct.
- Ask height/weight before plan (BMR).
- Health flags must be confirmed.

## 5.4 `answer_fitness_question` internals

1. Load profile health / demographics; augment retrieval query.
2. `classify_turn_intent(query)` → `info` | `exercise_qa` | `plan`.
3. Retrieve guideline chunks (`retrieve_multi_query`, Tier 1–2).
4. If `exercise_qa`: semantic exercise retrieval with equipment / body-part / difficulty soft gates.
5. Build a grounded tool message for the LLM (passages + exercise list + instructions).
6. For `info`: forbid inventing workouts / GIF spam.
7. Empty guideline retrieval → explicit “none retrieved” instruction (no fake citations).

## 5.5 `generate_plan` tool return contract

- Blocked if `not profile.is_safe_to_plan()`.
- On success: plan stored in session.
- **Thin program:** explicit warning — agent must not claim a rich full program.
- Diet-only failure: explicit failure if meals empty after enrichment fail.

---

# 6. Profile model, validation, safety gates

**File:** `backend/app/conversation/profile_store.py`

## 6.1 Profile fields

| Field | Notes |
|-------|-------|
| `goal` | Controlled enum (`lose_fat`, `build_muscle`, …) |
| `target_body_parts` | From `BODY_PARTS`; `"full body"` expands to all regions |
| `age`, `gender` | Validated ranges / male\|female |
| `height_cm`, `weight_kg` | Required for plan; ranges 100–250 cm, 25–300 kg |
| `health_flags` | Known flags or `none`; required for planning |
| `custom_health_notes` | Free text (e.g. cholesterol) — RAG enrichment only |
| `available_equipment` | Controlled + synonyms |
| `fitness_level` | beginner \| intermediate \| expert |
| `time_per_day_minutes` | 0–180; `0` allowed for diet-only |
| `plan_mode` | `full` \| `diet_only` |

## 6.2 `merge()` behaviour

- Applies only non-empty updates (except explicit `0` for time).
- Invalid enums → **`rejected`** with `got` + `valid_options` (fixes silent drop deadlocks).
- Full-body aliases expand to all body parts.
- Health: `"none"` alone clears; otherwise merges flags.

## 6.3 `missing_fields()` / `is_safe_to_plan()`

**Always required for any plan:** goal, age, gender, weight_kg, height_cm, health_flags.

**Additionally for `plan_mode=full`:** target_body_parts, available_equipment, fitness_level, time_per_day_minutes.

**Diet-only:** skips the exercise slots above.

`is_safe_to_plan()` = no missing fields **and** health_flags answered (notes alone insufficient).

## 6.4 Taxonomies & filters

**File:** `backend/app/conversation/state.py`

- Controlled lists: goals, body parts, equipment, fitness, activity, health flags.
- `INJURY_BODY_PART_EXCLUSIONS` — single source for hard excludes (knee → legs, etc.).
- `build_sql_filters(profile)` — equipment (bodyweight-only if none), difficulty, body parts, injury excludes.
- `build_rag_filters(profile)` — trust tiers + content types; custom notes influence breadth, not hard SQL blocks.

---

# 7. Semantic NLU & turn intents

**File:** `backend/app/services/semantic_nlu.py`

## 7.1 Role

Replaces brittle keyword/regex routing for:

- Plan mode (diet-only vs full)
- Goal / equipment / fitness / body parts (embedding prototypes)
- Turn intent classification
- Optional LLM demographic/health extract

## 7.2 Turn intents

| Intent | Typical use |
|--------|-------------|
| `info` | Hydration, nutrition facts, guideline questions — **no** exercise cards |
| `exercise_qa` | “Arm fat exercises”, form tips — guidelines + demos |
| `plan` | “Make me a weekly plan” — steer toward profile/plan tools |

## 7.3 `SEMANTIC_LLM_EXTRACT`

- Default: **off** (`"0"`).
- When `"1"`: LLM JSON extract for age, gender, height, weight, health_flags (with negation).
- When off: regex/embedding for some demographics; **does not** auto-set health_flags from embeddings (avoids false unlock of planning). Cholesterol-style notes may still be captured as `custom_health_notes`.

## 7.4 Shared embeddings

**File:** `backend/app/services/embedder.py`  
Loads MiniLM once; used by NLU, guideline RAG, exercise semantic RAG, nutrition lookup.

---

# 8. Guideline RAG

## 8.1 Collection

- Qdrant collection: **`fitness_guidelines`**
- Path: typically `backend/rag/qdrant_local` via `app.config.QDRANT_PATH`  
  (**Ops note:** some retrieval helpers historically defaulted to repo-root `rag/qdrant_local` — set `QDRANT_PATH` explicitly if collections look empty.)

## 8.2 Pipeline (ingest)

```
PDF/raw sources
  → data/download_sources.py          (optional download)
  → backend/rag/chunk_all_sources.py  (~800-char chunks + metadata)
  → backend/rag/embed_and_store.py    → fitness_guidelines
```

Metadata typically includes: `source_name`, `page_number`, `trust_tier` (Tier 1/2), `content_type`, category.

## 8.3 Runtime retrieval

**File:** `backend/app/services/rag_retrieval.py`

- `retrieve_multi_query(queries, filters, top_k_per_query=…)`
- Filters: trust tier, content type, optional category
- Used by: conversation Q&A, plan enrichment, plan-agent `fetch_guidelines`

## 8.4 What guidelines are for

- Ground diet / safety / lifestyle advice in cited passages.
- Soft context for injuries and conditions (plus hard SQL excludes for injuries).
- Generation evals measure relevance, groundedness, citation rate.

---

# 9. Exercise retrieval

Two complementary systems:

## 9.1 Structured SQL catalog

**File:** `backend/app/services/exercise_retrieval.py`  
**Store:** SQLite `exercises` table (~2.2k rows after merge pipeline)

Filters: body_part, equipment, difficulty, exclude_body_parts, media preference, etc.

## 9.2 Semantic exercise RAG

**File:** `backend/app/services/exercise_rag.py`  
**Collection:** `exercise_semantic`  
**Ingest:** `backend/rag/embed_exercises.py` (one embedding per exercise row — **no PDF-style chunking**)

Used when meaning matters (“calves” vs tag “lower legs”), chat demos, and plan day-type fetch.

## 9.3 Program selection

**File:** `backend/app/services/exercise_selection.py`

- `resolve_split(target_body_parts)` → day types for the week  
- `resolve_exercise_sets` / semantic-first selection for each day type  
- Shared by deterministic `plan_generator` and agentic `plan_agent` so split shape stays consistent

## 9.4 Media

- GIF / image / video URLs on exercise rows  
- Served under `/static/…` (symlinks to hasaneyldrm dataset)  
- UI falls back GIF → image when needed  

---

# 10. Plan generation (deterministic) — Module heart

**File:** `backend/app/services/plan_generator.py`  
**Public API used by product:** `generate_plan(profile, sql_filters, rag_filters)`  
Wrapped by `plan_agent.generate_plan_agentic` so callers always get this path’s guarantees.

## 10.1 Pipeline stages

1. **Diet-only branch** — skip workout scheduling when `plan_mode=diet_only`.
2. **Guideline retrieve** — multi-query RAG for diet + safety context.
3. **Schedule** — `select_and_schedule` / exercise selection using SQL + semantic sets; apply injury excludes.
4. **Enrich** — LLM (`synthesize_plan`) adds meal suggestions, safety notes objects, citations, day notes, optional form cues.  
   - Calorie target computed in **code** (Mifflin) and injected; LLM must not recompute.
5. **Ground diet** — `ground_diet_plan_in_nutrition_db` looks up INDB macros for dish names.
6. **Normalize safety_notes** — always `{flag, note, citation?}` for Plan UI.
7. **Stats** — distinct exercises, enrichment_failed, **thin_program** (< 3 distinct exercises), has_bmr.

## 10.2 Activity → TDEE

`_resolve_activity_level(activity_level, fitness_level)` maps:

- sedentary / lightly_active / moderately_active / very_active  
- aliases: moderate, light, active, beginner, intermediate, expert  

## 10.3 Output contract (plan JSON)

```text
plan_id, plan_mode, generated_at
profile_summary   # includes activity_level mapped for persistence
week_plan[]       # day, label, focus, exercises[{name,sets,reps,gif_url,modification,…}], notes
diet_plan         # meals, bmr, tdee, target_calories, macros, tips, hydration
safety_notes[]    # {flag, note, citation}
citations[]
weekly_tips[]
stats             # distinct_exercises_in_split, thin_program, enrichment_failed, has_bmr, …
```

---

# 11. Plan agent (agentic opt-in)

**File:** `backend/app/services/plan_agent.py`

## 11.1 When it runs

- Env `PLAN_AGENTIC=1`  
- Default is **off** → direct `_build_fallback_plan` → `generate_plan` once.

## 11.2 Agentic loop tools

| Tool | Role |
|------|------|
| `fetch_exercises_for_day_type` | Semantic (+ SQL fallback) exercises for one day type |
| `fetch_guidelines` | Guideline passages |
| `get_calorie_target` | Mifflin via MCP (activity aliases resolved) |
| `validate_plan` | Keyword/injury heuristics + count vs time |

Empty retrieval returns structured `{count:0, hint:…}` so the model can widen and retry.

## 11.3 Critical contract

1. Agent may draft JSON after tool calls.  
2. `extract_plan_node` **parses only** — does **not** call `generate_plan` (avoids double RAG/enrich cost).  
3. `generate_plan_agentic` **always** returns `_build_fallback_plan` → one `generate_plan`.  
4. Raw agent JSON is **never** the user-facing plan (would skip SQL filters / INDB / thin-program stats).

## 11.4 Why agentic exists

Self-correcting retrieve → validate loop when enabled (widen equipment, re-fetch day types). Product safety still rests on the deterministic re-ground.

---

# 12. Nutrition, BMR, INDB

## 12.1 Calories

**File:** `backend/app/mcp_server/server.py` → `calculate_calories`

- Mifflin–St Jeor BMR by sex  
- Activity multipliers (with fitness aliases)  
- Goal adjustments (e.g. lose_fat −500)  
- India-leaning macro splits  

Attached onto `diet_plan` as `bmr`, `tdee`, `target_calories`, nested `calorie_target`.

## 12.2 INDB grounding

**Files:**  
- `nutrition_lookup.py` — semantic match dish → macros  
- `nutrition_models.py` — ORM  
- `app/scripts/load_nutrition_db.py` — load `INDB.xlsx` into SQLite  

LLM chooses dish **names**; numbers prefer verified DB values or are marked unverified.

## 12.3 Frontend

`DietPlan.jsx` displays BMR / TDEE / target when present.

---

# 13. MCP server & persistence

**File:** `backend/app/mcp_server/server.py`

| Tool | Persistence / side effect |
|------|---------------------------|
| `save_plan` | Upsert user; deactivate old plans; insert Plan; email; map fitness → Mifflin `activity_level` |
| `log_workout` | WorkoutLog rows |
| `get_user_plan` | Active plan JSON |
| `get_workout_progress` | Aggregates / streak |
| `send_reminder` | SMTP |
| `calculate_calories` | Pure compute |

**ORM:** `backend/app/models/models.py` — User, Plan, WorkoutLog, Exercise, Taxonomy.

---

# 14. LLM providers

**File:** `backend/app/llm.py`

- `LLM_PROVIDER`: groq (default), azure, ollama  
- Groq rate-limit → Azure failover when Azure env is configured  
- Helpers: `get_llm`, `get_llm_with_tools`  
- Used by conversation agent, plan enrichment, optional NLU extract, plan agent  

---

# 15. Frontend architecture

**Stack:** React + Vite + Tailwind  
**API client:** `frontend/src/api/client.js` → `VITE_API_URL`

| Page / component | Role |
|------------------|------|
| `App.jsx` | Shell; Chat / Plan / Progress; lifts plan + email state |
| `ChatPage.jsx` | Conversation; exercise cards; save flow; plan-ready handoff |
| `PlanPage.jsx` | Week / diet / safety tabs; workout logging; safety_notes objects (+ string fallback) |
| `ProgressPage.jsx` | Stats + reminder |
| `ExerciseCard.jsx` | Media + instructions |
| `DayCard.jsx` | Expandable day |
| `DietPlan.jsx` | Meals + BMR/TDEE |
| `ProgressStats.jsx` | Streak UI |

**Save UX:** requires success / plan_id; surfaces `saved.message` on failure.

---

# 16. Data & ingestion pipelines

## 16.1 Two separate pipelines (do not conflate)

```
PDFs / guideline text
  → chunk_all_sources.py → embed_and_store.py → fitness_guidelines

Exercises (atomic records)
  → fetch_external_sources.py
  → build_exercise_corpus.py
  → load_db.py              → SQLite exercises
  → embed_exercises.py      → exercise_semantic
```

Exercises are **not** PDF-chunked; each row is one embedding.

## 16.2 Script index

| Script | Purpose |
|--------|---------|
| `data/download_sources.py` | Download guideline PDFs / layout |
| `backend/scripts/fetch_external_sources.py` | External exercise JSON |
| `backend/scripts/build_exercise_corpus.py` | Merge / dedupe corpus + taxonomy |
| `backend/scripts/load_db.py` | Load SQLite |
| `backend/rag/embed_exercises.py` | Exercise vectors |
| `backend/rag/chunk_all_sources.py` | Guideline chunks |
| `backend/rag/embed_and_store.py` | Guideline vectors |
| `backend/app/scripts/load_nutrition_db.py` | INDB → nutrition_items |
| `backend/scripts/ingest_rag.py` | Alternate PDF ingest |
| `backend/scripts/merge_exercises.py` | Older embedding merge |
| `backend/scripts/pull_github_exercises.py` | GitHub puller |

Canonical order is also documented in `SETUP_AND_TEST_GUIDE.md`.

---

# 17. Evaluation harness

**Folder:** `evals/`

| Runner | Measures |
|--------|----------|
| `run_rag_eval.py` | Guideline Hit@k, Precision@k, MRR; answer relevance, groundedness, citation rate; **non-zero exit** if below thresholds |
| `run_exercise_eval.py` | Name Hit@k, equipment/apparatus, instructions, media, diet-intent |
| `run_agent_eval.py` | Slot / intent accuracy via `semantic_nlu` (threshold ≥ 0.75) |
| `run_all_evals.py` | Orchestrates all; aggregates exit codes |
| `metrics.py` | Shared IR/generation metrics; **empty gold labels do not auto-pass** |
| `qdrant_lock.py` | Temp-copy Qdrant if uvicorn holds the local lock |

**Datasets:** `evals/datasets/{guideline_rag,exercise_rag,agent_slots}.jsonl`

```bash
cd backend && source ../env/bin/activate
python ../evals/run_all_evals.py
```

---

# 18. Configuration & environment

## 18.1 Paths (`app/config.py`)

- `DB_PATH` → `backend/data/fitness.db`  
- `QDRANT_PATH` → `backend/rag/qdrant_local`  

## 18.2 Important env vars

| Variable | Role |
|----------|------|
| `LLM_PROVIDER` | groq / azure / ollama |
| `GROQ_API_KEY`, `GROQ_MODEL` | Primary LLM |
| `AZURE_OPENAI_*` | Failover / alternate |
| `OLLAMA_MODEL` | Local LLM |
| `FRONTEND_URL` | Extra CORS origin |
| `VITE_API_URL` | Frontend → API base |
| `QDRANT_PATH` | Override vector store dir |
| `PLAN_AGENTIC` | `1` = agentic retrieve/validate before generate |
| `SEMANTIC_LLM_EXTRACT` | `1` = LLM profile extract in NLU |
| `SMTP_*` | Plan / reminder email |

## 18.3 Local run

```bash
# Backend
cd backend && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm run dev
```

Health: `GET http://localhost:8000/health`

---

# 19. Safety & design contracts (checklist)

1. **Injury:** `health_flags` → `exclude_body_parts` in SQL/semantic selection.  
2. **Equipment:** none/body-only → bodyweight equipment set; else equipment + bodyweight.  
3. **Planning unlock:** explicit `health_flags` required.  
4. **Calories:** need age, sex, height, weight; activity mapped honestly.  
5. **safety_notes:** objects for UI; thin_program / enrichment flagged.  
6. **Agent thin plan:** tool string forces honest user messaging.  
7. **No raw agentic plan JSON** to clients.  
8. **Intent gating** prevents nutrition Qs becoming workout carousels.  
9. **Rejected profile values** returned to LLM for self-correction.  
10. **Evals fail closed** on weak retrieval / generation / slot scores.

---

# 20. Known limitations & ops notes

| Topic | Note |
|-------|------|
| Auth | Email-based plan APIs are not strongly authenticated (left as product follow-up) |
| Profile auto health flags | With LLM extract off, conditions like “diabetic” may rely on `update_profile` tool calls |
| Agentic mode cost | `PLAN_AGENTIC=1` runs an extra LLM tool loop; draft still discarded after one `generate_plan` |
| Guideline quality | Injury-aware advice quality depends on PDF corpus content, not only filters |
| Dual Qdrant paths | Ensure ingest path == runtime `QDRANT_PATH` |
| Session store | In-memory — lost on process restart; plans should be saved for durability |
| Media coverage | Only a subset of exercises have YouTube/video URLs |
| Legacy | `intake_graph.py` is superseded by the unified agent; kept for reference |

---

# 21. File index (`main_scripts` copy)

See `MANIFEST.txt` for the exact list. Logical groups:

### Backend entry & infra
`backend/main.py`, `app/config.py`, `app/llm.py`

### Conversation
`agent.py`, `profile_store.py`, `state.py`, `intake_graph.py` (legacy)

### Services
`plan_generator.py`, `plan_agent.py`, `semantic_nlu.py`, `rag_retrieval.py`, `exercise_rag.py`, `exercise_retrieval.py`, `exercise_selection.py`, `embedder.py`, `nutrition_lookup.py`

### MCP / models / schemas / routers
`mcp_server/server.py`, `models/*`, `schemas/models.py`, `routers/{conversation,plan,workout,exercises}.py`

### Ingest
`scripts/*`, `rag/chunk_all_sources.py`, `rag/embed_and_store.py`, `rag/embed_exercises.py`, `app/scripts/load_nutrition_db.py`, `data/download_sources.py`

### Frontend
`App.jsx`, `main.jsx`, `api/client.js`, pages + key components

### Evals
All runners, `metrics.py`, `qdrant_lock.py`, datasets JSONL

---

# 22. Glossary

| Term | Meaning |
|------|---------|
| **Thread** | One chat session id |
| **Slot** | Profile field the agent must fill |
| **SQL filters** | Structured exercise query constraints |
| **RAG filters** | Metadata filters for guideline vectors |
| **Thin program** | Fewer than 3 distinct exercises after filtering |
| **Trust tier** | Source quality label on guideline chunks |
| **INDB** | Indian Nutrient Databank used for meal macros |
| **ReAct** | Reason + Act agent loop with tools |
| **Diet-only** | Plan mode without workout requirements |

---

# 23. Recommended reading order for a new engineer

1. This document §§1–3, 5–6, 10–11  
2. `backend/main.py` + `routers/conversation.py`  
3. `conversation/agent.py` (tools + prompt)  
4. `profile_store.py` + `state.py`  
5. `plan_generator.py` then `plan_agent.py`  
6. `rag_retrieval.py` + `exercise_rag.py` + `semantic_nlu.py`  
7. `SETUP_AND_TEST_GUIDE.md` + `evals/README.md`  
8. Frontend `ChatPage.jsx` → `PlanPage.jsx`  

---

*Documentation generated for the Adaptive Fitness Planner module snapshot in `main_scripts/`. Prefer the live repo tree for execution and ongoing edits.*
