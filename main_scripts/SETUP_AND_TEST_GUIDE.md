# Setup & Test Runbook

## First: do you need to chunk the exercise data? No.

Chunking (`rag/chunk_all_sources.py`) is only for the **PDF guideline
corpus** (ICMR-NIN, WHO, FSSAI, Fit India, your authored notes) — long
documents that need to be split into ~800-char windows so retrieval can
pull the relevant passage instead of the whole PDF.

Exercises are different: each record is already short and atomic (one
exercise = one embedding). `embed_exercises.py` embeds each exercise
row directly — no chunking step involved, and none needed. The two
pipelines are and stay completely separate:

```
PDFs (guidelines)  → chunk_all_sources.py → embed_and_store.py → fitness_guidelines collection
Exercises (merged) → build_exercise_corpus.py → load_db.py (SQL)
                                               → embed_exercises.py (exercise_semantic collection)
```

## Step 0 — merge the two deliveries into one project folder

You have two zips from me so far. Extract both into the same
`backend/` folder, in this order (second overwrites/adds to first, no
conflicts — different files):
1. `adaptive-fitness-planner-refactor.zip` (agent + exercise data merge)
2. Your existing project (everything not touched by either zip stays as-is)

## Step 1 — environment setup

```bash
cd backend
python3 -m venv env
source env/bin/activate          # Windows: env\Scripts\activate

pip install fastapi uvicorn python-dotenv sqlalchemy \
    langchain-core langgraph langchain-groq \
    qdrant-client sentence-transformers requests
```

Create `backend/.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
(Get a free Groq key at console.groq.com if you don't have one.)

## Step 2 — run the data pipeline, in this exact order

```bash
# 1. Exercise data (SQL + exercise_semantic collection)
python scripts/fetch_external_sources.py
python scripts/build_exercise_corpus.py
python scripts/load_db.py
python rag/embed_exercises.py

# 2. Guideline RAG (PDFs — unchanged, but re-run so the json_exercise fix takes effect)
python rag/chunk_all_sources.py
python rag/embed_and_store.py
```

Watch for:
- `load_db.py` should print `TOTAL exercises in DB: 2283` (or close to
  it, ± whatever's in your actual raw files)
- `embed_and_store.py` should now print an `Excluding N chunks with
  doc_type in {'json_exercise'}` line — if you don't see this line,
  the old un-patched version is still on your PATH somewhere
- `embed_exercises.py` should print `'exercise_semantic' — 2283 vectors
  stored` (or similar)

If any of these files/folders are missing (`data/fitness.db` didn't get
created, Qdrant collections empty), stop here and paste me the error —
don't move on to step 3 with broken data underneath.

## Step 3 — start the backend

```bash
uvicorn main:app --reload --port 8000
```

Check `http://localhost:8000/health` responds, then check
`http://localhost:8000/docs` to confirm `/api/conversation/start` and
`/api/conversation/message` are listed (this confirms the new router
wired up correctly).

## Step 4 — start the frontend (separate terminal)

```bash
cd frontend
npm install
npm run dev
```

Open whatever URL Vite prints (usually `http://localhost:5173`) and go
to the Chat tab.

## Step 5 — test cases to actually try, in this order

**A. Pure Q&A (tests the RAG-first fix)**
> "Is it safe to do squats with a knee injury?"

Expect: an answer citing a guideline source, with **no** attempt to
ask you for profile info. If it starts asking "what's your goal?"
instead of answering, the agent is routing wrong — tell me and I'll
look at the system prompt / tool descriptions.

**B. Exercise-specific Q&A (tests the new exercise_semantic layer)**
> "What's a good exercise for lower back pain?"

Expect: specific exercise names + descriptions, not just generic
guideline text. If it returns nothing useful, check that
`embed_exercises.py` actually ran and the collection isn't empty.

**C. Plan request — normal path**
> "I'd like a workout plan"

Expect: it starts asking for profile info conversationally (not a
rigid one-field-at-a-time interrogation — it can group a couple of
questions). Give it goal, body parts, age, gender, equipment, fitness
level, time per day — but **don't** mention health conditions yet.

**D. The safety gate (important — verify this explicitly)**
Keep answering until everything else is filled in, but withhold health
info. Expect: it should **refuse to generate a plan** and specifically
ask about health conditions — this is the hard-coded gate in
`Profile.is_safe_to_plan()`, not a suggestion. If it generates a plan
anyway without asking about health flags, that's a real bug — tell me
immediately, this is the one thing that must not fail.

Then say "none" (or list a real condition) — expect it to proceed to
`generate_plan` only now.

**E. Save + reminder**
After the plan is generated, expect it to ask if you want it
saved/emailed. Give a name + email, confirm — check your inbox (or
your SMTP test setup) for the email.

## Step 6 — what to send back to me

For anything that behaves wrong, the most useful things to paste are:
1. The exact user message you sent
2. The exact agent reply
3. Your terminal/server logs for that turn (LangGraph prints tool
   calls — I need to see which tool it picked and with what args)

That's enough for me to tell whether it's a prompt/tool-description
issue (easy fix) vs. a deeper flow issue.

## Step 7 — once B–D above are solid, we move to deployment

At that point we tackle, in order: Qdrant Cloud migration (get off
local-disk storage), Supabase Postgres migration (get off SQLite), then
the GitHub Actions cron for reminders, then actual hosting config
(Render/Railway + Vercel). Don't start on deployment before the chat
behavior above is verified — debugging agent behavior is much harder
once it's also wrapped in deployment infrastructure.
