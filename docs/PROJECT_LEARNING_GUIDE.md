# Adaptive Fitness Planner — Deep Learning Guide

A study document for understanding this project end-to-end: architecture, request flows, LLM call sites, design tradeoffs, interview prep, and why each tool was chosen.

> **Naming note:** **React** (frontend UI library) ≠ **ReAct** (Reason + Act agent pattern). Both appear in this repo.

---

## Table of contents

1. [What this product does](#1-what-this-product-does)
2. [High-level architecture](#2-high-level-architecture)
3. [Technical mental model — how an input is processed layer by layer](#3-technical-mental-model--how-an-input-is-processed-layer-by-layer)
4. [Deep dive: “how to do pranayama”](#4-deep-dive-how-to-do-pranayama)
5. [Other important flows](#5-other-important-flows)
6. [Where every LLM call happens](#6-where-every-llm-call-happens)
7. [What is NOT an LLM call](#7-what-is-not-an-llm-call)
8. [Safety, grounding, and autonomy boundaries](#8-safety-grounding-and-autonomy-boundaries)
9. [Data pipelines (offline)](#9-data-pipelines-offline)
10. [Evaluation harness](#10-evaluation-harness)
11. [Tools & frameworks — why chosen, vs alternatives](#11-tools--frameworks--why-chosen-vs-alternatives)
12. [Interview practice — questions an AI engineer interviewer might ask](#12-interview-practice--questions-an-ai-engineer-interviewer-might-ask)
13. [Suggested study order](#13-suggested-study-order)
14. [Quick file map](#14-quick-file-map)
15. [Live behind-the-scenes trace (real run)](#15-live-behind-the-scenes-trace-real-run)
16. [Chunking strategy used in this project](#16-chunking-strategy-used-in-this-project)

---

## 1. What this product does

India-first personalised fitness assistant:

1. User chats in natural language (React UI).
2. A LangGraph **ReAct** agent decides per turn whether to:
   - answer a fitness/nutrition/yoga question (RAG + optional demo media),
   - collect profile slots conversationally,
   - generate a weekly plan,
   - save / email / log workouts / show progress.
3. Plans are **not invented as free chat prose** — they go through `generate_plan` → mostly deterministic `plan_generator` (schedule + LLM enrichment + INDB grounding).

---

## 2. High-level architecture

```text
┌─────────────┐     HTTP JSON      ┌──────────────┐
│  React UI   │ ─────────────────► │   FastAPI    │
│ Chat/Plan/  │ ◄───────────────── │   routers    │
│ Progress    │                    └──────┬───────┘
└─────────────┘                           │
                                          ▼
                               ┌─────────────────────┐
                               │ LangGraph ReAct     │
                               │ conversation agent  │
                               │ (tool-calling loop) │
                               └──────────┬──────────┘
                                          │
          ┌───────────────┬───────────────┼───────────────┬──────────────┐
          ▼               ▼               ▼               ▼              ▼
   answer_fitness   update_profile   generate_plan   save / email   log / progress
   _question              │               │               │              │
          │               ▼               │         SQLite + SMTP   SQLite
          │         ProfileStore          │
          ▼                               ▼
   Text RAG + CLIP images          plan_generator
   (Qdrant)                        (schedule → LLM enrich → INDB)
```

| Layer | Role |
|-------|------|
| **Frontend** (`frontend/`) | Chat, Plan tabs, Progress; holds shared state in `App.jsx` |
| **Routers** | Thin HTTP: conversation, plan, workout, exercises |
| **Conversation agent** | ReAct tools: Q&A, profile, plan, save, reminders |
| **Services** | RAG, semantic NLU, exercise selection, plan generator |
| **MCP tools** | save_plan, calories, log workout, progress, email |
| **Stores** | Qdrant (vectors), SQLite (exercises, plans, nutrition, logs) |

LLM: **Groq primary**, **Azure OpenAI failover** when configured.

---

## 3. Technical mental model — how an input is processed layer by layer

Think of every user message as passing through **stacked processors**. Each layer does one job and passes a richer object downward / upward.

### Layer A — Browser / React

**Input:** keystrokes → “how to do pranayama”  
**Does:**

- Local UI state (`messages`, `loading`, `threadId`)
- `POST /api/conversation/message` via Axios (`frontend/src/api/client.js`)
- Base URL: `VITE_API_URL` or `http://localhost:8000`

**Does not:** RAG, LLM, or profile logic.

**Output:** JSON request `{ thread_id, user_message }`

---

### Layer B — FastAPI router

**File:** `backend/app/routers/conversation.py`

**Does:**

- Validate `thread_id` and non-empty message
- Call `process_user_message(...)`
- Map result → Pydantic `ConversationResponse`

**Does not:** decide yoga vs gym vs plan.

**Output:** HTTP response with `message`, `exercises`, `guideline_images`, `profile`, `plan`, …

---

### Layer C — Conversation turn orchestration

**File:** `backend/app/conversation/agent.py` → `process_user_message`

**Order of operations:**

1. Set `_current_thread` (ContextVar) so tools know the session.
2. Clear this turn’s exercise GIFs / guideline images from session store.
3. **Semantic auto-ingest** (embeddings; optional LLM extract if env on).
4. Invoke LangGraph ReAct agent with `HumanMessage` + `thread_id` checkpointer config.
5. On Groq rate-limit/failover errors → retry with Azure agent.
6. Take last `AIMessage` as user-facing text.
7. Attach `exercises` / `guideline_images` from session store (side effects of tools).

**Mental model:** this layer is the **turn controller**. Tools do work; this layer packs the API response.

---

### Layer D — LangGraph ReAct agent

**Built with:** `create_react_agent(llm, TOOLS, checkpointer=MemorySaver(), prompt=SYSTEM_PROMPT)`

**Internal loop:**

```text
┌──────────────┐
│  agent node  │  ← LLM sees system prompt + history + latest user msg
└──────┬───────┘
       │
  tool_calls?
    /        \
  yes         no → END (final reply)
   │
   ▼
┌────────────┐
│ tools node │  ← runs Python @tool functions
└─────┬──────┘
      │
      └── back to agent (LLM sees ToolMessage observations)
```

**Important:**

- One HTTP request can contain **multiple LLM invocations** inside this loop.
- Calling a tool and using its result **cannot** be the same LLM call (result does not exist yet).
- Multiple tools can be requested **in parallel** in one LLM response; after they run, another LLM call still writes the final answer.

**Tools available:**

| Tool | Job |
|------|-----|
| `answer_fitness_question` | RAG Q&A + yoga photos or gym GIFs |
| `update_profile` | Write validated profile slots |
| `get_profile_status` | What’s missing before plan |
| `generate_plan` | Build plan (gated by `is_safe_to_plan`) |
| `save_generated_plan` | Persist + email |
| `send_reminder_email` | Reminder |
| `log_completed_workout` | Log completion |
| `get_progress` | Streak / stats |

---

### Layer E — Tool internals (example: Q&A)

**File:** `answer_fitness_question` in `agent.py`

**Pipeline inside the tool (no chat LLM here):**

1. Resolve `intent` (`info` | `exercise_qa` | `plan`) and `media` (`none` | `yoga_protocol` | `gym_catalog`).
2. Early exits: plan redirect; broad “yoga AND gym” clarify.
3. Augment query with soft age/gender/health context from profile.
4. Expand queries (yoga CLIP variants, age-band protocol titles).
5. **Text RAG:** `retrieve_multi_query` → MiniLM → Qdrant `fitness_guidelines` → rerank.
6. **Image RAG (yoga only):** same-page images from chunks, else hybrid CLIP retrieval.
7. Optionally pull more text from matched image pages.
8. **Gym path:** semantic exercise retrieval (not for yoga_protocol).
9. Store media in session; return a **briefing string** for the ReAct LLM (passages + instructions), not the final polished UI copy.

**Mental model:** tool = **grounded evidence fetcher + policy**. ReAct LLM #2 turns that evidence into natural language.

---

### Layer F — Vector / data services

| Service | Input | Output |
|---------|-------|--------|
| MiniLM embedder | text | dense vector |
| CLIP text encoder | query variants | image-space vector |
| Qdrant | vector + filters | top-k chunks / images |
| SQLite exercises | SQL / semantic hits | catalog rows + gif URLs |
| INDB nutrition | meal names / targets | real macros |

---

### Layer G — Response assembly → UI

1. Agent returns final text.
2. Router includes `guideline_images` / `exercises`.
3. React renders text + `GuidelineImagesStrip` / exercise cards.
4. Media URLs like `/static/...` are prefixed with API base URL.

---

## 4. Deep dive: “how to do pranayama”

This is the canonical **yoga technique Q&A** path.

### Step-by-step

| Step | Where | What happens |
|------|-------|----------------|
| 0 | React | Session already has `thread_id` from `/start` |
| 1 | `ChatPage.handleSend` | User bubble; `POST .../message` |
| 2 | FastAPI | Validate → `process_user_message` |
| 3 | Prep | Clear media; optional embedding ingest; set ContextVar |
| 4 | **LLM #1** | ReAct routes → `answer_fitness_question(intent=exercise_qa, media=yoga_protocol)` |
| 5a | Tool | Confirm not plan / not broad mixed ask |
| 5b | Tool | Query variants: original + `"pranayama breathing yoga demonstration photo"` |
| 5c | Text RAG | MiniLM search Tier-1/2 guidelines; rerank |
| 5d | Image RAG | Prefer images on matched text pages; else CLIP hybrid on SRC009/SRC003 |
| 5e | Tool | Fetch technique text from image pages; **skip gym GIFs** |
| 5f | Tool | Return briefing: passages + photo captions + “teach from these” |
| 6 | **LLM #2** | Writes step-by-step answer with citations; ties to photos |
| 7 | Assemble | `message` + `guideline_images` (exercises empty) |
| 8 | React | Text bubble + protocol photo strip |

### What does NOT run

- `generate_plan` / `plan_generator`
- Gym catalog semantic search
- Plan `StateGraph` (`PLAN_AGENTIC`)
- Profile LLM extract (default off)

### Typical cost for this turn

- **LLM calls:** ~2  
- **Embeddings:** MiniLM (text) + CLIP (images) — not chat LLM

### Why yoga_protocol vs gym_catalog

- Yoga booklet demos live in PDF image collection (`guideline_images`, SRC009 Common Yoga Protocol, SRC003 Fit India).
- Gym moves live in exercise catalog (GIFs via `exercise_semantic` / SQLite).
- Mixing both on one vague ask is explicitly avoided (clarify A/B/C).

---

## 5. Other important flows

### A. Nutrition fact (“how much water?”)

```text
LLM #1 → answer_fitness_question(intent=info, media=none)
→ text RAG only → LLM #2 cites guidelines → no photos/GIFs
```

### B. Gym form (“how do I squat?”)

```text
LLM #1 → answer_fitness_question(intent=exercise_qa, media=gym_catalog)
→ text RAG + retrieve_exercise_semantic → LLM #2
→ UI shows GIF cards
```

### C. Plan generation

```text
User wants a plan
→ update_profile / get_profile_status (several turns, each with ReAct LLM loops)
→ generate_plan tool
   → blocked if !profile.is_safe_to_plan()
   → generate_plan_agentic(...)
        PLAN_AGENTIC=0 (default): plan_generator only
        PLAN_AGENTIC=1: optional StateGraph warm-up (draft discarded) then plan_generator
   → plan_generator:
        1. schedule exercises (code / SQL / semantic)
        2. retrieve guidelines
        3. LLM enrichment (diet names, tips, safety)  ← usually +1 LLM
        4. ground diet in INDB + Mifflin calories (code)
→ session stores plan → agent summarizes → UI Plan tab
```

### D. Save / progress

MCP-backed tools: persist plan, email, log workouts, progress stats — mostly SQLite + SMTP, not RAG.

---

## 6. Where every LLM call happens

### Production chat path

| # | Location | When | Count |
|---|----------|------|-------|
| 1 | `create_react_agent` agent node | Every time the model reasons or emits tool calls / final text | **Variable** (usually 1 if no tools; **≥2** if tools used) |
| 2 | Groq→Azure failover | If primary fails with rate-limit-like error | Retries the **turn** (can roughly double that turn’s agent calls) |
| 3 | `semantic_nlu.llm_extract_profile` | Only if `SEMANTIC_LLM_EXTRACT=1` | **+1 per user message** |
| 4 | `plan_generator.synthesize_plan` | When `generate_plan` runs successfully into enrichment | **+1** |
| 5 | `plan_agent.agent_node` | Only if `PLAN_AGENTIC=1` | **+1 to +6** (cap), then draft discarded |

### Defaults (important)

| Env var | Default | Effect |
|---------|---------|--------|
| `SEMANTIC_LLM_EXTRACT` | `0` | No extract LLM each turn |
| `PLAN_AGENTIC` | `0` | No plan StateGraph LLM loop |
| `LLM_PROVIDER` | `groq` | Primary chat/enrichment provider |

### Why tool turns need ≥2 LLM calls

```text
Call 1: model requests tool(s)
        → tools execute outside the model
Call 2: model reads ToolMessage(s) → writes user answer
```

You **can** batch multiple tool calls in call 1. You **cannot** consume their results in that same call.

### Offline / eval LLM calls (not production chat)

- `evals/run_rag_eval.py` generation stage (if not `--skip-generation`)
- Any ad-hoc scripts using `get_llm`

---

## 7. What is NOT an LLM call

These are easy to confuse with “AI calls”:

| Mechanism | Model / tech | Used for |
|-----------|--------------|----------|
| MiniLM (`sentence-transformers`) | Embedding model | Text guideline + caption similarity |
| CLIP | Multimodal embedding | Yoga/protocol demo image retrieval |
| Qdrant search | Vector DB | Nearest neighbors |
| SQLAlchemy / SQLite | DB | Exercises, plans, nutrition, logs |
| Semantic NLU prototypes | Embedding similarity | Intent / plan_mode / body parts (no LLM by default) |
| Mifflin BMR / filters | Pure Python | Calories, injury exclusions |

---

## 8. Safety, grounding, and autonomy boundaries

This is a core interview talking point.

| Concern | Who decides | How enforced |
|---------|-------------|--------------|
| Which tool this turn? | LLM (ReAct) + system prompt | Soft |
| Can we generate a plan? | Code | `Profile.is_safe_to_plan()` — tool returns `BLOCKED` |
| Invent weekly plan in chat prose? | Forbidden by prompt + architecture | Plan only via `generate_plan` |
| Hallucinated meal macros? | Code grounding | INDB after LLM meal *names* |
| Injury-contraindicated demos | Code filters | `injury_excluded_body_parts` on gym search |
| Yoga vs gym media mix | Prompt + tool policy | `yoga_protocol` skips gym GIFs |
| Trust of guideline text | Filters | Tier 1 / Tier 2 in retrieval |
| Final plan structure | Mostly deterministic | `plan_generator` schedule + grounding |

**Design slogan:** *Autonomy in routing; safety and structure in code.*

---

## 9. Data pipelines (offline)

Stop the API before writing local Qdrant (single-writer lock).

```text
Exercise catalog                         Guidelines + images
─────────────────                        ───────────────────
fetch_external_sources.py                chunk_all_sources.py
        │                                        │
        ▼                                        ▼
build_exercise_corpus.py                 extract_pdf_images.py  → static/guideline_images/
        │                                        │
        ▼                                        ▼
load_db.py  → SQLite exercises           embed_and_store.py     → fitness_guidelines
        │                                embed_images_clip.py   → guideline_images
        ▼
embed_exercises.py → exercise_semantic

Optional: load_nutrition_db → nutrition_items (INDB)
```

---

## 10. Evaluation harness

**Folder:** `evals/`

### Text guideline RAG — `run_rag_eval.py`

- Dataset: `datasets/guideline_rag.jsonl` (query + `must_contain_any` keywords)
- Calls real `retrieve_multi_query`
- Metrics: Hit@k, Precision@k, MRR, mean similarity
- Optional generation: relevance, groundedness (lexical overlap), citation rate
- Labels are **keyword proxies**, not gold chunk IDs (cheaper, noisier)

### Multimodal image RAG — `run_multimodal_rag_eval.py`

- Gold: `(source_id, page_number)` for yoga queries (e.g. pranayama → SRC009 p.36/37)
- Calls `retrieve_guideline_images` with `score_threshold=0.0` to inspect ranking
- Helps tune `IMAGE_SCORE_THRESHOLD` in `rag_retrieval.py`

### Other

- `run_exercise_eval.py` — exercise retrieval quality
- `run_agent_eval.py` — slot / intent behaviour
- `run_all_evals.py` — suite runner

---

## 11. Tools & frameworks — why chosen, vs alternatives

### Frontend

| Choice | Why here | Alternatives | Tradeoff |
|--------|----------|--------------|----------|
| **React 19** | Component UI, ecosystem, team familiarity | Vue, Svelte, plain JS | React is heavier than Svelte; fine for this app size |
| **Vite** | Fast dev server, simple React plugin | CRA (deprecated), Next.js | No SSR needed → Vite SPA is enough |
| **Tailwind** | Rapid mobile-shell styling | CSS modules, MUI | Utility classes can get noisy |
| **Axios** | Simple JSON client | `fetch` | Slightly more deps; either fine |
| **No React Router** | 3 tabs via `useState` | react-router | Simpler; chat kept mounted with CSS `hidden` |

### Backend API

| Choice | Why here | Alternatives | Tradeoff |
|--------|----------|--------------|----------|
| **FastAPI** | Typed APIs, auto `/docs`, async-friendly | Flask, Django REST | FastAPI wins for LLM/tool APIs |
| **Uvicorn** | ASGI server for FastAPI | Hypercorn, Gunicorn+uvicorn workers | Standard pairing |
| **Pydantic** | Request/response validation | marshmallow, manual | Native to FastAPI |

### Agent orchestration

| Choice | Why here | Alternatives | Tradeoff |
|--------|----------|--------------|----------|
| **LangGraph `create_react_agent`** | Tool loop + checkpointer out of the box | Raw LangChain AgentExecutor, custom while-loop, OpenAI Assistants API | Prebuilt ReAct is fast to ship; less control than hand-rolled graph |
| **Custom `StateGraph` in `plan_agent.py`** | Explicit agent↔tools↔extract for plan warm-up | Always use ReAct / never agentic plan | Currently optional; draft discarded → honesty about determinism |
| **LangChain tools (`@tool`)** | Shared schema for LLM tool-calling | OpenAI function schema hand-written | Couples you to LC ecosystem |
| **MemorySaver** | In-process thread memory | Redis/Postgres checkpointer | Dev-friendly; not multi-process durable |

### LLMs

| Choice | Why here | Alternatives | Tradeoff |
|--------|----------|--------------|----------|
| **Groq (Llama 3.3 70B)** | Fast inference, good tool-calling, cost | OpenAI, Anthropic, local Ollama | Rate limits → Azure failover |
| **Azure OpenAI** | Enterprise failover | Second Groq key | Needs Azure config |
| **Not using LLM for retrieval** | Cheaper, grounded | LLM-only answers | Requires good RAG |

### Retrieval & multimodal

| Choice | Why here | Alternatives | Tradeoff |
|--------|----------|--------------|----------|
| **Qdrant (local)** | Simple local vector DB, filters | Pinecone, Weaviate, pgvector, Chroma | Local lock issues with concurrent writers |
| **sentence-transformers MiniLM** | Strong text retrieval, local, free | OpenAI embeddings, larger SBERT | Smaller model = slightly weaker recall |
| **CLIP for images** | Align text↔yoga photos | Caption-only search, GPT-4V every query | CLIP needs good captions + hybrid rerank; VLM every query is expensive |
| **Hybrid image rank** (text anchors + CLIP + lexical) | Fixes OCR/name gaps (tadasana vs tree) | Pure CLIP | More code; much more reliable demos |
| **PyMuPDF** | Extract PDF images/text | pdf2image, pdfplumber | Good for this pipeline |

### Data & persistence

| Choice | Why here | Alternatives | Tradeoff |
|--------|----------|--------------|----------|
| **SQLite** | Zero ops for exercises/plans/nutrition | Postgres | Fine for single-node demo; not heavy multi-tenant |
| **SQLAlchemy** | ORM for models | raw SQL, Prisma | Standard Python choice |
| **INDB grounding** | Real Indian food macros | LLM-invented nutrition | Correctness > fluency |

### Protocol / tooling surface

| Choice | Why here | Alternatives | Tradeoff |
|--------|----------|--------------|----------|
| **MCP server tools** | Shared action surface (save, email, progress) | Only LangChain tools | Extra abstraction; useful if other MCP clients connect |

### Evals

| Choice | Why here | Alternatives | Tradeoff |
|--------|----------|--------------|----------|
| **Custom Python evals + keyword labels** | Cheap, runnable offline | Ragas, TruLens, human eval only | Keywords ≠ true IR relevance |
| **Page-level image gold** | Concrete multimodal metric | CLIP score alone | Small labeled set; high signal for tuning |

---

## 12. Interview practice — questions an AI engineer interviewer might ask

Use these for self-study. After each question, try answering out loud from this doc, then check the code.

### A. System design & architecture

1. Walk me through what happens when a user sends “how to do pranayama.” Where is latency spent?
2. Why is the conversation agent a ReAct loop instead of a finite-state intake machine?
3. Why isn’t the final workout plan produced purely by the LLM?
4. How do you separate **autonomy** (routing) from **safety** (hard gates)? Give code-level examples.
5. Chat state is lifted to `App.jsx` and Chat is CSS-hidden. Why not unmount or use a router?
6. What breaks if two processes open local Qdrant at once? How do evals mitigate that?
7. Compare retrieve-then-generate (1 LLM call) vs ReAct tool-calling (≥2). When would you switch?

### B. RAG & multimodal

8. Explain text RAG in this repo: chunking → embedding → store → retrieve → rerank → cite.
9. Why hybrid CLIP retrieval instead of pure image similarity?
10. How do `_clip_query_variants` help for “pranayama”?
11. What’s the difference between `fitness_guidelines` and `guideline_images` collections?
12. How do you prevent gym GIFs from showing on a yoga technique turn?
13. Your guideline eval uses `must_contain_any` keywords. What are the failure modes of that metric vs gold `chunk_id`s?
14. How would you improve recall for a rare asana name that OCR mangled in the PDF?
15. Where does `IMAGE_SCORE_THRESHOLD` matter, and how do you tune it?

### C. Agents & LangGraph

16. What does `create_react_agent` give you that a raw `while` loop would require you to write?
17. Why can’t tool calling and answering from tool results be one LLM call?
18. What does `MemorySaver` + `thread_id` actually persist? What happens on process restart?
19. Explain `PLAN_AGENTIC=1`: what runs, what is discarded, and why keep that design?
20. How does the system prompt encode intent/media routing, and what happens if the model omits those args?
21. How would you add a new tool (e.g. “find nearby parks”) without breaking safety?

### D. LLMs, cost, reliability

22. Map all LLM call sites in a plan-generation conversation. Estimate calls for a 6-turn intake + generate.
23. How does Groq→Azure failover work? Which errors trigger it?
24. Tradeoffs of Llama-on-Groq vs GPT-4o for tool-calling and JSON enrichment?
25. How would you reduce LLM cost 30% without killing answer quality?
26. What’s the risk of setting `SEMANTIC_LLM_EXTRACT=1` on every message?

### E. Planning, nutrition, safety

27. Describe `plan_generator` stages. Which stages are allowed to fail soft?
28. Why ground diet in INDB after LLM enrichment?
29. How are injury / health flags applied to exercise demos vs weekly plans?
30. What is `plan_mode` (`full` / `diet_only` / `yoga_only`) and how does it change retrieval and UI?
31. How do you handle missing height/weight for calorie targets?

### F. Frontend ↔ backend contract

32. Which response fields drive chat media UI? What if the LLM claims a photo exists but `guideline_images` is empty?
33. Why clear exercises/images at the start of each turn?
34. How would you stream tokens to the UI with the current ReAct+tools design?

### G. MLOps / quality

35. Design an offline eval for yoga image retrieval. What is your gold label?
36. How do you detect regressions when someone changes the system prompt?
37. What’s your strategy for prompt vs code changes when the agent mis-routes “make me a plan” into random tips?
38. How would you add observability (traces) for tool calls and retrieval scores in production?

### H. “Senior” challenge questions

39. The agent sometimes invents steps not in retrieved passages. Where do you fix it: prompt, tool return format, post-hoc groundedness check, or retrieval?
40. Propose an architecture that keeps agent flexibility but guarantees **exactly one** LLM call for FAQ traffic.
41. How would you multi-tenant this (many users, durable memory, shared Qdrant)?
42. CLIP retrieved the wrong asana page. Walk me through your debugging checklist.
43. Defend keeping keyword-based RAG metrics in CI despite their noise.

### Suggested strong answers (cheat themes)

- **ReAct vs React:** different concepts.
- **Grounding:** tool returns evidence; LLM narrates; UI media comes from session store, not model imagination.
- **Plan determinism:** schedule + filters + INDB beat freeform JSON.
- **Multimodal:** hybrid > pure CLIP for PDF demos with OCR noise.
- **Cost:** embeddings cheap; agent loop expensive; defaults keep extract/agentic plan off.

---

## 13. Suggested study order

1. `frontend/src/App.jsx` + `api/client.js` — UI ↔ API contract  
2. `backend/main.py` + `routers/conversation.py` — entry  
3. `conversation/agent.py` — ReAct, tools, `process_user_message`  
4. `services/rag_retrieval.py` — text + CLIP hybrid  
5. `services/semantic_nlu.py` — embedding NLU vs optional LLM extract  
6. `services/plan_generator.py` + `plan_agent.py` — plan path  
7. `evals/run_rag_eval.py` + `run_multimodal_rag_eval.py` — how quality is measured  
8. Re-read §12 and answer 10 questions aloud  

---

## 14. Quick file map

| Path | Purpose |
|------|---------|
| `frontend/src/App.jsx` | Tab shell + shared plan/chat state |
| `frontend/src/pages/ChatPage.jsx` | Chat UI + media strips |
| `frontend/src/api/client.js` | Axios API helpers |
| `backend/main.py` | FastAPI app, CORS, static |
| `backend/app/routers/conversation.py` | `/api/conversation/*` |
| `backend/app/conversation/agent.py` | LangGraph ReAct + tools |
| `backend/app/conversation/profile_store.py` | Profile / session store |
| `backend/app/services/rag_retrieval.py` | Guideline text + image RAG |
| `backend/app/services/semantic_nlu.py` | Embedding intent / slots |
| `backend/app/services/plan_generator.py` | Deterministic plan pipeline |
| `backend/app/services/plan_agent.py` | Optional plan StateGraph |
| `backend/app/mcp_server/server.py` | Save / email / progress tools |
| `backend/app/llm.py` | Groq / Azure clients + failover |
| `backend/rag/*` | Chunk, embed, CLIP image pipeline |
| `evals/*` | Retrieval / agent evals |
| `README.md` | Product overview + setup |

---

## Appendix — ReAct vs React (one more time)

| Term | Meaning in this project |
|------|-------------------------|
| **React** | Frontend library rendering Chat / Plan / Progress |
| **ReAct** | Agent pattern: Reason → Act(tool) → Observe → Reason → … |
| **`create_react_agent`** | LangGraph helper that implements ReAct, not related to the UI library |

---

## 15. Live behind-the-scenes trace (real run)

You can reprint this anytime:

```bash
cd backend
source ../env/bin/activate
python ../evals/trace_rag_query.py "how to do pranayama"
python ../evals/trace_rag_query.py "how to do tadasana"
python ../evals/trace_rag_query.py "water intake daily" --no-images
```

Script: `evals/trace_rag_query.py`  
It walks the **real** Qdrant index + embedding models (no chat LLM).

### Snapshot from a real run — query: `how to do pranayama`

#### Vector DB (Qdrant) shape

| Collection | Points | Vector dim | Distance | Contents |
|------------|--------|------------|----------|----------|
| `fitness_guidelines` | **3437** | **384** | Cosine | text chunks (MiniLM) |
| `guideline_images` | **83** | **512** | Cosine | demo photos (CLIP) |
| `exercise_semantic` | (present) | — | — | gym exercise vectors |

A stored point = **id + payload (metadata/text) + vector (list of floats)**.

Example text vector sample: `[0.0025, 0.0341, -0.0114, …, -0.0354]` (384 numbers).  
Example image vector sample: `[-0.0302, 0.0036, -0.0333, …]` (512 numbers).

Sample text payload keys: `chunk_id`, `source_id`, `source_name`, `page_number`, `text`, `trust_tier`, `content_type`, …  
Sample image payload keys: `image_url`, `caption`, `source_id`, `page_number`, …

#### Embedding models

| Model | Dim | Used for |
|-------|-----|----------|
| `all-MiniLM-L6-v2` | 384 | query + guideline text + caption re-rank |
| `clip-ViT-B-32` | 512 | query text tower ↔ image vectors |

**Rule:** never mix MiniLM vectors with CLIP vectors (different spaces).

Query variants generated for yoga:

1. `how to do pranayama`
2. `pranayama breathing yoga demonstration photo`

MiniLM query vector: dim 384, L2 norm ≈ 1.0 (normalized).  
CLIP text query vector: dim 512.

#### Text chunks actually retrieved (after multi-query + rerank)

| Rank | chunk_id | Source | Page | Score | Content gist |
|------|----------|--------|------|-------|----------------|
| 1 | `SRC009_p037` | Common Yoga Protocol | **37** | ~0.70 | PRĀNĀYĀMA technique (alternate nostril) |
| 2 | `SRC009_p040` | CYP | 40 | ~0.59 | meditation / breath focus |
| 3 | `SRC009_p038` | CYP | 38 | ~0.58 | breathing + benefits |
| 4 | `SRC009_p035` | CYP | 35 | ~0.58 | relaxation |
| 5 | `SRC009_p018` | CYP | 18 | ~0.57 | neck bends (weaker match) |

Top chunk text includes real steps: sit in meditative posture, spine straight, eyes closed, Jnāna mudra, Nāsāgra mudra, alternate-nostril breathing…

**Process:** embed each query → cosine top-k in `fitness_guidelines` → filter Tier 1/2 → dedupe by `chunk_id` → sort by score → `rerank_guideline_chunks`.

#### Images actually matched

Preferred path = **same-page images** from text hits:

| File | Page | Role |
|------|------|------|
| `/static/guideline_images/SRC009/p037_0.png` | 37 | PRĀNĀYĀMA demo |
| `/static/guideline_images/SRC009/p038_0.png` | 38 | breathing / benefits page |

CLIP hybrid also ranked p.37/p.38 highly (scores ~0.69–0.70 ≥ production threshold `0.30`).

**Hybrid image steps:**

1. MiniLM text anchors → likely PDF pages  
2. CLIP text query vs image vectors  
3. Caption MiniLM + lexical boost re-rank  
4. Keep scores ≥ `IMAGE_SCORE_THRESHOLD` (0.30)

#### What the ReAct LLM sees next (not final UI prose)

`answer_fitness_question` returns a **briefing**: intent/media tags + guideline passages + matched photo captions.  
Then **LLM call #2** narrates the user answer; React renders text + `GuidelineImagesStrip`.

```text
"how to do pranayama"
        │
        ├─ MiniLM → 384-d → cosine vs 3437 text vectors → chunks (p.37…)
        │
        └─ CLIP text → 512-d → cosine vs 83 image vectors → PNGs (p.37, p.38)
              (or same-page images from text hits — used first)
```

---

## 16. Chunking strategy used in this project

**Script:** `backend/rag/chunk_all_sources.py`  
**Output:** `data/.../chunks/all_chunks.json` → embedded by `embed_and_store.py` into Qdrant `fitness_guidelines`.

This is **not** LangChain’s RecursiveCharacterTextSplitter as a library call. It is a **custom, format-aware chunker** built for guideline PDFs + exercise JSON.

### Size constants (character-based, not tokens)

| Constant | Value | Meaning |
|----------|-------|---------|
| `CHUNK_TARGET` | **800** chars | Target size for sliding-window pieces |
| `CHUNK_OVERLAP` | **150** chars | Sentence overlap between consecutive pieces |
| `MIN_CHUNK` | **80** chars | Drop tiny noise (headers, page numbers) |
| `MAX_CHUNK` | **1200** chars | Hard ceiling before forced sub-split |
| `PAGE_KEEP_WHOLE_CHARS` | **2800** | Booklet pages ≤ this stay one chunk |
| `TABLE_LAYOUT_THRESHOLD` | **0.55** | Layout score to treat page as a table |

### Sliding window (prose sub-splitter)

Function: `sliding_window(text, size=800, overlap=150)`

1. Split text on **sentence boundaries** (`[.!?]`)
2. Accumulate sentences until buffer exceeds ~800 chars
3. Emit chunk; keep ~150 chars of leading sentences as overlap
4. Continue

So it is a **sentence-aware sliding window**, not a blind character cut mid-word.

### PDF strategy selection

```text
analyse PDF (pages, text_ratio)
        │
        ├─ text_ratio < 0.4  →  pdf_image      (mostly scanned/image; page-level OCR/text)
        ├─ pages ≤ 80        →  pdf_structured (booklets: CYP, Fit India, …)
        └─ else              →  pdf_prose      (long ICMR/WHO/FSSAI prose)
```

| Strategy | How pages become chunks | Why |
|----------|-------------------------|-----|
| **`pdf_structured`** | **One chunk per page** by default; keep tables whole; only long prose pages sub-split with heading stamped on each piece | Preserves correct `page_number` so yoga photos (p.37) link to the right text |
| **`pdf_image`** | Page-level; keep whole | Sparse text pages shouldn’t be shredded |
| **`pdf_prose`** | Accumulate pages between detected **section headings**, then `sliding_window` | Long manuals → section-aware ~800-char chunks |

**Important nuance:** older code used a prose path that mashed later CYP pages into the wrong `page_number`. Current design prefers **page-level chunking for booklets ≤80 pages** so multimodal (text page ↔ image page) stays aligned. That is why pranayama retrieves `SRC009_p037` and image `/static/.../p037_0.png` together.

### Table / protocol pages

`page_looks_like_table()` uses **layout geometry** (multi-column rows, aligned x-positions, short cells, horizontal rules) — not keyword lists.  
If it looks like a protocol table → **keep the whole page as one chunk** so titles like “50-65 Years” stay with the body.

### Chunk IDs

| Kind | Pattern | Example |
|------|---------|---------|
| PDF page | `{source_id}_p{page:03d}` | `SRC009_p037` |
| PDF sub-split | `{source_id}_p{page:03d}_s{idx:02d}` | `SRC009_p012_s01` |
| JSON exercise | `{source_id}_ex{idx:04d}` | `SRC011_ex0042` |

### Metadata attached to every chunk

`source_id`, `source_name`, `trust_tier`, `country`, `page_number`, `section_title`, `content_type` (nutrition/exercise/…), `doc_type`, optional `image_urls` (same-page stamps from `pdf_images_map.json`; CLIP retrieval is still authoritative for chat demos).

### JSON exercises

`chunk_json`: **one chunk per exercise record** (name + instructions + muscles + equipment as a text blob). Oversized records go through `safe_chunks` / sliding window.

### What is NOT used

- Fixed token windows from tiktoken  
- Pure recursive character split without sentence awareness (except as size safety net)  
- LLM-based semantic chunking  
- One giant document embedding  

### Chunk → embed → retrieve (full offline→online path)

```text
PDF/JSON
  → chunk_all_sources.py   (strategies above)
  → all_chunks.json
  → embed_and_store.py     (MiniLM 384-d → Qdrant fitness_guidelines)
  → at query time: embed query → cosine → chunks like SRC009_p037
  → (optional) extract_pdf_images + embed_images_clip → guideline_images
```

### Interview-ready one-liner

> “We use a **custom format-aware chunker**: booklet PDFs are mostly **page-level** (so page numbers align with CLIP demo photos), long prose is **section-aware sentence sliding windows (~800 chars, 150 overlap)**, tables stay **whole**, and exercises are **one record = one chunk**.”

---

*Generated for learning. When code drifts, trust the source files listed above over this document. Re-run `evals/trace_rag_query.py` for fresh live numbers.*
