# Adaptive Fitness Planner — Module Outline

India-first personalised fitness system: conversational intake → dual RAG (guidelines + exercises) → deterministic (optionally agentic) plan generation → SQLite persistence → React UI.

---

## 1. Product purpose

- Chat agent collects a validated profile and answers fitness/nutrition questions grounded in Indian guidelines.
- Generates a 7-day workout + diet plan (or diet-only) with media, safety notes, and citations.
- Saves plans, logs workouts, shows progress, optional email.

## 2. System map

```
User (React)
    │  /api/conversation/*  /api/plan/*  /api/workout/*  /api/exercises/*
    ▼
FastAPI (backend/main.py)
    ├── Conversation agent (LangGraph ReAct + tools)
    ├── Plan pipeline (generate_plan → filters + schedule + enrich + INDB)
    ├── Dual RAG (Qdrant guidelines + exercise_semantic)
    ├── SQLite (exercises, users, plans, workouts, nutrition_items)
    └── MCP tools (save, calories, log, reminder, progress)
```

## 3. Outline of sections (full doc)

1. **Overview & goals**
2. **Repository layout**
3. **End-to-end user journeys**
4. **API surface**
5. **Conversation agent & tools**
6. **Profile model, validation, safety gates**
7. **Semantic NLU & turn intents**
8. **SQL / RAG filter builders**
9. **Guideline RAG**
10. **Exercise retrieval (SQL + semantic)**
11. **Plan generation (deterministic)**
12. **Plan agent (agentic opt-in)**
13. **Nutrition / BMR / INDB grounding**
14. **MCP server & persistence**
15. **LLM providers**
16. **Frontend architecture**
17. **Data & ingestion pipelines**
18. **Evaluation harness**
19. **Configuration & environment**
20. **Safety / design contracts**
21. **Known limitations & ops notes**
22. **File index (`main_scripts`)**

## 4. Core runtime sequence

1. `POST /api/conversation/start` → `thread_id` + greeting  
2. User messages → agent tools (`answer_fitness_question`, `update_profile`, …)  
3. Profile complete + health confirmed → `generate_plan`  
4. Deterministic `plan_generator.generate_plan` (always; agentic draft never returned raw)  
5. Optional save / email → Plan & Progress tabs  

## 5. Data stores

| Store | Role |
|-------|------|
| SQLite `fitness.db` | Exercises, users, plans, workout logs, nutrition_items |
| Qdrant `fitness_guidelines` | Chunked PDF / guideline passages |
| Qdrant `exercise_semantic` | One vector per exercise for meaning search |
| In-memory session store | Per-`thread_id` profile, plan, exercise cards |

## 6. Feature flags

| Env | Effect |
|-----|--------|
| `PLAN_AGENTIC=1` | Run LangGraph retrieve/validate loop before one `generate_plan` |
| `SEMANTIC_LLM_EXTRACT=1` | LLM JSON extract for demographics / health in NLU |
| `LLM_PROVIDER` | `groq` (default) / Azure / Ollama |

## 7. Eval suites

- Guideline RAG: Hit@k, P@k, MRR + generation relevance / groundedness / citations  
- Exercise retrieval: name Hit@k, equipment, media, diet-intent  
- Agent slots: age / gender / intent / plan_mode (NLU)  

See `MODULE_DOCUMENTATION.md` for the full depth.
