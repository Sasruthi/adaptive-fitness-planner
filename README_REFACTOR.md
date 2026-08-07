# Adaptive Fitness Planner — Agentic Refactor

## What changed and why

**1. One agent, not two.** `app/conversation/agent.py` replaces
`intake_graph.py`'s regex/keyword state machine. It's a single
`langgraph.prebuilt.create_react_agent` with 8 tools. The LLM decides,
per message, whether to answer a question, ask for more profile info, or
generate a plan — that decision used to be hardcoded stage logic; now
it's genuinely autonomous tool-calling, which is what "agentic" should
mean.

**2. Safety is NOT left to the LLM.** `Profile.is_safe_to_plan()` in
`profile_store.py` hard-blocks `generate_plan` in code unless health
flags are explicitly set (real flags or an explicit `["none"]`). The
agent can decide *when* to ask, never *whether* the check happens.

**3. RAG Q&A is now reachable mid-conversation.** `answer_fitness_question`
is a tool the agent can call any time — this was the main gap: previously
`trigger_background_rag` only silently preloaded chunks for later plan
generation, it never answered a live question.

**4. Thin exercise semantic layer added**, separate from the guideline
collection (`exercise_semantic` vs `fitness_guidelines`). Built from the
already-cleaned SQL `Exercise` table via `rag/embed_exercises.py`, not
the raw JSON — one source of truth. Used only for open Q&A
("what's good for lower back pain"); plan generation still uses SQL
filters (`exercise_retrieval.py`) for exact filtering, which was already
correct.

**5. Fixed a real bug you were about to ship with:** `embed_and_store.py`
had zero filtering and was embedding ~800+ raw exercise JSON records
into the *guideline* collection, despite the docstring claiming
otherwise. Now filtered by `doc_type`.

**6. Fixed `generate_plan_agentic` signature mismatch** — it only
accepted `profile` but was already being called with 3 args in
`routers/plan.py`; would have 500'd on first real request.

## Files in this package

```
backend/
  app/
    conversation/
      agent.py            NEW — the unified agent (replaces intake_graph.py)
      profile_store.py    NEW — pydantic Profile + thread-scoped session store
    services/
      exercise_rag.py     NEW — thin semantic layer over exercises
      plan_agent.py        FIXED — signature bug patched
    routers/
      conversation.py      REWRITTEN — now calls agent.py
    schemas/
      models.py             PATCHED — ConversationResponse gained `plan`
  rag/
    embed_and_store.py      FIXED — excludes json_exercise chunks now
    embed_exercises.py    NEW — builds exercise_semantic collection
```

`intake_graph.py` and the old `state.py` extraction helpers are NOT
deleted — `agent.py` still imports `build_sql_filters`/`build_rag_filters`
from `state.py` (those were fine, keep them). You can delete
`intake_graph.py` once you've confirmed the new agent covers your test
cases; I left it in place for you to diff against / roll back to if
needed.

## Migration steps

1. Drop these files into your existing project at the paths shown above.
2. `pip install langgraph langchain-core langchain-groq` (you likely
   already have most of these from `plan_agent.py`).
3. Re-run your existing ingestion in this order:
   ```
   python rag/chunk_all_sources.py      # unchanged
   python rag/embed_and_store.py        # now excludes exercise JSON
   python scripts/load_db.py            # unchanged — exercises into SQL
   python rag/embed_exercises.py        # NEW — exercise_semantic collection
   ```
4. Test conversationally: ask a pure question first ("what's a good
   protein source for vegetarians"), confirm it answers via RAG without
   trying to extract profile slots from it. Then start giving profile
   info and confirm `update_profile`/`get_profile_status` are being
   called (check server logs — LangGraph will show tool calls).

## Not yet done (next pass, as agreed)

- **Deployment**: Qdrant is still local-disk (`QdrantClient(path=...)`),
  SQLite is still file-based — both will lose data on most free-tier
  redeploys. Migrate to Qdrant Cloud free tier + Supabase Postgres free
  tier before hosting.
- **Scheduled reminders**: `send_reminder_email` is still a manual tool
  call (agent calls it when the user asks, or you call the endpoint
  directly) — there's still no autonomous cron. A GitHub Actions
  scheduled workflow hitting `/api/workout/reminder` is the pragmatic
  free option, planned for the deployment pass.
- **`SessionStore` is in-memory** — fine for one process during dev/demo,
  will not survive a restart or scale past one worker. Flagged in the
  docstring; swap for Redis/Postgres before real users.
