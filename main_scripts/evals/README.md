# Evaluation harness

Run from an activated venv with backend deps installed (Qdrant index + optional LLM keys):

```bash
cd backend
source ../env/bin/activate

python ../evals/run_all_evals.py
python ../evals/run_rag_eval.py              # retrieval + generation
python ../evals/run_rag_eval.py --skip-generation
python ../evals/run_exercise_eval.py
python ../evals/run_agent_eval.py
```

Generation / profile LLM extract need `GROQ_API_KEY` (or Azure). Set `SEMANTIC_LLM_EXTRACT=0` to skip LLM profile extract during evals.

If uvicorn holds the local Qdrant lock, evals auto-copy the index to a temp dir.

## What is measured

### Retrieval (guideline RAG)
Hit@k, Precision@k, MRR, mean similarity score

### Generation (RAG answer)
Answer relevance, Groundedness, Citation rate

### Exercise retrieval
Name Hit@k, equipment/apparatus compliance, instruction + media coverage, diet-intent accuracy

### Agent slots (semantic NLU)
Age/gender/health ingest, diet vs exercise intent, plan_mode, body parts — via embeddings + LLM extract (not keyword lists)

## Architecture note

Product intent / slot gates live in `backend/app/services/semantic_nlu.py` (embedding prototypes + optional LLM JSON extract). Keyword/regex intent lists were removed from the agent path.
