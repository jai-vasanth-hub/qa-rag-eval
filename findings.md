## Observation 1 - Embedding Model Deprecation
Date: 29-05-2026
Issue: models/embedding-001 is deprecated in Gemini API v1beta
Fix: Updated to models/text-embedding-004
Learning: Always check API model availability before building

## Observation 3 - Deprecated vs Current Google AI Package
❌ google.generativeai — fully deprecated as of 2026
✅ google.genai — current package
✅ langchain-google-genai — handles this internally when updated
Always run pip install -U to keep packages current.

## Observation 4 - Free Tier Rate Limits
Gemini free tier limit: 100 embed requests/minute
484 chunks needs batching strategy
Fix: Process in batches of 50 with 65s delay between batches
Learning: Always design ingestion pipelines with rate limit handling

## Observation 5 - Ingestion Success
Date: 29-05-2026
✅ 78 pages → 484 chunks → ChromaDB
Embedding model: gemini-embedding-001
Batch size: 50 chunks
Wait time: 80s between batches
Total ingestion time: ~13 minutes
Key learning: Free tier needs careful rate limit handling

## Observation 6 - LLM Free Tier Limits
gemini-2.0-flash daily quota exhausted quickly on free tier
Fix: Switch to gemini-1.5-flash or gemini-1.5-flash-8b
Learning: Always have a fallback model strategy
for free tier API usage

## Observation 7 - Available Gemini Models (May 2026)
Best model for free tier chatbot use: gemini-2.5-flash-lite
Avoid: gemini-2.0-flash (daily quota exhausts quickly)
Always run list_models() to check available models
before hardcoding model names in code.

## Observation 8 - First Chatbot Run Results
Date: 30-05-2026
Questions answered correctly: 2/4 (50%)
Questions unanswered despite source existing: 2/4

Failures:
- "Seven testing principles" — sources found on pg 16,17
  but answer not retrieved correctly
- "Test oracle" — sources found but answer not retrieved

Hypothesis: Chunk size 500 too small for multi-point answers
Experiment planned: Increase k from 4 to 8, observe change

This is exactly what RAG evaluation looks like in practice.

## Observation 9 - Custom Eval Framework Built
Date: 05-06-2026
Abandoned DeepEval due to version conflicts with LangChain 1.x
Built custom evaluator using Gemini as judge LLM
Metrics: Faithfulness + Relevancy scored 0.0 to 1.0
Judge model: gemini-3.1-flash-lite
Chatbot model: gemini-3.5-flash
Awaiting quota reset for baseline run

## Experiment 1 — k=4 vs k=8 Results
Date: 30-05-2026

Result: Increasing k from 4 to 8 did NOT improve failed questions
Observation: Page 17 retrieved 3 times for "seven principles"
             — same chunks repeated, not new information

Root cause hypothesis: Chunk size 500 too small
The seven testing principles span multiple pages
Each principle is in a separate chunk
No single chunk has enough context to answer fully

Next experiment: Rebuild vector DB with chunk_size=1000
Expected outcome: Principles consolidated into fewer, richer chunks
Risk: Larger chunks may hurt precision on narrow questions