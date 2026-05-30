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