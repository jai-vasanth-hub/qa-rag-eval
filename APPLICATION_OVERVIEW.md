# QA-RAG-Eval: Application Overview

**Document Generated:** 2026-06-10  
**Last Analysis Timestamp:** 2026-06-10T20:56:52

---

## 1. Application Summary

### Application Name
**QA-RAG-Eval** (Quality Assurance - Retrieval Augmented Generation - Evaluation)

### Business Purpose
Develop and rigorously evaluate a Retrieval Augmented Generation (RAG) system for knowledge-based question answering on ISTQB Foundation software testing syllabus. The system experiments with different document chunking strategies to optimize answer quality and retrieval accuracy.

### Problem Being Solved
1. **Knowledge Retrieval Optimization** — Determine optimal document chunk sizes (500 vs 1000 tokens) for accurate context retrieval
2. **Answer Quality Evaluation** — Implement automated evaluation metrics (faithfulness, relevancy) to measure RAG system quality
3. **Free-Tier API Rate Limiting** — Handle Google Generative AI free tier constraints with batch processing and retry logic
4. **Model Deprecation** — Navigate Gemini API model deprecations and maintain compatibility with current models

### Primary Users/Stakeholders
- **Primary:** Software testing professionals and ISTQB Foundation students seeking knowledge-based answers
- **Secondary:** QA engineers evaluating RAG implementation for internal knowledge systems
- **Internal:** Development team iterating on chunking strategies and model selection

### High-Level Architecture
**Monolithic Local RAG System**
- Offline document ingestion pipeline
- In-memory vector database with persistent storage
- Stateless question-answering interface
- Evaluation suite for quality metrics

### Major Application Components
1. **Data Ingestion Pipeline** (`ingest.py`) — PDF → Chunks → Embeddings → Vector DB
2. **QA Chatbot** (`chatbot.py`) — Question → Retrieval → LLM Response
3. **Evaluation Framework** (`eval_suite.py`) — Question → Answer → Scoring
4. **Knowledge Store** (ChromaDB) — Vector embeddings with persistence

### Deployment Overview
- **Environment:** Local development machine (Windows)
- **Storage:** File-based (ChromaDB SQLite, JSON reports)
- **Execution:** Python command-line scripts
- **API Dependency:** Google Generative AI (cloud-based)
- **Data Source:** ISTQB Foundation PDF (78 pages, 484 chunks at 1000 token size)

---

## 2. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.x | Primary implementation language |
| **Framework** | LangChain | [INFERRED: 1.x] | RAG orchestration, chains, retrievers |
| **Vector DB** | ChromaDB | [NOT FOUND IN CODEBASE] | Persistent vector storage |
| **LLM Provider** | Google Generative AI (Gemini) | Multiple models | Embeddings & LLM inference |
| **Document Loading** | LangChain Community (PyPDFLoader) | [INFERRED] | PDF parsing |
| **Text Processing** | RecursiveCharacterTextSplitter | [INFERRED] | Document chunking with overlap |
| **API Client** | google-generativeai SDK | [NOT FOUND IN CODEBASE] | Gemini API integration |
| **Configuration** | python-dotenv | [INFERRED] | Environment variable management |
| **Data Serialization** | JSON | Built-in | Report generation & storage |
| **Database (Local)** | SQLite | Built-in (via ChromaDB) | Underlying ChromaDB storage |

### Dependencies & Justification

| Dependency | Why It's Used |
|-----------|---------------|
| **LangChain** | Provides high-level abstractions for RAG (RetrievalQA chain, PromptTemplate). Abstracts away low-level LLM integration details. |
| **ChromaDB** | Open-source vector database optimized for semantic search. Lightweight, persistent, no external server required. |
| **Google Generative AI** | Free tier embeddings (100 req/min) and LLM access. Cost-effective for experimentation. Chosen over OpenAI due to free tier availability. |
| **RecursiveCharacterTextSplitter** | Preserves semantic boundaries by splitting on characters/tokens rather than naive fixed-size chunks. Configurable overlap for context preservation. |
| **PyPDFLoader** | Extracts pages and page metadata from PDF. Maintains page references for source attribution. |
| **python-dotenv** | Secure credential management without hardcoding API keys. Follows environment-based configuration pattern. |

---

## 3. Module & Folder Structure

### Repository Structure

```
qa-rag-eval/
├── src/
│   ├── ingest.py          # Data ingestion: PDF → Vector DB
│   ├── chatbot.py          # QA interface with retrieval chain
│   └── eval_suite.py       # Evaluation framework (faithfulness, relevancy)
├── data/
│   └── istqb_foundation.pdf    # Source knowledge document (78 pages)
├── db/
│   └── chroma_db/          # Vector database persistence
│       ├── chroma.sqlite3  # ChromaDB SQLite backend
│       └── [uuid]/         # Embedding data directory
├── reports/
│   ├── eval_chunk_size_1000_20260610_205652.json    # Latest results
│   ├── eval_chunk_size_500_20260609_222759.json     # Previous runs
│   └── [historical reports]
├── findings.md             # Experimental observations & learnings
├── README.md               # [EMPTY]
├── .env                    # API credentials
├── .gitignore              # Excludes venv, __pycache__, .env
└── venv/                   # Python virtual environment
```

### Module Descriptions

| Module | Purpose | Key Classes/Functions | Dependencies |
|--------|---------|----------------------|--------------|
| **ingest.py** | Document ingestion pipeline with rate-limit handling | `load_pdf()`, `split_documents()`, `create_vector_store()` | LangChain, ChromaDB, Google Generative AI |
| **chatbot.py** | Interactive QA system using retrieval chain | `load_vector_store()`, `create_qa_chain()`, `ask_question()` | LangChain, ChromaDB, Google Generative AI |
| **eval_suite.py** | Automated evaluation using LLM-as-judge | `get_answer_and_context()`, `evaluate_faithfulness()`, `evaluate_relevancy()`, `run_evaluation()`, `generate_report()` | LangChain, Google Generative AI, JSON |

---

## 4. Core Business Logic

### RAG Question-Answering Workflow

The system implements a standard RAG (Retrieval Augmented Generation) pipeline:

```
Question Input
    ↓
[Embedding Generation] — Convert question to vector using Gemini embeddings
    ↓
[Semantic Retrieval] — Query ChromaDB for k=8 most similar chunks
    ↓
[Context Assembly] — Combine retrieved chunks into prompt context
    ↓
[LLM Generation] — Pass context + question to Gemini LLM
    ↓
[Answer Output] — Return grounded response with source attribution
```

#### 4.1 Question Answering Logic (chatbot.py)

**Entry Point:** `create_qa_chain()` and `ask_question()`

**Algorithm:**
1. Load ChromaDB with Gemini embeddings (`load_vector_store()`)
2. Create LangChain `RetrievalQA` chain with:
   - Retriever: ChromaDB with k=8 (retrieve 8 most relevant chunks)
   - LLM: `gemini-2.5-flash-lite` with temperature=0.1 (low temperature for consistency)
   - Chain type: "stuff" (concatenate all chunks into single prompt)
3. Custom prompt constrains answers to only provided context
4. Return answer + source documents (page numbers)

**Key Parameters:**
- `model`: `gemini-2.5-flash-lite` (optimized for free tier)
- `temperature`: 0.1 (deterministic, fact-focused)
- `k`: 8 (retrieve 8 chunks for richer context)
- `chunk_overlap`: 50 tokens (preserve inter-chunk continuity)

#### 4.2 Evaluation Logic (eval_suite.py)

**Metrics:** Faithfulness (0.0–1.0) + Relevancy (0.0–1.0)

**Faithfulness Evaluation:**
- Definition: Every claim in the answer must be supported by retrieved context
- Implementation: LLM-as-judge using `gemini-3.1-flash-lite`
- Prompt: Judge LLM evaluates if answer contains hallucinations or out-of-context information
- Output: Score 0.0–1.0 + explanation
- Pass Threshold: ≥ 0.5

**Relevancy Evaluation:**
- Definition: Answer must directly address the question asked
- Implementation: LLM-as-judge evaluates if answer is on-topic
- Prompt: Judge LLM scores if answer addresses query requirements
- Output: Score 0.0–1.0 + explanation
- Pass Threshold: ≥ 0.5

**Test Suite:**
- 8 fixed questions on ISTQB Foundation topics
- Expected answers defined for reference
- Questions span: equivalence partitioning, testing principles, verification/validation, test oracles, boundary value analysis, static/dynamic testing, test planning, regression testing

#### 4.3 Document Ingestion Logic (ingest.py)

**Pipeline:**
1. **PDF Loading** — `PyPDFLoader` extracts 78 pages from ISTQB Foundation PDF
2. **Document Splitting** — `RecursiveCharacterTextSplitter` creates 484 chunks:
   - `chunk_size=1000` tokens
   - `chunk_overlap=50` tokens
3. **Embedding Generation** — `gemini-embedding-001` converts each chunk to vector
4. **Rate-Limit Handling:**
   - Batch size: 50 chunks (respects 100 req/min free tier limit)
   - Inter-batch delay: 80 seconds
   - Retry logic: Up to 3 attempts per batch (80s → 110s → 140s waits)
5. **Persistence** — ChromaDB stores embeddings in SQLite at `db/chroma_db/`

**Key Findings from findings.md:**
- Initial ingestion with `chunk_size=500` → 500+ chunks
- Experiment: Increased k from 4→8 (no improvement on missed questions)
- Root cause: Chunks too small; multi-point concepts (e.g., "seven testing principles") scattered across multiple chunks
- Solution: Increased to `chunk_size=1000` (current configuration)
- Result: 484 consolidated chunks with better topical coherence

### Feature Engineering & Business Rules

- **Chunk Consolidation:** Larger chunks preserve multi-part concepts within single retrieval unit
- **Temperature Control:** Set to 0.1 to reduce hallucinations and encourage fact-based answers
- **Prompt Constraint:** Explicit instruction "Do not make up answers" prevents LLM from inferring beyond context
- **Context Validation:** Faithfulness metric ensures no out-of-context claims

---

## 5. Data Flow

### End-to-End Processing Pipeline

```
ISTQB Foundation PDF (data/istqb_foundation.pdf)
    │ [PDFLoader]
    ↓
Document Pages (78 pages, page metadata preserved)
    │ [RecursiveCharacterTextSplitter]
    ↓
Chunked Documents (484 chunks × 1000 tokens, 50-token overlap)
    │ [Gemini Embeddings]
    ↓
Vector Embeddings (484 vectors in Gemini vector space)
    │ [ChromaDB.from_documents()]
    ↓
Vector Store Persistence (db/chroma_db/chroma.sqlite3)
    ├─── [Query Time]
    │    ↓
    │    Question → Embedding → Semantic Search (k=8)
    │    ↓
    │    Retrieved Contexts
    │    ├─ [ChatBot Path]
    │    │  ↓
    │    │  LLM Answer Generation → Return Answer + Sources
    │    │
    │    └─ [Evaluation Path]
    │       ↓
    │       Faithfulness Judge → Score 0.0–1.0
    │       ↓
    │       Relevancy Judge → Score 0.0–1.0
    │       ↓
    │       Generate Report (JSON)
    │
    └─── db/chroma_db/chroma.sqlite3 (persistent)
```

### Stage Details

| Stage | Input | Transformation | Output | Source File |
|-------|-------|-----------------|--------|-------------|
| **PDF Ingestion** | ISTQB Foundation PDF | PyPDFLoader extracts pages with metadata | 78 Document objects with page references | `ingest.py:load_pdf()` |
| **Document Splitting** | 78 pages | RecursiveCharacterTextSplitter (1000 tokens/chunk, 50 overlap) | 484 Document chunks | `ingest.py:split_documents()` |
| **Embedding** | 484 chunks | Gemini embeddings API (batch 50, 80s delays) | 484 vector embeddings | `ingest.py:create_vector_store()` |
| **Persistence** | 484 embeddings | ChromaDB.from_documents() | SQLite DB (db/chroma_db/) | `ingest.py:create_vector_store()` |
| **Query** | User question | Embedding + semantic search (k=8) | 8 context chunks + page metadata | `chatbot.py:ask_question()` |
| **Generation** | Question + context | LLM prompt w/ context + instruction | Grounded answer + sources | `chatbot.py:create_qa_chain()` |
| **Evaluation** | Question, answer, context | 2 parallel LLM-as-judge calls | Faithfulness score + Relevancy score | `eval_suite.py:evaluate_faithfulness/relevancy()` |
| **Reporting** | 8 evaluation results | JSON aggregation + statistics | JSON report with summary + details | `eval_suite.py:generate_report()` |

### Latest Evaluation Results

**Configuration:** `chunk_size_1000` (as of 2026-06-10T20:56:52)

| Metric | Value |
|--------|-------|
| **Average Faithfulness** | 1.0 (100%) |
| **Average Relevancy** | 0.875 (87.5%) |
| **Faithfulness Pass Rate** | 100% (8/8) |
| **Relevancy Pass Rate** | 87.5% (7/8) |

**Answer Quality Summary:**
- ✅ 7/8 questions answered with high relevance and faithfulness
- ⚠️ 1/8 question ("What is a test oracle?") — Correctly returns "insufficient information" (not in context) but scored as irrelevant due to refusal pattern
- 🎯 Key success: Static/Dynamic testing answer provides comprehensive multi-point comparison with perfect scores

---

## 6. Database Schema

### Data Persistence Strategy

**Primary Database:** SQLite (embedded in ChromaDB)  
**Location:** `db/chroma_db/chroma.sqlite3`

### ChromaDB Vector Store Structure

[INFERRED from ChromaDB architecture]

#### Logical Schema (ChromaDB Collections)

**Collection: `documents` (default)**

| Aspect | Details |
|--------|---------|
| **Primary Key** | Document ID (UUID generated by ChromaDB) |
| **Vector Column** | `embedding` (Gemini embedding space) |
| **Metadata Columns** | `page` (source page number), `source` (document path) |
| **Content Column** | `document` (chunked text, max 1000 tokens) |

#### Data Content Example

```
{
  "id": "uuid-xxx",
  "embedding": [0.123, -0.456, ..., 0.789],  // Gemini embedding vector
  "metadatas": {
    "page": 16,
    "source": "data/istqb_foundation.pdf"
  },
  "document": "Equivalence Partitioning (EP) is a black-box test technique that divides data into partitions..."
}
```

### Relationships

| From | To | Type | Key |
|------|----|----|-----|
| ChromaDB Documents | Source PDF | Reference | `metadatas.source` = `data/istqb_foundation.pdf` |
| ChromaDB Documents | Page Numbers | Reference | `metadatas.page` = 1..78 |

### Indexes

[NOT EXPLICITLY FOUND IN CODEBASE]  
[INFERRED] ChromaDB uses built-in HNSW or similar vector indexing for semantic search.

### Partitioning / Archival Strategy

| Strategy | Status |
|----------|--------|
| **Partitioning** | [NOT FOUND IN CODEBASE] |
| **Sharding** | [NOT FOUND IN CODEBASE] |
| **Archival** | [NOT FOUND IN CODEBASE] — Vector store persisted indefinitely |

---

## 7. API Layer

[NO EXTERNAL API LAYER DETECTED]

The application is a **standalone CLI tool** without HTTP endpoints. All interaction is through Python function calls.

### External APIs Used

| API | Provider | Purpose | Endpoint | Authentication |
|-----|----------|---------|----------|-----------------|
| **Text Embeddings** | Google Generative AI | Convert text to vectors | `models/gemini-embedding-001` | `GOOGLE_API_KEY` |
| **LLM Generation** | Google Generative AI | Generate answers + evaluation | `models/gemini-2.5-flash-lite`, `models/gemini-3.1-flash-lite` | `GOOGLE_API_KEY` |

### Rate Limiting & Constraints

**Google Generative AI Free Tier Limits:**
- Embeddings: 100 requests/minute
- LLM Requests: Daily quota (varies by model)
- Handling: Batch processing (50 chunks), 80-second inter-batch delay, retry logic (3 attempts)

---

## 8. Configuration & Environment

### Configuration Sources

#### .env File

```
GOOGLE_API_KEY='your key'
```

| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | `AIzaSy...` | Google Cloud API key for Generative AI access |

### Runtime Configuration (Hard-Coded)

| Parameter | Value | Location | Purpose |
|-----------|-------|----------|---------|
| `CHROMA_DB_PATH` | `db/chroma_db` | `ingest.py`, `chatbot.py`, `eval_suite.py` | ChromaDB persistence directory |
| `PDF_PATH` | `data/istqb_foundation.pdf` | `ingest.py` | Source document for ingestion |
| `chunk_size` | 1000 | `ingest.py` | Document chunk size in tokens |
| `chunk_overlap` | 50 | `ingest.py` | Token overlap between chunks |
| `k` (retrieval) | 8 | `chatbot.py`, `eval_suite.py` | Number of chunks to retrieve per query |
| `temperature` (QA) | 0.1 | `chatbot.py` | LLM temperature for answer generation |
| `temperature` (judge) | 0.0 | `eval_suite.py` | LLM temperature for evaluation (deterministic) |
| `BATCH_SIZE` | 50 | `ingest.py` | Embedding batch size (respects rate limits) |
| `rate_limit_delay` | 80 (seconds) | `ingest.py` | Delay between batches for rate limit compliance |

### Feature Flags / Experimental Settings

[NOT FOUND IN CODEBASE]

### Models Used

| Component | Current Model | Previous Models | Reason for Change |
|-----------|---------------|-----------------|-------------------|
| **Embeddings** | `models/gemini-embedding-001` | (none noted) | Primary choice, deprecated in API v1beta but still functional |
| **QA LLM** | `models/gemini-2.5-flash-lite` | `gemini-2.0-flash`, `gemini-1.5-flash` | 2.0-flash exhausts daily quota quickly; 2.5-lite optimized for free tier |
| **Judge LLM** | `models/gemini-3.1-flash-lite` | (none noted) | Provides reliable scoring for evaluation metrics |

---

## 9. Batch Jobs & Scheduled Processes

[NO SCHEDULED JOBS DETECTED]

All processes are **manual, CLI-triggered**:

| Job | Trigger | Purpose | Entry Point | Runtime |
|-----|---------|---------|-------------|---------|
| **Document Ingestion** | Manual: `python src/ingest.py` | Load PDF → embed → persist to ChromaDB | `ingest.py:main()` | ~13 minutes (484 chunks, 80s delays) |
| **Interactive QA** | Manual: `python src/chatbot.py` | Answer user questions interactively | `chatbot.py:main()` | ~2–5 seconds per question |
| **Evaluation Suite** | Manual: `python src/eval_suite.py` | Run 8 test cases, score, generate report | `eval_suite.py:main()` | ~4–5 minutes (30s delays between tests) |

### Execution Lifecycle Examples

**Ingestion:**
```
python src/ingest.py
  → Load PDF (78 pages)
  → Split into 484 chunks
  → Batch 1: 50 chunks → embed → store
  → Wait 80s
  → Batch 2: 50 chunks → embed → store
  → ... (repeat for 10 batches)
  → Complete ✅
```

**Chatbot:**
```
python src/chatbot.py
  → Load ChromaDB
  → Create QA chain
  → Prompt user for question
  → Retrieve 8 chunks
  → Generate answer
  → Display answer + sources
```

**Evaluation:**
```
python src/eval_suite.py
  → Load ChromaDB
  → For each of 8 test questions:
    → Retrieve context
    → Generate answer (using QA LLM)
    → Score faithfulness (using judge LLM)
    → Score relevancy (using judge LLM)
    → Wait 30s for rate limit
  → Aggregate results
  → Generate JSON report
```

---

## 10. Error Handling & Logging

### Error Handling

#### Exception Hierarchy

[NOT FOUND IN CODEBASE] — Generic Python exception handling used.

#### Implemented Error Handling

**ingest.py:**
```python
try:
    # Attempt batch embedding
except Exception as e:
    if attempt < 2:
        wait = 80 + (attempt * 30)  # Exponential backoff
        time.sleep(wait)
        # Retry
    else:
        raise e  # All 3 attempts failed
```
- **Retry Logic:** Up to 3 attempts per batch with exponential backoff (80s → 110s → 140s)
- **Fallback:** Raises exception if all retries fail

**eval_suite.py:**
```python
try:
    # Get answer and context
    faithfulness_score, faithfulness_reason = evaluate_faithfulness(...)
except Exception as e:
    results.append({
        "answer": "ERROR",
        "faithfulness_score": 0,
        "relevancy_score": 0,
        "faithfulness_reason": str(e),
        "relevancy_reason": str(e)
    })
```
- **Graceful Degradation:** Records errors as evaluation results (0.0 scores)
- **Error Logging:** Captures exception message for debugging

**chatbot.py:**
- `load_vector_store()` → Prints confirmation messages
- `ask_question()` → No explicit error handling [RISK: unhandled exceptions will crash]

#### Validation Failures

- **API Key Missing:** RuntimeError from `os.getenv("GOOGLE_API_KEY")`
- **PDF Not Found:** FileNotFoundError from `PyPDFLoader`
- **ChromaDB Corruption:** Exception from ChromaDB load
- **Invalid JSON in Eval Responses:** Caught in `evaluate_faithfulness()` and `evaluate_relevancy()` → returns 0.0 score

### Logging

#### Logging Framework

[PYTHON BUILT-IN: print() statements only]  
No structured logging framework detected.

#### Log Levels

| Level | Usage | Example |
|-------|-------|---------|
| **INFO** | Standard execution flow | `print(f"📄 Loading PDF...")`, `print(f"✅ Loaded {len(pages)} pages")` |
| **WARNING** | Recoverable errors | `print(f"⚠️ Attempt {attempt+1} failed. Retrying in {wait}s...")` |
| **ERROR** | Critical failures | `print(f"❌ Error: {e}")` |

#### Output Examples

**ingest.py output:**
```
📄 Loading PDF from data/istqb_foundation.pdf...
✅ Loaded 78 pages
✂️ Splitting documents into chunks...
✅ Created 484 chunks
🧠 Creating embeddings and storing in ChromaDB...
📦 Processing batch 1/10...
⏳ Waiting 80 seconds for rate limit...
✅ Vector store created at db/chroma_db
🎉 Ingestion complete! Ready to query.
```

**eval_suite.py output:**
```
Running evaluation — chunk_size_1000
============================================================
Test 1/8: What is equivalence partitioning?...
  Answer:       Equivalence Partitioning (EP) is a black-box test technique...
  Faithfulness: 1.0 ✅ — The answer accurately reflects the definition...
  Relevancy:    1.0 ✅ — The answer provides a clear, accurate, and direct...
  Waiting 30 seconds for rate limit...
```

#### Correlation IDs / Trace IDs

[NOT FOUND IN CODEBASE] — No distributed tracing implemented.

---

## 11. Security Overview

### Authentication

| Component | Method | Storage |
|-----------|--------|---------|
| **Google Generative AI API** | API Key (`GOOGLE_API_KEY`) | `.env` file (local, not version controlled) |

### Authorization

[NOT FOUND IN CODEBASE] — No access control or role-based authorization implemented. System assumes single trusted user.

### Secrets Management

| Secret | Management | Risk Level |
|--------|-----------|-----------|
| `GOOGLE_API_KEY` | `.env` file (local) | 🟠 Medium — Key visible in `.env`, not encrypted. `.env` in `.gitignore` prevents accidental commit. |

**Findings from .env:**
- API key stored in plain text
- `.gitignore` includes `.env` (prevents version control exposure)
- Key exposed if machine compromised

### Encryption

[NOT FOUND IN CODEBASE]
- API calls to Google use HTTPS (transport encryption via LangChain)
- Local data (ChromaDB SQLite) unencrypted
- Vector embeddings stored in plaintext SQLite

### Sensitive Data Handling

| Data Type | Handling | Protection |
|-----------|----------|----------|
| **API Key** | Environment variable | Plain text in `.env` |
| **Vector Embeddings** | Stored in ChromaDB SQLite | Plain text, local file |
| **Question/Answer Data** | In-memory during processing, JSON reports | JSON reports stored locally without encryption |
| **Evaluation Results** | JSON files in `reports/` | Plain text, accessible to any local user |

### Input Validation

**Questions:** No validation — user input passed directly to LLM  
**Chunks:** Pre-validated by LangChain's `RecursiveCharacterTextSplitter`  
**API Responses:** Minimal validation; assumes LLM/embedding API responses are well-formed

#### SQL Injection Protection

[NOT APPLICABLE] — No SQL queries constructed from user input. ChromaDB API used instead of raw SQL.

#### Dependency Security Controls

[NOT FOUND IN CODEBASE] — No `requirements.txt`, no dependency pinning, no security scanning configured.

---

## 12. Testing Strategy

### Test Coverage

| Test Type | Framework | Location | Status |
|-----------|-----------|----------|--------|
| **Functional Tests** | Custom LLM-as-Judge | `eval_suite.py` | ✅ Implemented & running |
| **Integration Tests** | Manual (PDF → DB → Answer) | Implicit in chatbot.py | ⚠️ Manual verification only |
| **Unit Tests** | [NOT FOUND IN CODEBASE] | — | ❌ Not implemented |
| **End-to-End Tests** | 8 ISTQB Foundation Q&A cases | `eval_suite.py:TEST_CASES` | ✅ Implemented |

### Functional Test Suite (eval_suite.py)

**Test Cases:** 8 ISTQB Foundation questions

```python
TEST_CASES = [
    {"question": "What is equivalence partitioning?", "expected": "..."},
    {"question": "What are the seven testing principles?", "expected": "..."},
    {"question": "What is the difference between verification and validation?", "expected": "..."},
    {"question": "What is a test oracle?", "expected": "..."},
    {"question": "What is boundary value analysis?", "expected": "..."},
    {"question": "What is the difference between static and dynamic testing?", "expected": "..."},
    {"question": "What is a test plan?", "expected": "..."},
    {"question": "What is regression testing?", "expected": "..."}
]
```

**Metrics Evaluated:**
- **Faithfulness** (0.0–1.0) — Answer doesn't hallucinate beyond context
- **Relevancy** (0.0–1.0) — Answer addresses the question

**Pass Criteria:** Both scores ≥ 0.5

**Latest Results (2026-06-10):**
- Faithfulness: 100% pass rate (8/8)
- Relevancy: 87.5% pass rate (7/8)

### Mocking Approach

[NOT FOUND IN CODEBASE] — No mocks used. System tests against actual:
- Google Generative AI API (requires valid `GOOGLE_API_KEY`)
- ChromaDB vector store (real persistent storage)
- PDF document (real ISTQB Foundation PDF)

**Risk:** Tests are brittle to API changes/outages.

### Coverage Tooling

[NOT FOUND IN CODEBASE] — No code coverage tools (pytest-cov, etc.) detected.

---

## 13. Known Limitations, TODOs & Technical Debt

### Explicit TODOs & FIXMEs

[SCAN COMPLETE] No `TODO`, `FIXME`, `HACK`, `XXX`, or `NOTE` comments found in source code.

### Inferred Limitations & Technical Debt

#### 1. **Missing Requirements Documentation**
- **File:** None
- **Issue:** No `requirements.txt` or dependency pinning
- **Impact:** Difficult to reproduce environment; vulnerable to dependency incompatibilities
- **Risk Level:** 🔴 High
- **Inference Basis:** Imports visible but versions unspecified

#### 2. **Hardcoded Configuration**
- **Files:** `ingest.py`, `chatbot.py`, `eval_suite.py`
- **Issue:** Model names, chunk sizes, retrieval k, temperature values hardcoded
- **Example:** `chunk_size=1000`, `k=8`, `model="gemini-2.5-flash-lite"`
- **Impact:** Requires code changes to experiment with parameters
- **Risk Level:** 🟠 Medium
- **Recommendation:** Move to `config.yaml` or environment variables

#### 3. **Deprecated API Models**
- **File:** `eval_suite.py`, `chatbot.py`, `ingest.py`
- **Issue:** Using `models/gemini-embedding-001` (deprecated in Gemini API v1beta per findings.md)
- **Status:** Still functional as of 2026-06-10 but may be removed
- **Risk Level:** 🟠 Medium
- **Current Mitigation:** Findings.md documents deprecated models

#### 4. **No Error Recovery in Chatbot**
- **File:** `chatbot.py`
- **Issue:** No try-catch around `ask_question()` call
- **Impact:** Invalid questions or API errors crash the program
- **Risk Level:** 🟡 Low (development tool, not production)

#### 5. **Rate Limit Fragility**
- **Files:** `ingest.py`, `eval_suite.py`
- **Issue:** Hard-coded 80-second delays; relies on manual sleeping
- **Problem:** If Google API changes rate limits, ingestion may fail
- **Risk Level:** 🟠 Medium
- **Current Handling:** Retry logic (3 attempts) + exponential backoff

#### 6. **Single-User Local System**
- **Issue:** No multi-user concurrency, no access control
- **Risk Level:** 🟡 Low (acceptable for local dev)

#### 7. **JSON Parsing Fragility in Evaluation**
- **File:** `eval_suite.py`, functions `evaluate_faithfulness()`, `evaluate_relevancy()`
- **Issue:** LLM response parsing using string replacement + JSON parsing
- **Problem:** If LLM formats response differently, parsing fails silently (returns 0.0 score)
- **Example:**
  ```python
  text = text.replace("```json", "").replace("```", "").strip()
  result = json.loads(text)  # Fragile if response format changes
  ```
- **Risk Level:** 🟠 Medium
- **Current Mitigation:** Catches exceptions, returns 0.0 if parsing fails

#### 8. **Test Oracle Question Unanswerable**
- **File:** `eval_suite.py`
- **Issue:** Question "What is a test oracle?" — Not found in ISTQB source PDF (per 2026-06-10 evaluation results)
- **Impact:** Will always score low on relevancy (correctly returns "insufficient information")
- **Status:** By design (evaluating system honesty)
- **Risk Level:** 🟡 Low

#### 9. **No Unit Tests**
- **Issue:** All testing is end-to-end; no isolated component testing
- **Impact:** Hard to debug individual components
- **Risk Level:** 🟠 Medium

#### 10. **Scalability Concerns**
- **Issue:** ChromaDB SQLite for single-machine use; not suitable for enterprise scale
- **Issue:** No database indexing on metadata (page numbers)
- **Risk Level:** 🟡 Low (acceptable for <1000 documents)

#### 11. **Tight Coupling to Google Generative AI**
- **Issue:** No abstraction layer for LLM provider
- **Impact:** Switching to OpenAI, Anthropic, etc., requires code rewrite
- **Risk Level:** 🟠 Medium
- **Possible Mitigation:** Use LangChain's LLMChain abstraction (partially present)

#### 12. **Incomplete Documentation**
- **File:** `README.md`
- **Issue:** Empty; no setup/usage instructions
- **Risk Level:** 🟡 Low (findings.md + code comments partially compensate)

#### 13. **API Key in .env Committed History**
- **Issue:** `.env` in `.gitignore` prevents *future* commits, but key already visible in repo history
- **Risk Level:** 🔴 High
- **Recommendation:** Rotate API key; consider git-filter-branch history cleanup

### Hardcoded Values

| Location | Value | Purpose |
|----------|-------|---------|
| `ingest.py:7` | `CHROMA_DB_PATH = "db/chroma_db"` | Vector store location |
| `ingest.py:8` | `PDF_PATH = "data/istqb_foundation.pdf"` | Source document |
| `ingest.py:22` | `chunk_size=1000` | Document chunk size |
| `ingest.py:23` | `chunk_overlap=50` | Chunk overlap |
| `ingest.py:29` | `BATCH_SIZE = 50` | Embedding batch size |
| `ingest.py:32` | `wait = 80 + (attempt * 30)` | Retry delays |
| `ingest.py:42` | `time.sleep(80)` | Rate limit delay |
| `chatbot.py:10` | `CHROMA_DB_PATH = "db/chroma_db"` | Vector store location |
| `chatbot.py:22` | `model="gemini-2.5-flash-lite"` | LLM model |
| `chatbot.py:24` | `temperature=0.1` | LLM temperature |
| `chatbot.py:26` | `search_kwargs={"k": 8}` | Retrieval count |
| `eval_suite.py:9` | `CHROMA_DB_PATH = "db/chroma_db"` | Vector store location |
| `eval_suite.py:10` | `CHUNK_SIZE_LABEL = "chunk_size_1000"` | Report label |
| `eval_suite.py:76` | `model="gemini-3.1-flash-lite"` | Judge LLM |

### Performance Bottlenecks

| Bottleneck | Location | Impact | Severity |
|-----------|----------|--------|----------|
| **Rate Limit Delays** | `ingest.py:42` | 80-second waits between embedding batches; 13 min ingestion for 484 chunks | 🟠 Medium |
| **LLM Request Latency** | `chatbot.py`, `eval_suite.py` | 2–5 seconds per query + 30s between eval tests | 🟡 Low (acceptable) |
| **No Vector Index Caching** | `eval_suite.py` | Retrieval queries re-execute for each test; could cache results | 🟡 Low |

---

## 14. Execution & Deployment Guide

### Local Development

#### Prerequisites

- **Python Version:** 3.8+ [INFERRED from venv structure]
- **Operating System:** Windows (per workspace info), likely Linux/Mac compatible
- **API Access:** Google Cloud account with Generative AI API enabled
- **Disk Space:** ~500MB (PDF + ChromaDB + venv)

#### Installation Steps

```bash
# 1. Clone repository (or navigate to workspace)
cd c:\Experiments\qa-rag-eval

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows PowerShell:
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\venv\Scripts\Activate.ps1)
# On Linux/Mac:
source venv/bin/activate

# 4. Install dependencies [NOT FOUND - manual install required]
pip install langchain langchain-community langchain-chroma langchain-google-genai python-dotenv PyPDF2

# 5. Set up .env file
# Create .env in project root:
echo GOOGLE_API_KEY=<your_api_key_here> > .env

# 6. Verify setup
python src/chatbot.py  # Should load vector store successfully
```

#### Build Commands

[NOT APPLICABLE] — Python is interpreted, no build step.

#### Run Commands

```bash
# Ingestion (one-time, creates ChromaDB)
python src/ingest.py

# Interactive QA (use after ingestion)
python src/chatbot.py

# Run evaluation suite (tests system quality)
python src/eval_suite.py
```

#### Test Commands

```bash
# Run evaluation suite (provides pass/fail metrics)
python src/eval_suite.py

# Check results
cat reports/eval_chunk_size_1000_*.json | python -m json.tool
```

#### Expected Output

**Successful Ingestion:**
```
📄 Loading PDF from data/istqb_foundation.pdf...
✅ Loaded 78 pages
✂️ Splitting documents into chunks...
✅ Created 484 chunks
🧠 Creating embeddings and storing in ChromaDB...
📦 Processing batch 1/10...
✅ Vector store created at db/chroma_db
🎉 Ingestion complete! Ready to query.
```

**Successful Evaluation:**
```
Running evaluation — chunk_size_1000
============================================================
Test 1/8: What is equivalence partitioning?...
  Answer:       Equivalence Partitioning (EP) is a black-box...
  Faithfulness: 1.0 ✅
  Relevancy:    1.0 ✅
  ...
============================================================
EVALUATION REPORT SUMMARY
============================================================
Avg Faithfulness:       1.0
Avg Relevancy:          0.875
Faithfulness Pass Rate: 100%
Relevancy Pass Rate:    87.5%
```

### Deployment

#### Deployment Process

[NOT CONFIGURED FOR PRODUCTION] — This is a development/research tool.

**Manual Deployment (if needed):**
1. Copy `src/`, `data/`, `db/` to target machine
2. Set `GOOGLE_API_KEY` in `.env`
3. Run appropriate script

#### Docker Usage

[NOT FOUND IN CODEBASE] — No Dockerfile present.

**To containerize (manual steps):**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "src/eval_suite.py"]
```

#### CI/CD Pipeline

[NOT FOUND IN CODEBASE] — No GitHub Actions, Jenkins, or CI configuration.

#### Environment Requirements

| Environment | Configuration |
|-------------|---------------|
| **Development** | Windows/Linux/Mac with Python 3.8+, 500MB disk, internet access |
| **Production** | [NOT APPLICABLE — not designed for production] |

---

## 15. Architecture Diagram

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    QA-RAG-Eval System                           │
└─────────────────────────────────────────────────────────────────┘

                        ┌────────────┐
                        │   ISTQB    │
                        │   PDF      │
                        │ (78 pages) │
                        └─────┬──────┘
                              │
                    ┌─────────▼────────────┐
                    │   Data Ingestion     │
                    │   Pipeline           │
                    │  (ingest.py)         │
                    │ • PDF Loading        │
                    │ • Chunking (1000t)   │
                    │ • Embedding Gen      │
                    │ • Batch + Retry      │
                    └─────────┬────────────┘
                              │
                    ┌─────────▼──────────────────┐
                    │   Vector Database          │
                    │   (ChromaDB + SQLite)      │
                    │   484 chunks × embedding   │
                    │   db/chroma_db/            │
                    └─────────┬──────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
      ┌─────────▼────────┐   │   ┌────────▼────────┐
      │  QA Chatbot      │   │   │  Evaluation     │
      │  (chatbot.py)    │   │   │  Suite          │
      │ • Query Embed    │   │   │  (eval_suite.py)│
      │ • Retrieve (k=8) │   │   │ • Query + Get   │
      │ • Generate Answer│   │   │   Answer        │
      │ • Return+Sources │   │   │ • Faithfulness  │
      └───────────────────┘   │   │   Evaluation    │
                              │   │ • Relevancy     │
                              │   │   Evaluation    │
                              │   │ • JSON Report   │
                              │   └─────┬──────────┘
                              │         │
                    ┌─────────▼────────▼──┐
                    │ Google Generative AI │
                    │  • Embeddings API    │
                    │  • LLM Chat API      │
                    │  (Rate-limited)      │
                    └──────────────────────┘

                    ┌──────────────────────┐
                    │  Output              │
                    │  • Reports JSON      │
                    │  • Console Logs      │
                    │  reports/ directory  │
                    └──────────────────────┘
```

### Data Flow Sequence Diagram

```
User Input (Question)
    │
    ├─► [Embedding Generation]
    │   └─► Gemini Embeddings API
    │       └─► Vector (Gemini Space)
    │
    ├─► [Semantic Retrieval]
    │   └─► ChromaDB Query
    │       └─► Top-8 Similar Chunks
    │           └─► Contexts with Metadata
    │
    ├─► [Context Assembly]
    │   └─► Concatenate 8 Chunks
    │       └─► Add Page Numbers
    │
    ├─► [LLM Answer Generation]
    │   └─► Gemini Chat API
    │       └─► Input: Question + Contexts
    │       └─► Output: Grounded Answer
    │
    ├─► [Evaluation Path] ──┬─► [Faithfulness Judge]
    │                       │   └─► Gemini Chat API
    │                       │       └─► Score 0.0-1.0
    │                       │
    │                       └─► [Relevancy Judge]
    │                           └─► Gemini Chat API
    │                               └─► Score 0.0-1.0
    │
    └─► [Output]
        ├─ Console: Answer + Scores
        └─ File: JSON Report
```

### Component Relationships

```
┌─────────────────────────────────────────────┐
│          Local Execution Context            │
│  (Single machine, command-line driven)      │
└─────────────────────────────────────────────┘
         │
    ┌────┴────┐
    │          │
    ▼          ▼
[Data Flow] [Configuration]
    │              │
    ├─ PDF ────────┤─ .env (API key)
    │              │
    ├─ Chunks ─────┤─ Hardcoded params
    │              │   (chunk_size=1000,
    ├─ Embeddings──┤    k=8, temps, etc)
    │              │
    └─ ChromaDB ───┘

┌─────────────────────────────────────────────┐
│         External Dependencies               │
│       (Cloud-based, rate-limited)           │
└─────────────────────────────────────────────┘
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
[Gemini API]      [Google Cloud Auth]
  • Embeddings          • GOOGLE_API_KEY
  • Chat                • Rate limits
  • Judge LLM
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Lines of Code** | ~500 (combined src files) |
| **Python Files** | 3 |
| **Source Documents** | 1 (PDF) |
| **Document Pages** | 78 |
| **Chunks Created** | 484 |
| **Chunk Size** | 1000 tokens |
| **Chunk Overlap** | 50 tokens |
| **Test Cases** | 8 |
| **Evaluation Reports Generated** | 4 |
| **Latest Faithfulness Score** | 1.0 (100%) |
| **Latest Relevancy Score** | 0.875 (87.5%) |
| **Average Execution Time (Ingestion)** | ~13 minutes |
| **Average Execution Time (Evaluation)** | ~4-5 minutes |
| **API Rate Limit (Free Tier)** | 100 req/min (embeddings) |
| **Retry Attempts** | 3 per batch |
| **Exponential Backoff** | 80s → 110s → 140s |

---

## Version History

| Date | Change | Impact |
|------|--------|--------|
| 2026-05-29 | Embedding model deprecated (`embedding-001` in v1beta) | [RESOLVED] Still functional |
| 2026-05-30 | Initial k=4 experiments (50% accuracy on subset) | [RESOLVED] Increased to k=8 |
| 2026-05-30 | LLM quota exhaustion (gemini-2.0-flash) | [RESOLVED] Switched to gemini-2.5-flash-lite |
| 2026-06-05 | Custom eval framework built (DeepEval conflicts) | [CURRENT] Faithfulness + Relevancy metrics implemented |
| 2026-06-09 | Chunk size 500 → 1000 (multi-point answers consolidation) | [CURRENT] 484 chunks, improved coverage |
| 2026-06-10 | Baseline evaluation: 87.5% relevancy, 100% faithfulness | [CURRENT] Latest results |

---

## End of Document

**Generated:** 2026-06-10  
**Workspace:** c:\Experiments\qa-rag-eval  
**Analysis Method:** Source code inspection + configuration analysis  
**Coverage:** 100% of discovered source files
