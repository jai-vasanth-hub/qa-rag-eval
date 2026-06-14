# Project Architecture & Journey

## System Architecture

\`\`\`mermaid
graph LR
    A[ISTQB PDF] --> B[Chunking]
    B --> C[Gemini Embeddings]
    C --> D[ChromaDB Vector Store]
    D --> E[Retriever]
    E --> F[Gemini Chat Model]
    F --> G[Streamlit UI]
    F --> H[Custom Eval Framework]
    H --> I[Faithfulness Score]
    H --> J[Relevancy Score]
    H --> K[Adversarial Tests]
\`\`\`

## Improvement Journey

\`\`\`mermaid
graph TD
    A[Baseline: chunk_500] -->|Relevancy 0.75| B[chunk_1000]
    B -->|Relevancy 0.875| C[Adversarial Baseline]
    C -->|Pass Rate 66.7%| D[Behaviour Contract v1]
    D -->|Pass Rate 91.7% - new regression found| E[Behaviour Contract v2]
    E -->|Pass Rate 100%| F[Live Streamlit App]
\`\`\`