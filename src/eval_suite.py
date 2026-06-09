import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

# Constants
CHROMA_DB_PATH = "db/chroma_db"
CHUNK_SIZE_LABEL = "chunk_size_500"

# Test cases
TEST_CASES = [
    {
        "question": "What is equivalence partitioning?",
        "expected": "Equivalence partitioning divides data into partitions where all elements are processed the same way by the test object."
    },
    {
        "question": "What are the seven testing principles?",
        "expected": "The seven testing principles are: testing shows presence of defects, exhaustive testing is impossible, early testing saves time and money, defects cluster together, pesticide paradox, testing is context dependent, absence of errors fallacy."
    },
    {
        "question": "What is the difference between verification and validation?",
        "expected": "Verification checks whether the system meets specified requirements while validation checks whether the system meets users and stakeholders needs."
    },
    {
        "question": "What is a test oracle?",
        "expected": "A test oracle is a source to determine expected results to compare with actual results of the software under test."
    },
    {
        "question": "What is boundary value analysis?",
        "expected": "Boundary value analysis tests values at the boundaries of equivalence partitions including minimum and maximum values."
    },
    {
        "question": "What is the difference between static and dynamic testing?",
        "expected": "Static testing reviews work products without executing code while dynamic testing executes the software being tested."
    },
    {
        "question": "What is a test plan?",
        "expected": "A test plan is a document describing the scope objectives resources and schedule of intended test activities."
    },
    {
        "question": "What is regression testing?",
        "expected": "Regression testing confirms that changes have not caused adverse effects and that the system still meets requirements."
    }
]

def load_vector_store():
    print("Loading knowledge base...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    print("Knowledge base loaded successfully!")
    return vectorstore

def get_answer_and_context(vectorstore, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = [doc.page_content.strip() for doc in docs]
    context_text = "\n".join(context)

    prompt = f"""You are a QA knowledge assistant trained on ISTQB Foundation syllabus.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information."
Do not make up answers.

Context:
{context_text}

Question:
{question}

Answer:"""

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        actual_output: str = " ".join(
            item.get('text', '') if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    else:
        actual_output = str(response.content).strip()
    return actual_output, context

def evaluate_faithfulness(question, answer, context, judge_llm):
    context_text = "\n".join(context)
    prompt = f"""You are an expert evaluator for AI systems.

Evaluate whether the answer is faithful to the context provided.
Faithful means: every claim in the answer is supported by the context.
Not faithful means: the answer contains information not found in the context.

Context:
{context_text}

Question:
{question}

Answer:
{answer}

Respond with ONLY a JSON object like this:
{{"score": 0.85, "reason": "brief reason"}}

Score must be between 0.0 and 1.0.
1.0 = completely faithful, 0.0 = completely hallucinated."""

    response = judge_llm.invoke(prompt)
    try:
        content = response.content
        # Handle list response format
        if isinstance(content, list):
            text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
        else:
            text = str(content)
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return round(float(result["score"]), 2), result["reason"]
    except Exception as e:
        return 0.0, "Could not parse response"

def evaluate_relevancy(question, answer, judge_llm):
    prompt = f"""You are an expert evaluator for AI systems.

Evaluate whether the answer is relevant to the question asked.
Relevant means: the answer directly addresses what was asked.
Not relevant means: the answer talks about something else.

Question:
{question}

Answer:
{answer}

Respond with ONLY a JSON object like this:
{{"score": 0.85, "reason": "brief reason"}}

Score must be between 0.0 and 1.0.
1.0 = perfectly relevant, 0.0 = completely irrelevant."""

    response = judge_llm.invoke(prompt)
    try:
        content = response.content
        # Handle list response format
        if isinstance(content, list):
            text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
        else:
            text = str(content)
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return round(float(result["score"]), 2), result["reason"]
    except Exception as e:
        return 0.0, "Could not parse response"

def run_evaluation(vectorstore):
    print(f"\nRunning evaluation — {CHUNK_SIZE_LABEL}")
    print("=" * 60)

    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0
    )

    results = []

    for i, tc in enumerate(TEST_CASES):
        print(f"\nTest {i+1}/{len(TEST_CASES)}: {tc['question'][:50]}...")

        try:
            actual_answer, context = get_answer_and_context(
                vectorstore,
                tc['question']
            )

            faithfulness_score, faithfulness_reason = evaluate_faithfulness(
                tc['question'], actual_answer, context, judge_llm
            )
            relevancy_score, relevancy_reason = evaluate_relevancy(
                tc['question'], actual_answer, judge_llm
            )

            result = {
                "question": tc['question'],
                "answer": actual_answer,
                "faithfulness_score": faithfulness_score,
                "relevancy_score": relevancy_score,
                "faithfulness_passed": faithfulness_score >= 0.5,
                "relevancy_passed": relevancy_score >= 0.5,
                "faithfulness_reason": faithfulness_reason,
                "relevancy_reason": relevancy_reason
            }
            results.append(result)

            print(f"  Answer:       {actual_answer[:80]}...")
            print(f"  Faithfulness: {faithfulness_score} {'✅' if faithfulness_score >= 0.5 else '❌'} — {faithfulness_reason[:60]}")
            print(f"  Relevancy:    {relevancy_score} {'✅' if relevancy_score >= 0.5 else '❌'} — {relevancy_reason[:60]}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "question": tc['question'],
                "answer": "ERROR",
                "faithfulness_score": 0,
                "relevancy_score": 0,
                "faithfulness_passed": False,
                "relevancy_passed": False,
                "faithfulness_reason": str(e),
                "relevancy_reason": str(e)
            })

        if i < len(TEST_CASES) - 1:
            print(f"  Waiting 30 seconds for rate limit...")
            time.sleep(30)

    return results

def generate_report(results):
    avg_faithfulness = sum(r['faithfulness_score'] for r in results) / len(results)
    avg_relevancy = sum(r['relevancy_score'] for r in results) / len(results)
    faithfulness_pass_rate = sum(1 for r in results if r['faithfulness_passed']) / len(results) * 100
    relevancy_pass_rate = sum(1 for r in results if r['relevancy_passed']) / len(results) * 100

    report = {
        "label": CHUNK_SIZE_LABEL,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_questions": len(results),
            "avg_faithfulness": round(avg_faithfulness, 3),
            "avg_relevancy": round(avg_relevancy, 3),
            "faithfulness_pass_rate": round(faithfulness_pass_rate, 1),
            "relevancy_pass_rate": round(relevancy_pass_rate, 1)
        },
        "details": results
    }

    report_path = f"reports/eval_{CHUNK_SIZE_LABEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT SUMMARY")
    print("=" * 60)
    print(f"Label:                  {CHUNK_SIZE_LABEL}")
    print(f"Total Questions:        {report['summary']['total_questions']}")
    print(f"Avg Faithfulness:       {report['summary']['avg_faithfulness']}")
    print(f"Avg Relevancy:          {report['summary']['avg_relevancy']}")
    print(f"Faithfulness Pass Rate: {report['summary']['faithfulness_pass_rate']}%")
    print(f"Relevancy Pass Rate:    {report['summary']['relevancy_pass_rate']}%")
    print(f"\nReport saved to: {report_path}")

    return report

if __name__ == "__main__":
    vectorstore = load_vector_store()
    results = run_evaluation(vectorstore)
    report = generate_report(results)
    print("\nBaseline evaluation complete!")