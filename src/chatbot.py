import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load API key
load_dotenv()

# Constants
CHROMA_DB_PATH = "db/chroma_db"

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

def create_qa_chain(vectorstore):
    # Custom prompt — keeps answers grounded in the document
    prompt_template = """
    You are a QA knowledge assistant trained on ISTQB Foundation syllabus.
    Answer the question using ONLY the context provided below.
    If the answer is not in the context, say "I don't have enough information in my knowledge base to answer this."
    Do not make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

def ask_question(qa_chain, question):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    result = qa_chain.invoke({"query": question})
    print(f"Answer: {result['result']}")
    print("\nSources:")
    for i, doc in enumerate(result['source_documents']):
        print(f"  [{i+1}] Page {doc.metadata.get('page', 'N/A')}")
    return result

if __name__ == "__main__":
    # Load knowledge base
    vectorstore = load_vector_store()
    qa_chain = create_qa_chain(vectorstore)

    # Test questions — your first QA test cases!
    questions = [
        "What is equivalence partitioning?",
        "What are the seven testing principles?",
        "What is the difference between verification and validation?",
        "What is a test oracle?"
    ]

    for question in questions:
        ask_question(qa_chain, question)
        print("\n" + "=" * 60)