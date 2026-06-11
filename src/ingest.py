import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import time

# Load API key from .env
load_dotenv()

# Constants
PDF_PATH = "data/istqb_foundation.pdf"
CHROMA_DB_PATH = "db/chroma_db"

def load_pdf(path):
    print(f"📄 Loading PDF from {path}...")
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"✅ Loaded {len(pages)} pages")
    return pages

def split_documents(pages):
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    print("🧠 Creating embeddings and storing in ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    BATCH_SIZE = 50
    vectorstore = None
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        total_batches = -(-len(chunks)//BATCH_SIZE)
        print(f"📦 Processing batch {batch_num}/{total_batches}...")
        
        # Retry up to 3 times per batch
        for attempt in range(3):
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=CHROMA_DB_PATH
                    )
                else:
                    vectorstore.add_documents(batch)
                break  # Success — exit retry loop
                
            except Exception as e:
                if attempt < 2:
                    wait = 80 + (attempt * 30)  # 80s, then 110s
                    print(f"⚠️ Attempt {attempt+1} failed. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise e  # All 3 attempts failed
        
        if i + BATCH_SIZE < len(chunks):
            print("⏳ Waiting 80 seconds for rate limit...")
            time.sleep(80)
    
    print(f"✅ Vector store created at {CHROMA_DB_PATH}")
    return vectorstore

if __name__ == "__main__":
    try:
        pages = load_pdf(PDF_PATH)
        chunks = split_documents(pages)
        vectorstore = create_vector_store(chunks)
        print("🎉 Ingestion complete! Ready to query.")
    except Exception as e:
        print(f"❌ Error: {e}")