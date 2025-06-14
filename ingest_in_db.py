# ─── Section 1: Imports & Environment Setup ────────────────────────────────────
import os
from dotenv import load_dotenv

from supabase.client import create_client, Client

from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# ─── Section 2: Supabase & Embedding Model Initialization ────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Use a small embedding model for converting text → vectors
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ─── Section 3: Document Loading & Chunking ───────────────────────────────────
def load_and_chunk_documents(
    pdf_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> list:
    """
    Load all PDFs from `pdf_dir`, split them into overlapping text chunks,
    and return a list of Document objects.
    """
    # 1. Load PDFs from the directory
    pdf_loader = PyPDFDirectoryLoader(pdf_dir)
    raw_docs = pdf_loader.load()

    # 2. Split into smaller overlapping chunks for better retrieval quality
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(raw_docs)

# ─── Section 4: Ingest into Supabase Vector Store ─────────────────────────────
def ingest_to_vector_store(
    documents: list,
    table_name: str = "documents",
    query_name: str = "match_documents",
    chunk_size: int = 1000
):
    """
    Take pre-split documents and upsert them into a SupabaseVectorStore.
    """
    vector_store = SupabaseVectorStore.from_documents(
        documents,
        embedding_model,
        client=supabase_client,
        table_name=table_name,
        query_name=query_name,
        chunk_size=chunk_size,
    )
    return vector_store

# ─── Section 5: Script Entry Point ────────────────────────────────────────────
def main():
    # Directory containing your PDF files
    PDF_DIRECTORY = "documents"

    # Load, chunk, and ingest
    docs = load_and_chunk_documents(pdf_dir=PDF_DIRECTORY)
    ingest_to_vector_store(documents=docs)

    print(f"Ingested {len(docs)} document chunks into Supabase.")

if __name__ == "__main__":
    main()
