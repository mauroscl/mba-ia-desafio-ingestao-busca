import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import INGEST_REQUIRED_ENV, get_env, validate_env
from model_provider import create_embeddings
import logging

logger = logging.getLogger(__name__)

def mostrar_pdf(documents:list[Document]):
    for doc in documents:
            print(f"Content: {doc.page_content[:100]}...")  # Print the first 100 characters of content
            print(f"Metadata: {doc.metadata}")  # Print metadata

def ingest_pdf():
    validate_env(INGEST_REQUIRED_ENV)

    PDF_PATH = get_env("PDF_PATH")

    if not os.path.isfile(PDF_PATH):
        logger.error("File not found: %s", PDF_PATH)
        return

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    logger.info("chunk_size: 1000, chunk_overlap: 200")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    documents = [Document(page_content=chunk.page_content, 
                          metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)}) 
                 for chunk in chunks]
    
    if os.getenv("INGEST_SHOW_PDF", "").lower() in ("1", "true", "yes", "on"):
        mostrar_pdf(documents)

    if not documents:
        logger.warning("No chunks generated from PDF. Nothing to ingest.")
        return

    embeddings = create_embeddings()
    
    collection_name = get_env("PG_VECTOR_COLLECTION_NAME")

    logger.info("Embedding model: %s", embeddings.model)
    logger.info("Target collection: %s", collection_name)
    
    store= PGVector(embeddings=embeddings, collection_name=collection_name, connection=get_env("DATABASE_URL"), use_jsonb=True)
    
    logger.info("Storing %d documents in the vector store...", len(documents))
    
    # IDs in langchain_pg_embedding are globally unique, not per collection.
    ids = [f"{collection_name}_doc_{i}" for i in range(len(documents))]
    store.add_documents(documents, ids=ids)


if __name__ == "__main__":
    ingest_pdf()


