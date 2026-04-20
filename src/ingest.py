import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import INGEST_REQUIRED_ENV, get_env, get_optional_env, validate_env

def mostrar_pdf(documents:list[Document]):
    for doc in documents:
            print(f"Content: {doc.page_content[:100]}...")  # Print the first 100 characters of content
            print(f"Metadata: {doc.metadata}")  # Print metadata

def ingest_pdf():
    validate_env(INGEST_REQUIRED_ENV)

    PDF_PATH = get_env("PDF_PATH")

    if not os.path.isfile(PDF_PATH):
        print(f"File not found: {PDF_PATH}")
        return

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    documents = [Document(page_content=chunk.page_content, 
                          metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)}) 
                 for chunk in chunks]
    
    mostrar_pdf(documents)
    
    embeddings = OpenAIEmbeddings(model=get_optional_env("OPENAI_MODEL", "text-embedding-3-small"))
    
    store= PGVector(embeddings=embeddings, collection_name=get_env("PG_VECTOR_COLLECTION_NAME"), connection=get_env("DATABASE_URL"), use_jsonb=True)
    
    print(f"Storing {len(documents)} documents in the vector store...")
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    store.add_documents(documents, ids=ids)


if __name__ == "__main__":
    ingest_pdf()


