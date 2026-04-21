import os
import logging
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import COMMON_REQUIRED_ENV, get_env, validate_env
from model_provider import create_embeddings
from vector_store import create_pgvector_store

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

def normalizar_pdf_path(pdf_path: str) -> str:
    if os.path.isfile(pdf_path):
        return pdf_path

    # Converte caminho Windows (C:\...) para /mnt/c/... quando rodando em Linux/WSL.
    windows_path_match = re.match(r"^([a-zA-Z]):[\\/](.*)$", pdf_path)
    if os.name != "nt" and windows_path_match:
        drive = windows_path_match[1].lower()
        rest = windows_path_match[2].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    return pdf_path

def mostrar_pdf(documents:list[Document]):
    for doc in documents:
            print(f"Content: {doc.page_content[:100]}...")  # Print the first 100 characters of content
            print(f"Metadata: {doc.metadata}")  # Print metadata

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingestao de PDF para PGVector.",
        formatter_class=HelpFormatter,
        epilog=(
            "Exemplo:\n"
            "  python src/ingest.py --chunk-size 1200 --chunk-overlap 100"
        ),
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Caminho do PDF para ingestao. Se informado, tem prioridade sobre PDF_PATH do ambiente.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Tamanho do chunk usado na divisao do documento.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Quantidade de sobreposicao entre chunks consecutivos.",
    )

    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size deve ser maior que 0.")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap deve ser maior ou igual a 0.")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap deve ser menor que --chunk-size.")

    return args


def ingest_pdf(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    pdf_path: str | None = None,
):
    validate_env(COMMON_REQUIRED_ENV)

    pdf_path_from_input = pdf_path or get_env("PDF_PATH")
    normalized_pdf_path = normalizar_pdf_path(pdf_path_from_input)

    if not os.path.isfile(normalized_pdf_path):
        logger.error("File not found: %s", normalized_pdf_path)
        return

    loader = PyPDFLoader(normalized_pdf_path)
    documents = loader.load()
    logger.info("chunk_size: %d, chunk_overlap: %d", chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    ingestion_timestamp = datetime.now(timezone.utc).isoformat()
    source_file = str(Path(normalized_pdf_path).resolve())
    source_filename = Path(normalized_pdf_path).name

    documents = []
    for chunk_index, chunk in enumerate(chunks):
        sanitized_metadata = {
            k: v for k, v in chunk.metadata.items() if v not in ("", None)
        } | {
            "source_file": source_file,
            "source_filename": source_filename,
            "chunk_index": chunk_index,
            "ingestion_timestamp": ingestion_timestamp,
        }
        documents.append(Document(page_content=chunk.page_content, metadata=sanitized_metadata))

    if not documents:
        logger.warning("No chunks generated from PDF. Nothing to ingest.")
        return

    if os.getenv("INGEST_SHOW_PDF", "").lower() in ("1", "true", "yes", "on"):
        mostrar_pdf(documents)

    embeddings = create_embeddings()

    collection_name = get_env("PG_VECTOR_COLLECTION_NAME")

    logger.info("Embedding model: %s", embeddings.model)
    logger.info("Target collection: %s", collection_name)

    store = create_pgvector_store(embeddings=embeddings)

    logger.info("Storing %d documents in the vector store...", len(documents))

    ids = [str(uuid4()) for _ in range(len(documents))]
    store.add_documents(documents, ids=ids)


if __name__ == "__main__":
    cli_args = parse_args()
    ingest_pdf(
        chunk_size=cli_args.chunk_size,
        chunk_overlap=cli_args.chunk_overlap,
        pdf_path=cli_args.pdf_path,
    )


