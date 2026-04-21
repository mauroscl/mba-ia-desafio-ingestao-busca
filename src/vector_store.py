from langchain_postgres import PGVector

from config import get_env
from model_provider import create_embeddings


def create_pgvector_store(embeddings=None) -> PGVector:
    if embeddings is None:
        embeddings = create_embeddings()

    return PGVector(
        embeddings=embeddings,
        collection_name=get_env("PG_VECTOR_COLLECTION_NAME"),
        connection=get_env("DATABASE_URL"),
        use_jsonb=True,
    )