import os
from dotenv import load_dotenv

load_dotenv()

COMMON_REQUIRED_ENV = (
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
)

OPENAI_REQUIRED_ENV = (
    "OPENAI_API_KEY",
)

INGEST_REQUIRED_ENV = COMMON_REQUIRED_ENV + (
    "PDF_PATH",
)


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


def get_optional_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def validate_env(required: tuple[str, ...]) -> None:
    for key in required:
        get_env(key)