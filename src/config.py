import os
from dotenv import load_dotenv

load_dotenv()

COMMON_REQUIRED_ENV = (
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
)

CHAT_REQUIRED_ENV = ()

INGEST_REQUIRED_ENV = COMMON_REQUIRED_ENV + (
    "PDF_PATH",
)


def get_env(name: str) -> str:
    if value := os.getenv(name):
        return value
    else:
        raise RuntimeError(f"Environment variable {name} is not set")


def get_optional_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value


def validate_env(required: tuple[str, ...]) -> None:
    for key in required:
        get_env(key)