import os
import logging
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


def setup_logging() -> None:
    """Configura logging padrão para scripts CLI do projeto."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    log_level_name = get_optional_env("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    log_format = get_optional_env(
        "LOG_FORMAT",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.basicConfig(level=log_level, format=log_format)

    # Evita poluicao de logs com requisições HTTP de bibliotecas cliente.
    noisy_loggers = ("httpx", "httpcore")
    noisy_level_name = get_optional_env("LOG_HTTP_CLIENT_LEVEL", "WARNING").upper()
    noisy_level = getattr(logging, noisy_level_name, logging.WARNING)
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(noisy_level)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Auto-bootstrap: ao importar config, logging já fica disponível para todo o projeto.
if _is_truthy(get_optional_env("AUTO_SETUP_LOGGING", "true")):
    setup_logging()