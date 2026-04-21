import os
import logging

logger = logging.getLogger(__name__)

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import get_optional_env


def _has_env(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip())


def _selected_provider() -> str:
    # Prefer OpenAI when both providers are configured.
    if _has_env("OPENAI_API_KEY"):
        return "openai"
    if _has_env("GOOGLE_API_KEY"):
        return "google"
    raise RuntimeError(
        "No provider configured. Set OPENAI_API_KEY or GOOGLE_API_KEY."
    )


def create_embeddings():
    provider = _selected_provider()
    if provider == "openai":
        return OpenAIEmbeddings(
            model=get_optional_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )

    return GoogleGenerativeAIEmbeddings(
        model=get_optional_env("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
    )


def create_chat_model(temperature: float = 0.5):
    provider = _selected_provider()
    if provider == "openai":
        open_api_model = get_optional_env("OPENAI_CHAT_MODEL", "gpt-5-mini")
        logger.info("Using chat model: %s", open_api_model)
      
        return ChatOpenAI(
            model=open_api_model,
            temperature=temperature,
        )

    google_model = get_optional_env("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
    logger.info("Using chat model: %s", google_model)
    return ChatGoogleGenerativeAI(
        model=google_model,
        temperature=temperature,
    )
