import os

import pytest

import model_provider


def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_selected_provider_prefere_openai_quando_ambas_keys_existem(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    assert model_provider._selected_provider() == "openai"


def test_selected_provider_usa_google_quando_apenas_google_existe(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    assert model_provider._selected_provider() == "google"


def test_selected_provider_lanca_erro_quando_nao_ha_provider_configurado(monkeypatch):
    _clear_provider_env(monkeypatch)

    with pytest.raises(RuntimeError, match="No provider configured"):
        model_provider._selected_provider()


def test_create_chat_model_openai_usa_temperature_e_model(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-5-mini")

    captured = {}

    class DummyOpenAIChat:
        def __init__(self, model, temperature):
            captured["model"] = model
            captured["temperature"] = temperature

    monkeypatch.setattr(model_provider, "ChatOpenAI", DummyOpenAIChat)

    _ = model_provider.create_chat_model(temperature=0.3)

    assert captured == {"model": "gpt-5-mini", "temperature": 0.3}


def test_create_embeddings_google_usa_model_do_env(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")

    captured = {}

    class DummyGoogleEmbeddings:
        def __init__(self, model):
            captured["model"] = model

    monkeypatch.setattr(model_provider, "GoogleGenerativeAIEmbeddings", DummyGoogleEmbeddings)

    _ = model_provider.create_embeddings()

    assert captured["model"] == "models/embedding-001"
