from types import SimpleNamespace

import vector_store


def test_create_pgvector_store_usa_embeddings_injetados(monkeypatch):
    injected_embeddings = SimpleNamespace(name="injected")
    captured = {}

    class DummyPGVector:
        def __init__(self, embeddings, collection_name, connection, use_jsonb):
            captured["embeddings"] = embeddings
            captured["collection_name"] = collection_name
            captured["connection"] = connection
            captured["use_jsonb"] = use_jsonb

    monkeypatch.setattr(vector_store, "PGVector", DummyPGVector)
    monkeypatch.setattr(vector_store, "get_env", lambda key: {"PG_VECTOR_COLLECTION_NAME": "collection", "DATABASE_URL": "db-url"}[key])

    vector_store.create_pgvector_store(embeddings=injected_embeddings)

    assert captured["embeddings"] is injected_embeddings
    assert captured["collection_name"] == "collection"
    assert captured["connection"] == "db-url"
    assert captured["use_jsonb"] is True


def test_create_pgvector_store_cria_embeddings_quando_nao_fornecido(monkeypatch):
    created_embeddings = SimpleNamespace(name="created")
    captured = {}

    class DummyPGVector:
        def __init__(self, embeddings, collection_name, connection, use_jsonb):
            captured["embeddings"] = embeddings

    monkeypatch.setattr(vector_store, "PGVector", DummyPGVector)
    monkeypatch.setattr(vector_store, "create_embeddings", lambda: created_embeddings)
    monkeypatch.setattr(vector_store, "get_env", lambda key: {"PG_VECTOR_COLLECTION_NAME": "collection", "DATABASE_URL": "db-url"}[key])

    vector_store.create_pgvector_store()

    assert captured["embeddings"] is created_embeddings
