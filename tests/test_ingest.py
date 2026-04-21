from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import ingest


def test_parse_args_usa_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ingest.py"])

    args = ingest.parse_args()

    assert args.chunk_size == ingest.DEFAULT_CHUNK_SIZE
    assert args.chunk_overlap == ingest.DEFAULT_CHUNK_OVERLAP


@pytest.mark.parametrize(
    "argv,expected_error",
    [
        (["ingest.py", "--chunk-size", "0"], "--chunk-size deve ser maior que 0."),
        (["ingest.py", "--chunk-overlap", "-1"], "--chunk-overlap deve ser maior ou igual a 0."),
        (
            ["ingest.py", "--chunk-size", "100", "--chunk-overlap", "100"],
            "--chunk-overlap deve ser menor que --chunk-size.",
        ),
    ],
)
def test_parse_args_rejeita_valores_invalidos(monkeypatch, argv, expected_error):
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        ingest.parse_args()


def test_normalizar_pdf_path_converte_windows_path_no_linux(monkeypatch):
    monkeypatch.setattr(ingest.os.path, "isfile", lambda _: False)
    monkeypatch.setattr(ingest.os, "name", "posix")

    result = ingest.normalizar_pdf_path(r"C:\Users\mauro\doc.pdf")

    assert result == "/mnt/c/Users/mauro/doc.pdf"


def test_ingest_pdf_armazena_documents_com_metadata_sanitizada_e_ids_prefixados(monkeypatch):
    monkeypatch.setattr(ingest, "validate_env", lambda _: None)

    env = {
        "PDF_PATH": "/tmp/document.pdf",
        "PG_VECTOR_COLLECTION_NAME": "minha_collection",
    }
    monkeypatch.setattr(ingest, "get_env", lambda key: env[key])
    monkeypatch.setattr(ingest, "normalizar_pdf_path", lambda path: path)
    monkeypatch.setattr(ingest.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(ingest.os, "getenv", lambda *_: "")

    loaded_docs = [Document(page_content="pagina 1", metadata={"page": 1})]

    class DummyLoader:
        def __init__(self, _):
            pass

        def load(self):
            return loaded_docs

    split_docs = [
        Document(page_content="chunk a", metadata={"source": "a", "empty": "", "none": None}),
        Document(page_content="chunk b", metadata={"source": "b"}),
    ]

    class DummySplitter:
        def __init__(self, chunk_size, chunk_overlap):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, _):
            return split_docs

    store_calls = {}

    class DummyStore:
        def add_documents(self, documents, ids):
            store_calls["documents"] = documents
            store_calls["ids"] = ids

    embeddings = SimpleNamespace(model="embed-model")

    monkeypatch.setattr(ingest, "PyPDFLoader", DummyLoader)
    monkeypatch.setattr(ingest, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(ingest, "create_embeddings", lambda: embeddings)
    monkeypatch.setattr(ingest, "create_pgvector_store", lambda embeddings: DummyStore())

    ingest.ingest_pdf(chunk_size=123, chunk_overlap=45)

    assert store_calls["ids"] == ["minha_collection_doc_0", "minha_collection_doc_1"]
    assert store_calls["documents"][0].metadata == {"source": "a"}
    assert store_calls["documents"][1].metadata == {"source": "b"}
