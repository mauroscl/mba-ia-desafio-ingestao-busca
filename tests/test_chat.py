from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import chat


def test_parse_args_usa_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["chat.py"])

    args = chat.parse_args()

    assert args.temperature == chat.DEFAULT_TEMPERATURE
    assert args.k == chat.DEFAULT_TOP_K


@pytest.mark.parametrize(
    "argv",
    [
        ["chat.py", "--temperature", "-0.1"],
        ["chat.py", "--k", "0"],
    ],
)
def test_parse_args_rejeita_valores_invalidos(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        chat.parse_args()


def test_busca_por_similaridade_concatena_content_e_respeita_k(monkeypatch):
    captured = {}

    class DummyStore:
        def similarity_search_with_score(self, pergunta, k):
            captured["pergunta"] = pergunta
            captured["k"] = k
            return [
                (Document(page_content="  trecho 1  ", metadata={}), 0.1),
                (Document(page_content="trecho 2", metadata={}), 0.2),
            ]

    monkeypatch.setattr(chat, "get_pgvector_store", lambda: DummyStore())

    contexto = chat.buscar_por_similaridade("pergunta teste", k=7)

    assert captured == {"pergunta": "pergunta teste", "k": 7}
    assert contexto == "trecho 1\ntrecho 2"


def test_main_repassa_temperature_e_k_para_dependencias(monkeypatch, capsys):
    captured = {
        "temperature": None,
        "k": None,
        "invocation": None,
    }

    class DummyChain:
        def invoke(self, payload):
            captured["invocation"] = payload
            return SimpleNamespace(content="resposta teste")

    monkeypatch.setattr(chat, "search_prompt", lambda temperature: captured.__setitem__("temperature", temperature) or DummyChain())
    monkeypatch.setattr(chat, "get_env", lambda _: "collection_test")
    monkeypatch.setattr(
        chat,
        "buscar_por_similaridade",
        lambda pergunta, k: captured.__setitem__("k", k) or "contexto montado",
    )

    questions = iter(["qualquer pergunta", KeyboardInterrupt()])

    def fake_input(_):
        item = next(questions)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("builtins.input", fake_input)

    chat.main(temperature=0.2, k=9)

    assert captured["temperature"] == 0.2
    assert captured["k"] == 9
    assert captured["invocation"] == {"pergunta": "qualquer pergunta", "contexto": "contexto montado"}

    output = capsys.readouterr().out
    assert "Chat iniciado" in output
    assert "resposta teste" in output
