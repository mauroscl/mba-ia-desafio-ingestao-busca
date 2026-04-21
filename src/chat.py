import argparse
import logging

from config import COMMON_REQUIRED_ENV, validate_env, get_env
from search import search_prompt
from vector_store import create_pgvector_store

_pgvector_store = None
logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_K = 10


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def get_pgvector_store():
    global _pgvector_store
    if _pgvector_store is None:
        validate_env(COMMON_REQUIRED_ENV)
        _pgvector_store = create_pgvector_store()

    return _pgvector_store

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat interativo com recuperacao via PGVector.",
        formatter_class=HelpFormatter,
        epilog=(
            "Exemplo:\n"
            "  python src/chat.py --temperature 0.3 --k 8"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperatura do modelo de chat.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Numero de documentos retornados na busca por similaridade.",
    )

    args = parser.parse_args()
    if args.temperature < 0:
        parser.error("--temperature deve ser maior ou igual a 0.")
    if args.k <= 0:
        parser.error("--k deve ser maior que 0.")

    return args


def buscar_por_similaridade(pergunta, k: int = DEFAULT_TOP_K):
    store = get_pgvector_store()
    results = store.similarity_search_with_score(pergunta, k=k)

    # retornar a concatenação dos textos mais relevantes como contexto para a pergunta do usuário
    return "\n".join([doc.page_content.strip() for doc, _ in results])

def main(temperature: float = DEFAULT_TEMPERATURE, k: int = DEFAULT_TOP_K):
    chain = search_prompt(temperature=temperature)

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    logger.info("Collection name: %s", get_env("PG_VECTOR_COLLECTION_NAME"))
    logger.info("Chat parameters: temperature=%s, k=%s", temperature, k)

    print("\n\nChat iniciado. Faça suas perguntas (Ctrl+C para sair).")
    
    try:
        while True:
            question = input("Digite sua pergunta: ").strip()
            if not question:
                continue

            # contexto deve ser obtido do PGVector
            contexto = buscar_por_similaridade(question, k=k)
            result = chain.invoke({"pergunta": question, "contexto": contexto})

            print(f"\n{'*'*15}Resposta do modelo:{'*'*15}\n")
            print(result.content)
            print(f"\n{'='*60}\n")
            print()
    except KeyboardInterrupt:
        print("\nEncerrando chat...")

if __name__ == "__main__":
    cli_args = parse_args()
    main(temperature=cli_args.temperature, k=cli_args.k)