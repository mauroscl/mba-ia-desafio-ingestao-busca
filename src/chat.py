from config import COMMON_REQUIRED_ENV, validate_env
from search import search_prompt
from vector_store import create_pgvector_store

_pgvector_store = None


def get_pgvector_store():
    global _pgvector_store
    if _pgvector_store is None:
        validate_env(COMMON_REQUIRED_ENV)
        _pgvector_store = create_pgvector_store()

    return _pgvector_store

def buscar_por_similaridade(pergunta):
    store = get_pgvector_store()
    results = store.similarity_search_with_score(pergunta, k=10)

    # retornar a concatenação dos textos mais relevantes como contexto para a pergunta do usuário
    return "\n".join([doc.page_content.strip() for doc, _ in results])

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Faça suas perguntas (Ctrl+C para sair).")

    try:
        while True:
            question = input("Digite sua pergunta: ").strip()
            if not question:
                continue

            # contexto deve ser obtido do PGVector
            contexto = buscar_por_similaridade(question)
            result = chain.invoke({"pergunta": question, "contexto": contexto})

            print(f"\n{'*'*15}Resposta do modelo:{'*'*15}\n")
            print(result.content)
            print(f"\n{'='*0}\n")
            print()
    except KeyboardInterrupt:
        print("\nEncerrando chat...")

if __name__ == "__main__":
    main()