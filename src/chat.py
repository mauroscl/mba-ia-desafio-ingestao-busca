from config import COMMON_REQUIRED_ENV, get_env, validate_env
from search import search_prompt
from langchain_postgres import PGVector
from model_provider import create_embeddings

# Inicializar embeddings e store uma única vez no carregamento do módulo.
validate_env(COMMON_REQUIRED_ENV)
_embeddings = create_embeddings()
_pgvector_store = PGVector(
    embeddings=_embeddings,
    collection_name=get_env("PG_VECTOR_COLLECTION_NAME"),
    connection=get_env("DATABASE_URL"),
    use_jsonb=True,
)

def buscar_por_similaridade(pergunta):
    results = _pgvector_store.similarity_search_with_score(pergunta, k=10)

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