import os
from dotenv import load_dotenv

from search import search_prompt
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

def buscar_por_similaridade(pergunta):
    # implementar a função de busca por similaridade usando o PGVector e o modelo de embedding da OpenAI.
    # A função deve retornar o contexto mais relevante para a pergunta do usuário.
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

    store= PGVector(embeddings=embeddings, collection_name=str(os.getenv("PG_VECTOR_COLLECTION_NAME")), connection=os.getenv("DATABASE_URL"), use_jsonb=True)

    results = store.similarity_search_with_score(pergunta, k=3)

    # for i, (doc, score) in enumerate(results, start=1):
    #     print("="*50)
    #     print(f"Resultado {i} (score: {score:.2f}):")
    #     print("="*50)

    #     print("\nTexto:\n")
    #     print(doc.page_content.strip())

    #     print("\nMetadados:\n")
    #     for k, v in doc.metadata.items():
    #         print(f"{k}: {v}")
            
    # retornar a concatenação dos textos mais relevantes como contexto para a pergunta do usuário
    contexto = "\n\n".join([doc.page_content.strip() for doc, _ in results])
    return contexto            

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    question = input("Digite sua pergunta: ")
    
    # contexto deve ser obtido do PGVector
    contexto = buscar_por_similaridade(question)
    
    result = chain.invoke({"pergunta": question, "contexto": contexto})
    print(f"\n{'*'*15}Resposta do modelo:{'*'*15}\n")
    print(result.content)

if __name__ == "__main__":
    main()