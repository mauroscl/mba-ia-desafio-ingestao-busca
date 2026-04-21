# Desafio MBA Engenharia de Software com IA - Ingestao e Busca Semantica

Projeto de RAG em linha de comando com:

- Python + LangChain
- PostgreSQL + pgvector
- Docker Compose para banco
- Ingestao de PDF e consulta semantica via CLI

O fluxo principal:

1. Ingestao: le o PDF, divide em chunks, gera embeddings e salva no PGVector.
2. Busca: recebe pergunta no terminal, busca trechos mais similares no banco e responde somente com base no contexto.

## Estrutura do projeto

```text
.
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py
│   ├── search.py
│   ├── chat.py
│   ├── model_provider.py
│   ├── vector_store.py
│   └── config.py
└── document.pdf
```

## Requisitos

- Python 3.11+
- Docker e Docker Compose
- API key de pelo menos um provedor:
	- OpenAI
	- Google Gemini

## Setup rapido

1. Criar e ativar ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Subir PostgreSQL + pgvector:

```bash
docker compose up -d
```

4. Criar arquivo de ambiente local:

```bash
cp .env.example .env
```

5. Preencher o arquivo .env com seus valores reais.

## Variaveis de ambiente

Exemplo base no arquivo .env.example:

```dotenv
# Configure exatamente uma chave de provedor (se ambas existirem, OpenAI tem prioridade)
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
GOOGLE_CHAT_MODEL=gemini-2.5-flash

# OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5-mini

DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=doc_google_pdf
PDF_PATH=document.pdf
```

Observacoes:

- O arquivo .env real nao deve ser versionado.
- O arquivo .env.example deve ser versionado com placeholders.
- Se houver vazamento de chave, revogue e gere uma nova.

Modelos recomendados no enunciado do desafio:

- OpenAI embeddings: text-embedding-3-small
- OpenAI chat: gpt-5-nano
- Google embeddings: models/embedding-001
- Google chat: gemini-2.5-flash-lite

Se quiser seguir exatamente o enunciado, ajuste esses valores no seu .env.

## Regra de selecao de provedor

Implementada em src/model_provider.py:

- Se OPENAI_API_KEY existir, usa OpenAI.
- Senao, se GOOGLE_API_KEY existir, usa Google.
- Se nenhuma chave existir, a execucao falha.

## Ordem de execucao

1. Subir banco:

```bash
docker compose up -d
```

2. Rodar ingestao:

```bash
python src/ingest.py
```

3. Rodar chat:

```bash
python src/chat.py
```

Se o comando python nao existir no PATH, use python3 ou venv/bin/python.

## Ingestao

Script: src/ingest.py

### Comando padrao

```bash
python src/ingest.py
```

### Parametros opcionais

- --chunk-size (default: 1000)
- --chunk-overlap (default: 200)

Exemplo:

```bash
python src/ingest.py --chunk-size 1200 --chunk-overlap 100
```

### Ajuda

```bash
python src/ingest.py --help
```

### Observacao sobre o desafio

No enunciado, o overlap solicitado e 150. Para aderencia estrita ao desafio, rode:

```bash
python src/ingest.py --chunk-size 1000 --chunk-overlap 150
```

## Chat CLI

Script: src/chat.py

### Comando padrao

```bash
python src/chat.py
```

### Parametros opcionais

- --temperature (default: 0.5)
- --k (default: 10)

Exemplo:

```bash
python src/chat.py --temperature 0.3 --k 8
```

### Ajuda

```bash
python src/chat.py --help
```

### Comportamento esperado

- A resposta deve ser baseada somente no contexto recuperado do banco vetorial.
- Em perguntas fora de contexto, a resposta esperada e:

"Nao tenho informacoes necessarias para responder sua pergunta."

## Prompt utilizado

O prompt esta em src/search.py e aplica as regras do enunciado:

- Responder apenas com base no CONTEXTO.
- Nao inventar.
- Retornar a frase de fallback quando faltar informacao.

## Troubleshooting

### 1) Erro de arquivo nao encontrado no PDF_PATH

Se estiver rodando em Linux/WSL e informar caminho Windows (exemplo C:\\...),
o script tenta converter automaticamente para /mnt/<drive>/... .

Exemplo:

- C:\Users\mauro\Documentos\arquivo.pdf
- /mnt/c/Users/mauro/Documentos/arquivo.pdf

### 2) Erro 403 no provedor (project denied access)

Causa comum:

- problema de billing/projeto/chave no provedor.

Acoes sugeridas:

- validar billing do projeto
- gerar nova API key
- confirmar que a chave pertence ao projeto correto

### 3) Logs HTTP muito verbosos

O projeto ja reduz logs de httpx/httpcore para WARNING em src/config.py.

## Comandos uteis

```bash
# Subir banco
docker compose up -d

# Ingestao com parametros do desafio
python src/ingest.py --chunk-size 1000 --chunk-overlap 150

# Chat com configuracao customizada
python src/chat.py --temperature 0.4 --k 10

# Ajuda de cada script
python src/ingest.py --help
python src/chat.py --help
```

## Entregavel

Repositorio publico com:

- codigo fonte completo
- README com instrucoes claras de execucao
- estrutura proposta no desafio