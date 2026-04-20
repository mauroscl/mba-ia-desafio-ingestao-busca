from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config import OPENAI_REQUIRED_ENV, get_optional_env, validate_env

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt():
  # Retorna uma cadeia de busca usando o PROMPT_TEMPLATE e a questão do usuário.
  validate_env(OPENAI_REQUIRED_ENV)

  question_template = PromptTemplate(
    input_variables=["contexto", "pergunta"],
    template=PROMPT_TEMPLATE,
  )
  model = ChatOpenAI(
    model=get_optional_env("CHAT_MODEL", "gpt-5-mini"),
    temperature=0.5,
  )

  return question_template | model