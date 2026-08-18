"""RAG: Retrieval-Augmented Generation.

En lugar de enviar todo el repositorio al LLM en cada pregunta, seguimos
este flujo:

    pregunta
      -> retriever (busca los chunks más parecidos en el vector store)
      -> contexto (solo los fragmentos relevantes)
      -> prompt (contexto + pregunta)
      -> LLM
      -> respuesta + fuentes

Las funciones están separadas en dos pasos para que LangGraph pueda
representarlos como dos *nodos* distintos (`retrieve` y `answer`).
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ai_code_analyzer.indexing import format_context_doc

SYSTEM_PROMPT = (
    "Eres un ingeniero de software senior que ayuda a entender un repositorio "
    "de código. Responde ÚNICAMENTE basándote en el contexto de código "
    "proporcionado. Si el contexto no contiene la respuesta, dilo claramente "
    "y no inventes detalles. Cita los archivos relevantes (rutas) cuando sea "
    "útil. Responde en el idioma en el que se hizo la pregunta."
)

USER_TEMPLATE = """Contexto del repositorio (fragmentos relevantes):

{context}

Pregunta del usuario:
{question}

Respuesta:"""


def retrieve_context(vectorstore, question: str, k: int = 6) -> tuple[str, list[str]]:
    """Paso 1 del RAG: recuperar el contexto relevante con el retriever."""
    # `as_retriever` convierte el vector store en un *retriever* de LangChain:
    # un objeto cuya única responsabilidad es "dame documentos parecidos".
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    context = "\n\n".join(format_context_doc(doc) for doc in docs)
    sources = sorted({doc.metadata.get("source", "?") for doc in docs})
    return context, sources


def generate_answer(llm, context: str, question: str) -> str:
    """Paso 2 del RAG: generar la respuesta a partir del contexto."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE)]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def answer_question(vectorstore, llm, question: str, k: int = 6) -> tuple[str, list[str]]:
    """Versión cómoda que encadena los dos pasos. Devuelve (respuesta, fuentes)."""
    context, sources = retrieve_context(vectorstore, question, k)
    answer = generate_answer(llm, context, question)
    return answer, sources
