"""Ensamblado de los grafos de LangGraph.

Un `StateGraph` define los nodos y las transiciones entre ellos. Para la
primera versión usamos **flujos controlados** (líneas fijas), que son
fáciles de leer y explicar. Un agente autónomo (donde el LLM decide qué
herramienta usar en cada paso) sería una evolución posterior.

Grafos:

    Indexación:   START -> clone -> read -> chunk -> index -> END
    Preguntas:    START -> retrieve -> answer -> END
    Análisis:     START -> semgrep -> explain -> END
    Informe:      START -> report -> END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai_code_analyzer.graph import nodes
from ai_code_analyzer.graph.state import AnalyzerState


def build_indexing_graph(runtime, settings):
    """Grafo que clona, indexa y guarda el repositorio en el vector store."""
    graph = StateGraph(AnalyzerState)

    graph.add_node("clone", nodes.clone_node(runtime, settings))
    graph.add_node("read", nodes.read_files_node(runtime, settings))
    graph.add_node("chunk", nodes.chunk_node(runtime, settings))
    graph.add_node("index", nodes.index_node(runtime, settings))

    graph.add_edge(START, "clone")
    graph.add_edge("clone", "read")
    graph.add_edge("read", "chunk")
    graph.add_edge("chunk", "index")
    graph.add_edge("index", END)

    return graph.compile()


def build_qa_graph(runtime, settings):
    """Grafo que responde una pregunta usando RAG."""
    graph = StateGraph(AnalyzerState)

    graph.add_node("retrieve", nodes.retrieve_node(runtime, settings))
    graph.add_node("answer", nodes.answer_node(runtime, settings))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def build_analysis_graph(runtime, settings):
    """Grafo que ejecuta Semgrep y pide al LLM que explique los hallazgos."""
    graph = StateGraph(AnalyzerState)

    graph.add_node("semgrep", nodes.semgrep_node(runtime, settings))
    graph.add_node("explain", nodes.explain_node(runtime, settings))

    graph.add_edge(START, "semgrep")
    graph.add_edge("semgrep", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


def build_report_graph(runtime, settings):
    """Grafo que genera el informe PDF a partir del análisis ya realizado."""
    graph = StateGraph(AnalyzerState)

    graph.add_node("report", nodes.report_node(runtime, settings))

    graph.add_edge(START, "report")
    graph.add_edge("report", END)

    return graph.compile()
