"""Nodos de LangGraph.

Un *nodo* en LangGraph no es más que una función Python que recibe el
estado y devuelve un diccionario con la parte del estado que actualiza.
Ningún nodo tiene "magia": el grafo simplemente decide en qué orden se
llaman y cómo se pasa el estado de uno a otro.

Los nodos se crean como *closures* sobre `runtime` y `settings`, de modo
que pueden acceder al LLM, al vector store y a la configuración sin usar
variables globales.
"""
from __future__ import annotations

from pathlib import Path

from ai_code_analyzer import analysis, indexing, rag, report, repo
from ai_code_analyzer.graph.state import AnalyzerState


# --------------------------------------------------------------------- #
# Nodos del grafo de indexación                                         #
# --------------------------------------------------------------------- #

def clone_node(runtime, settings):
    """Nodo 1: clona el repositorio (Python/subprocess, sin LLM)."""
    def node(state: AnalyzerState) -> dict:
        repo_url = state["repo_url"]
        project_name = repo.extract_project_name(repo_url)
        destination = settings.repos_dir / project_name

        repo.clone_repository(repo_url, destination)

        runtime.repo_url = repo_url
        runtime.repo_path = str(destination)
        runtime.project_name = project_name
        return {
            "repo_url": repo_url,
            "project_name": project_name,
            "repo_path": str(destination),
        }

    return node


def read_files_node(runtime, settings):
    """Nodo 2: lee y filtra los archivos relevantes del repositorio."""
    def node(state: AnalyzerState) -> dict:
        root = Path(state["repo_path"])
        files = repo.read_files(root, settings.max_file_kb)
        tech = repo.detect_technologies(root, files)

        runtime.files = files
        runtime.tech_summary = tech.to_dict()
        return {"num_files": len(files), "tech_summary": tech.to_dict()}

    return node


def chunk_node(runtime, settings):
    """Nodo 3: divide el código en chunks con metadatos."""
    def node(state: AnalyzerState) -> dict:
        documents = indexing.split_files(
            runtime.files, settings.chunk_size, settings.chunk_overlap
        )
        runtime.documents = documents
        return {"num_chunks": len(documents)}

    return node


def index_node(runtime, settings):
    """Nodo 4: crea embeddings y guarda los chunks en el vector store."""
    def node(state: AnalyzerState) -> dict:
        vectorstore = indexing.build_vectorstore(runtime.documents, runtime.embeddings)
        runtime.vectorstore = vectorstore
        runtime.num_chunks = len(runtime.documents)
        return {"num_chunks": len(runtime.documents)}

    return node


# --------------------------------------------------------------------- #
# Nodos del grafo de preguntas (RAG)                                    #
# --------------------------------------------------------------------- #

def retrieve_node(runtime, settings):
    """Nodo RAG 1: buscar los fragmentos más relevantes con el retriever."""
    def node(state: AnalyzerState) -> dict:
        question = state["question"]
        context, sources = rag.retrieve_context(
            runtime.vectorstore, question, k=settings.retrieval_k
        )
        return {"context": context, "sources": sources}

    return node


def answer_node(runtime, settings):
    """Nodo RAG 2: generar la respuesta a partir del contexto recuperado."""
    def node(state: AnalyzerState) -> dict:
        answer = rag.generate_answer(runtime.llm, state["context"], state["question"])
        return {"answer": answer}

    return node


# --------------------------------------------------------------------- #
# Nodos del grafo de análisis completo                                  #
# --------------------------------------------------------------------- #

def semgrep_node(runtime, settings):
    """Nodo A: ejecuta Semgrep (o el escáner heurístico de respaldo)."""
    def node(state: AnalyzerState) -> dict:
        root = Path(runtime.repo_path)
        available, findings, note = analysis.run_semgrep(
            root, settings.semgrep_config, settings.semgrep_timeout
        )
        if not available:
            findings = analysis.heuristic_scan(runtime.files)

        runtime.semgrep_available = available
        runtime.semgrep_findings = findings
        return {
            "semgrep_available": available,
            "num_findings": len(findings),
            "semgrep_note": note,
        }

    return node


def explain_node(runtime, settings):
    """Nodo B: el LLM explica los hallazgos en lenguaje natural."""
    def node(state: AnalyzerState) -> dict:
        structure = repo.summarize_structure(runtime.files)
        summary = analysis.explain_findings(
            runtime.llm,
            runtime.project_name,
            runtime.repo_url,
            runtime.tech_summary,
            structure,
            runtime.semgrep_findings,
        )
        runtime.analysis_summary = summary
        return {"analysis_summary": summary}

    return node


def report_node(runtime, settings):
    """Nodo C: genera el informe PDF a partir de los datos recogidos."""
    def node(state: AnalyzerState) -> dict:
        del state
        output = settings.reports_dir / f"{runtime.project_name}-analysis.pdf"
        data = {
            "project_name": runtime.project_name or "Repositorio",
            "repo_url": runtime.repo_url or "",
            "tech_summary": runtime.tech_summary,
            "structure": repo.summarize_structure(runtime.files),
            "analysis_summary": runtime.analysis_summary,
            "semgrep_findings": runtime.semgrep_findings,
        }
        path = report.generate_report(data, output)
        runtime.report_path = path
        return {"report_path": path}

    return node
