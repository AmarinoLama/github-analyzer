"""Estado de los grafos de LangGraph.

En LangGraph, un *estado* es un diccionario tipado que los nodos leen y
actualizan. Cada nodo devuelve un subconjunto del estado y LangGraph se
encarga de fusionarlo con el estado actual antes de pasar al siguiente
nodo.

Definimos un único `AnalyzerState` compartido por los tres grafos de la
aplicación (indexación, preguntas y análisis). `total=False` indica que no
todos los campos tienen que estar presentes en todas las ejecuciones.
"""
from __future__ import annotations

from typing import TypedDict


class AnalyzerState(TypedDict, total=False):
    """Datos que fluyen entre los nodos durante una ejecución del grafo."""

    # --- Entrada: repositorio a analizar ---
    repo_url: str

    # --- Resultado de la indexación ---
    project_name: str
    repo_path: str
    num_files: int
    num_chunks: int
    tech_summary: dict

    # --- Preguntas (RAG) ---
    question: str
    context: str
    sources: list[str]
    answer: str

    # --- Análisis completo ---
    semgrep_available: bool
    semgrep_note: str
    num_findings: int
    analysis_summary: str

    # --- Informe ---
    report_path: str

    # --- Errores ---
    error: str
