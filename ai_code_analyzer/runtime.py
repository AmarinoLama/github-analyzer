"""Contexto de ejecución de larga vida.

LangGraph usa un *estado* (`AnalyzerState`) que representa los datos que
fluyen entre nodos **durante una ejecución** del grafo. Sin embargo hay
objetos que queremos conservar **entre ejecuciones** (el vector store ya
indexado, el LLM, los archivos leídos...). Para eso usamos `Runtime`, un
simple contenedor que la CLI mantiene vivo durante toda la sesión.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings

from ai_code_analyzer.repo import FileInfo


@dataclass
class Runtime:
    """Estado de la aplicación que persiste entre invocaciones del grafo."""

    llm: Any  # OpenCodeLLM (o un modelo compatible de LangChain)
    embeddings: Embeddings

    # Resultado de la indexación
    vectorstore: Any = None
    repo_url: str | None = None
    repo_path: str | None = None
    project_name: str | None = None
    files: list[FileInfo] = field(default_factory=list)
    tech_summary: dict = field(default_factory=dict)
    documents: list = field(default_factory=list)
    num_chunks: int = 0

    # Resultado del análisis completo
    semgrep_available: bool = False
    semgrep_findings: list[dict] = field(default_factory=list)
    analysis_summary: str = ""

    # Resultado del informe
    report_path: str | None = None

    @property
    def indexed(self) -> bool:
        """Indica si ya hay un repositorio indexado."""
        return self.vectorstore is not None and bool(self.files)

    @property
    def analyzed(self) -> bool:
        """Indica si ya se ha ejecutado el análisis completo."""
        return bool(self.analysis_summary)
