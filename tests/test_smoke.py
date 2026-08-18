"""Smoke test end-to-end (sin red, sin OpenCode Server, sin descargas).

Crea un repositorio de ejemplo en una carpeta temporal y recorre todo el
pipeline con:
  * `FakeLLM`            -> respuestas simuladas (sustituye a OpenCodeLLM).
  * `HashEmbeddings`     -> fallback determinista (sin modelo descargado).
  * Semgrep no instalado -> se usa el escáner heurístico.

Ejecutar:  python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ai_code_analyzer.config import Settings
from ai_code_analyzer.embeddings import HashEmbeddings
from ai_code_analyzer.graph.workflows import (
    build_analysis_graph,
    build_indexing_graph,
    build_qa_graph,
    build_report_graph,
)
from ai_code_analyzer.runtime import Runtime

CANONICAL_ANSWER = "RESPUESTA SIMULADA DEL LLM"


class FakeLLM(BaseChatModel):
    """Modelo de chat que devuelve siempre la misma respuesta."""

    response: str = CANONICAL_ANSWER

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        del stop, run_manager, kwargs, messages
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=self.response))
            ]
        )


def _make_settings(tmp: Path) -> Settings:
    return Settings(
        opencode_base_url="http://127.0.0.1:4096",
        opencode_provider_id=None,
        opencode_model_id=None,
        opencode_agent=None,
        opencode_username=None,
        opencode_password=None,
        opencode_timeout=10.0,
        embeddings_provider="hash",
        hf_model_name="all-MiniLM-L6-v2",
        chunk_size=300,
        chunk_overlap=50,
        retrieval_k=3,
        max_file_kb=512,
        semgrep_config="auto",
        semgrep_timeout=60.0,
        workspace_dir=tmp / "workspace",
        repos_dir=tmp / "workspace" / "repos",
        reports_dir=tmp / "reports",
    )


def _make_sample_repo(tmp: Path) -> Path:
    repo = tmp / "sample-repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text(
        "# Aplicación demo\n"
        "import subprocess\n\n"
        "def run(cmd):\n"
        "    subprocess.run(cmd, shell=True)  # patrón de ejemplo\n",
        encoding="utf-8",
    )
    (repo / "src" / "models.py").write_text(
        "class User:\n    def __init__(self, name):\n        self.name = name\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# Proyecto de ejemplo\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("flask==3.0.0\n", encoding="utf-8")
    return repo


def _check(condition: bool, label: str) -> None:
    status = "OK " if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise AssertionError(label)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    tmp = Path(tempfile.mkdtemp(prefix="ai-code-analyzer-test-"))
    settings = _make_settings(tmp)
    repo_dir = _make_sample_repo(tmp)

    runtime = Runtime(llm=FakeLLM(), embeddings=HashEmbeddings())
    indexing = build_indexing_graph(runtime, settings)
    qa = build_qa_graph(runtime, settings)
    analysis = build_analysis_graph(runtime, settings)
    report = build_report_graph(runtime, settings)

    # --- 1) Indexación -------------------------------------------------
    state = indexing.invoke({"repo_url": str(repo_dir)})
    _check(runtime.indexed, "El repositorio queda indexado")
    _check(runtime.vectorstore is not None, "Se crea el vector store")
    _check(len(runtime.files) == 4, "Se leen los 4 archivos del fixture")
    _check(state["num_chunks"] >= 4, "Se generan chunks para indexar")
    _check("Flask" in runtime.tech_summary.get("technologies", []), "Se detecta Flask")

    # --- 2) Preguntas (RAG) -------------------------------------------
    qa_state = qa.invoke({"question": "¿Qué hace este proyecto?"})
    _check(qa_state["answer"] == CANONICAL_ANSWER, "El nodo answer devuelve la respuesta del LLM")
    _check(len(qa_state["sources"]) > 0, "El retriever devuelve fuentes")

    # --- 3) Análisis completo -----------------------------------------
    analysis_state = analysis.invoke({})
    _check(analysis_state["num_findings"] >= 1, "El escáner heurístico encuentra hallazgos")
    _check(runtime.analysis_summary == CANONICAL_ANSWER, "El LLM produce el análisis narrativo")

    # --- 4) Informe PDF -----------------------------------------------
    report_state = report.invoke({})
    pdf = Path(report_state["report_path"])
    _check(pdf.exists() and pdf.stat().st_size > 0, "Se genera el informe PDF")

    print(f"\n  PDF generado en: {pdf}")
    print(f"  Todos los checks pasaron.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
