"""Interfaz de línea de comandos.

La CLI es el "cliente" de la aplicación: muestra el menú, recoge la URL y
las preguntas del usuario y llama a los grafos de LangGraph. No contiene
lógica de negocio: esa vive en los nodos del grafo.
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

from ai_code_analyzer.config import load_settings
from ai_code_analyzer.embeddings import build_embeddings
from ai_code_analyzer.graph.workflows import (
    build_analysis_graph,
    build_indexing_graph,
    build_qa_graph,
    build_report_graph,
)
from ai_code_analyzer.llm import build_llm
from ai_code_analyzer.runtime import Runtime
from ai_code_analyzer.ui import progress

BANNER = r"""
================================
       AI CODE ANALYZER
================================
"""

MENU = """
[1] Analizar repositorio
[2] Hacer preguntas
[3] Análisis completo
[4] Generar informe PDF
[5] Salir
"""


def _line(char: str = "=", width: int = 56) -> str:
    return char * width


def _configure_console() -> None:
    """Fuerza UTF-8 en la salida para que ✓ y las tildes se vean bien.

    En consolas Windows antiguas puede ser necesario además `chcp 65001`.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _print_error(message: str) -> None:
    print(f"\n  [!] {message}\n")


def _check_opencode(llm) -> bool:
    """Comprueba si OpenCode Server está disponible."""
    try:
        info = llm.check_health()
        print(f"  [✓] OpenCode Server conectado (versión {info.get('version', '?')})")
        return True
    except Exception as exc:
        print(f"  [!] No se pudo conectar con OpenCode Server ({exc}).")
        print("      Arranca `opencode serve` para poder usar el LLM.")
        print("      Puedes indexar igualmente; las preguntas/análisis fallarán.")
        return False


def _print_index_result(state: dict) -> None:
    tech = state.get("tech_summary") or {}
    languages = list((tech.get("languages") or {}).keys())
    print(f"  [✓] Proyecto: {state.get('project_name')}")
    print(f"  [✓] Archivos indexados: {state.get('num_files')}")
    print(f"  [✓] Chunks creados: {state.get('num_chunks')}")
    if languages:
        print(f"  [✓] Lenguajes: {', '.join(languages[:6])}")
    print("\n  Repositorio listo.")


def _run_indexing(graph, runtime, url: str) -> None:
    try:
        with progress("Analizando repositorio"):
            state = graph.invoke({"repo_url": url})
    except Exception as exc:
        _print_error(f"Error durante la indexación: {exc}")
        return
    _print_index_result(state)


def _run_questions(graph, runtime) -> None:
    print(_line())
    print("                CHAT")
    print(_line())
    print("  Escribe tu pregunta. 'salir' para volver al menú.\n")
    while True:
        try:
            question = input("  Tú:\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"salir", "exit", "quit", "q"}:
            return
        if not question:
            continue

        try:
            with progress("Pensando"):
                state = graph.invoke({"question": question})
        except Exception as exc:
            _print_error(f"No se pudo responder (¿está OpenCode Server activo?): {exc}")
            continue

        print("\n  IA:")
        for line in (state.get("answer") or "").splitlines():
            print(f"  > {line}")
        sources = state.get("sources") or []
        if sources:
            print("\n  Fuentes:")
            for source in sources[:5]:
                print(f"    - {source}")
        print()


def _run_analysis(graph, runtime) -> None:
    try:
        with progress("Ejecutando análisis completo"):
            state = graph.invoke({})
    except Exception as exc:
        _print_error(f"Error durante el análisis: {exc}")
        return

    note = state.get("semgrep_note")
    if note:
        print(f"  [i] {note}")
    print("  [✓] Repositorio analizado")
    print("  [✓] Tecnologías detectadas")
    print("  [✓] Estructura analizada")
    print(f"  [✓] Semgrep ejecutado ({'disponible' if state.get('semgrep_available') else 'escáner heurístico'})")
    print(f"  [✓] Hallazgos: {state.get('num_findings')}")
    print("  [✓] Resultados procesados por IA")

    findings = runtime.semgrep_findings
    for finding in findings[:10]:
        print(
            f"\n  {finding['severity']} | {finding['rule']}\n"
            f"    Archivo: {finding['path']}:{finding.get('line')}\n"
            f"    {finding.get('message', '')}".rstrip()
        )
    print("\n  Análisis detallado:\n")
    for line in (state.get("analysis_summary") or "").splitlines():
        print(f"  {line}")


def _run_report(graph, runtime) -> None:
    try:
        with progress("Generando informe"):
            state = graph.invoke({})
    except Exception as exc:
        _print_error(f"Error al generar el informe: {exc}")
        return
    print("  [✓] Resumen del proyecto")
    print("  [✓] Tecnologías detectadas")
    print("  [✓] Estructura del proyecto")
    print("  [✓] Análisis del código")
    print("  [✓] Vulnerabilidades")
    print("  [✓] Recomendaciones")
    print(f"\n  Archivo:\n  {state.get('report_path')}\n")


def main() -> None:
    _configure_console()
    load_dotenv()
    settings = load_settings()

    print(BANNER)

    # Capa de IA: LLM + embeddings.
    llm = build_llm(settings)
    embeddings, _provider = build_embeddings(settings)
    runtime = Runtime(llm=llm, embeddings=embeddings)

    # Grafos de LangGraph (orquestación).
    indexing_graph = build_indexing_graph(runtime, settings)
    qa_graph = build_qa_graph(runtime, settings)
    analysis_graph = build_analysis_graph(runtime, settings)
    report_graph = build_report_graph(runtime, settings)

    _check_opencode(llm)

    url = input("\n  Introduce la URL del repositorio:\n  > ").strip()

    while True:
        print(MENU)
        try:
            choice = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "1":
            if url:
                _run_indexing(indexing_graph, runtime, url)
            else:
                print("\n  Primero introduce una URL de repositorio.\n")

        elif choice == "2":
            if not runtime.indexed:
                print("\n  Primero analiza un repositorio (opción 1).\n")
                continue
            _run_questions(qa_graph, runtime)

        elif choice == "3":
            if not runtime.indexed:
                print("\n  Primero analiza un repositorio (opción 1).\n")
                continue
            _run_analysis(analysis_graph, runtime)

        elif choice == "4":
            if not runtime.indexed:
                print("\n  Primero analiza un repositorio (opción 1).\n")
                continue
            if not runtime.analyzed:
                print("\n  No hay análisis todavía. Ejecutando análisis completo primero...")
                _run_analysis(analysis_graph, runtime)
            _run_report(report_graph, runtime)

        elif choice == "5":
            print("\n  ¡Hasta luego!")
            break

        else:
            print("\n  Opción no válida. Elige entre 1 y 5.\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
