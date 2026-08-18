"""Análisis completo: Semgrep + explicación con el LLM.

Semgrep es una herramienta de análisis estático que busca *patrones* de
código potencialmente peligrosos (SQL injection, eval, secretos...). Aquí
se ejecuta mediante subprocess (Python no necesita un LLM para esto) y,
después, el LLM recibe los resultados y los explica en lenguaje natural.

IMPORTANTE: el LLM debe presentar los resultados como *posibles hallazgos*
y apoyarse en Semgrep. Nunca debe garantizar que el código es seguro.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

ANALYSIS_SYSTEM = (
    "Eres un revisor de código senior. A continuación recibirás un resumen "
    "técnico de un repositorio y los hallazgos de un análisis estático "
    "(Semgrep). Redacta un informe claro en español con estas secciones: "
    "1) Resumen del proyecto, 2) Arquitectura probable, 3) Hallazgos "
    "explicados en lenguaje sencillo, 4) Recomendaciones concretas, "
    "5) Conclusión. Habla SIEMPRE de 'posibles vulnerabilidades/hallazgos' "
    "y aclara que el análisis no garantiza que el código sea seguro."
)

ANALYSIS_USER = """Repositorio: {project_name}
URL: {repo_url}

Tecnologías detectadas:
{tech_summary}

Estructura (carpetas principales):
{structure}

Hallazgos de Semgrep (puede estar vacío):
{semgrep_findings}
"""

# Patrones heurísticos usados SOLO cuando Semgrep no está instalado.
HEURISTIC_RULES: list[tuple[str, str, str, re.Pattern]] = [
    (
        "SQL injection (concatenación de consultas)",
        "posible",
        "MEDIUM",
        re.compile(r"(execute|executemany|query|raw)\s*\(\s*f?[\"'].*(\+\s*\w|\%\s*\w|\.format\()", re.IGNORECASE),
    ),
    (
        "Ejecución de código dinámico",
        "eval/exec puede ejecutar código arbitrario",
        "HIGH",
        re.compile(r"\b(eval|exec)\s*\(", re.IGNORECASE),
    ),
    (
        "Subprocess con shell=True",
        "shell=True permite inyección de comandos",
        "HIGH",
        re.compile(r"subprocess\..*shell\s*=\s*True", re.IGNORECASE),
    ),
    (
        "Posible secreto hardcodeado",
        "clave o token en el código fuente",
        "MEDIUM",
        re.compile(r"(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*[\"'][^\"']{6,}[\"']", re.IGNORECASE),
    ),
    (
        "Uso de pickle (deserialización insegura)",
        "cargar datos no confiables con pickle es peligroso",
        "MEDIUM",
        re.compile(r"\b(pickle\.loads?|cPickle\.loads?)\s*\(", re.IGNORECASE),
    ),
    (
        "Posible XSS (innerHTML)",
        "insertar HTML no saneado en el DOM",
        "MEDIUM",
        re.compile(r"\.innerHTML\s*=", re.IGNORECASE),
    ),
]


def _find_semgrep_executable() -> Optional[list[str]]:
    """Localiza el ejecutable de Semgrep (CLI o módulo de Python)."""
    cli = shutil.which("semgrep")
    if cli:
        return [cli]
    try:
        subprocess.run(
            [sys.executable, "-m", "semgrep", "--version"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return [sys.executable, "-m", "semgrep"]
    except Exception:
        return None


def run_semgrep(repo_path: Path, config: str, timeout: float) -> tuple[bool, list[dict], str]:
    """Ejecuta Semgrep y devuelve (disponible, hallazgos normalizados, nota)."""
    executable = _find_semgrep_executable()
    if executable is None:
        return False, [], "semgrep no está instalado; se usará el escáner heurístico."

    command = [
        *executable,
        "scan",
        "--json",
        "--config", config,
        "--metrics", "off",
        str(repo_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, [], "Semgrep agotó el tiempo; se usará el escáner heurístico."
    except OSError as exc:
        return False, [], f"No se pudo ejecutar Semgrep ({exc})."

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False, [], "La salida de Semgrep no era JSON válido."

    findings = _normalize_semgrep_results(payload)
    note = f"Semgrep ejecutado correctamente ({len(findings)} hallazgos)."
    return True, findings, note


def _normalize_semgrep_results(payload: dict) -> list[dict]:
    """Convierte la salida JSON de Semgrep a un formato homogéneo."""
    findings: list[dict] = []
    for result in payload.get("results", []):
        start = result.get("start", {})
        extra = result.get("extra", {})
        code = result.get("extra", {}).get("lines", "")
        findings.append(
            {
                "rule": result.get("check_id", "desconocido"),
                "message": extra.get("message", ""),
                "severity": (extra.get("severity") or "UNKNOWN").upper(),
                "path": result.get("path", "?"),
                "line": start.get("line"),
                "column": start.get("col"),
                "code": code.strip() if isinstance(code, str) else "",
            }
        )
    # Los hallazgos más graves primero.
    order = {"ERROR": 0, "HIGH": 1, "WARNING": 2, "MEDIUM": 3, "LOW": 4, "INFO": 5}
    findings.sort(key=lambda f: order.get(f["severity"], 9))
    return findings


def heuristic_scan(files) -> list[dict]:
    """Escáner de patrones propio usado cuando Semgrep no está disponible."""
    findings: list[dict] = []
    for file in files:
        for line_number, line in enumerate(file.content.splitlines(), start=1):
            for rule, message, severity, pattern in HEURISTIC_RULES:
                if pattern.search(line):
                    findings.append(
                        {
                            "rule": rule,
                            "message": message,
                            "severity": severity,
                            "path": file.path,
                            "line": line_number,
                            "column": None,
                            "code": line.strip()[:160],
                        }
                    )
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings.sort(key=lambda f: order.get(f["severity"], 9))
    return findings


def _findings_to_text(findings: list[dict], max_items: int = 40) -> str:
    """Serializa los hallazgos como texto legible para el prompt del LLM."""
    if not findings:
        return "(sin hallazgos)"
    lines = []
    for finding in findings[:max_items]:
        lines.append(
            f"- [{finding['severity']}] {finding['rule']} "
            f"en {finding['path']}:{finding['line']}"
        )
        if finding.get("message"):
            lines.append(f"    {finding['message']}")
    if len(findings) > max_items:
        lines.append(f"... y {len(findings) - max_items} hallazgos más.")
    return "\n".join(lines)


def explain_findings(
    llm,
    project_name: str,
    repo_url: str,
    tech_summary: dict,
    structure: list[str],
    findings: list[dict],
) -> str:
    """Pide al LLM un análisis narrativo a partir de los datos recogidos."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", ANALYSIS_SYSTEM), ("human", ANALYSIS_USER)]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "project_name": project_name,
            "repo_url": repo_url or "(desconocida)",
            "tech_summary": json.dumps(tech_summary, ensure_ascii=False, indent=2),
            "structure": "\n".join(structure) if structure else "(no disponible)",
            "semgrep_findings": _findings_to_text(findings),
        }
    )
