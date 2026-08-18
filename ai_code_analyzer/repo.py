"""Operaciones sobre el repositorio (sin IA).

Aquí viven las tareas que Python puede hacer directamente: clonar con
`git`, recorrer el árbol de archivos, leer el código y detectar
tecnologías. Ninguna de estas funciones usa un LLM; el LLM solo entra en
juego más adelante para *comprender* el código ya indexado.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# Directorios que nunca nos interesan para el análisis.
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build",
    "target", "out", "coverage", "__pycache__", ".venv", "venv", "env",
    ".next", ".nuxt", ".idea", ".vscode", ".gradle", ".cache", "tmp",
    "temp", ".pytest_cache", ".mypy_cache", ".ruff_cache", "site-packages",
}

# Extensiones claramente binarias o no analizables como texto.
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".class", ".jar", ".war",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".mov",
    ".avi", ".mkv", ".wav", ".db", ".sqlite", ".lock", ".pyc", ".pyo",
}

# Nombre de ficheros que se ignoran directamente.
IGNORED_FILENAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"}

# Mapa extensión -> lenguaje (heurístico, suficiente para un informe).
LANGUAGE_BY_EXT = {
    ".py": "Python", ".pyw": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".jsx": "JavaScript (JSX)",
    ".ts": "TypeScript", ".tsx": "TypeScript (TSX)",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".c": "C", ".h": "C/C++ Header", ".cpp": "C++", ".cc": "C++",
    ".cs": "C#", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
    ".php": "PHP", ".swift": "Swift", ".scala": "Scala", ".dart": "Dart",
    ".sh": "Shell", ".bash": "Shell", ".ps1": "PowerShell", ".bat": "Batch",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".less": "LESS", ".vue": "Vue", ".svelte": "Svelte",
    ".sql": "SQL", ".md": "Markdown", ".rst": "reStructuredText",
    ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
    ".xml": "XML", ".ini": "INI", ".cfg": "Config", ".conf": "Config",
    ".gradle": "Gradle", ".groovy": "Groovy", ".tf": "Terraform",
    ".dockerfile": "Dockerfile", ".proto": "Protobuf", ".graphql": "GraphQL",
    ".lua": "Lua", ".r": "R", ".pl": "Perl", ".ex": "Elixir", ".exs": "Elixir",
}

# Marcadores que revelan frameworks/herramientas a partir de ficheros clave.
TECH_MARKERS = {
    "package.json": "Node.js / npm",
    "tsconfig.json": "TypeScript",
    "vite.config.js": "Vite", "vite.config.ts": "Vite",
    "webpack.config.js": "Webpack", "webpack.config.ts": "Webpack",
    "angular.json": "Angular",
    "next.config.js": "Next.js", "next.config.ts": "Next.js",
    "nuxt.config.js": "Nuxt", "nuxt.config.ts": "Nuxt",
    "tailwind.config.js": "Tailwind CSS", "tailwind.config.ts": "Tailwind CSS",
    "requirements.txt": "Python (pip)", "setup.py": "Python (setuptools)",
    "pyproject.toml": "Python (pyproject)", "Pipfile": "Python (pipenv)",
    "pom.xml": "Maven", "build.gradle": "Gradle", "build.gradle.kts": "Gradle",
    "go.mod": "Go modules", "Cargo.toml": "Rust (Cargo)",
    "Gemfile": "Ruby (Bundler)", "composer.json": "PHP (Composer)",
    "package.swift": "Swift Package Manager",
    "Dockerfile": "Docker", "docker-compose.yml": "Docker Compose",
    "docker-compose.yaml": "Docker Compose", "compose.yml": "Docker Compose",
    ".github/workflows": "GitHub Actions",
    "README.md": "Documentación (README)",
}


@dataclass
class FileInfo:
    """Metadatos y contenido de un archivo relevante del repositorio."""

    path: str
    language: str
    extension: str
    size: int
    lines: int
    content: str
    is_doc: bool = False


@dataclass
class TechSummary:
    """Resumen de tecnologías detectadas en el repositorio."""

    languages: dict[str, int] = field(default_factory=dict)
    technologies: list[str] = field(default_factory=list)
    total_files: int = 0
    total_lines: int = 0

    def to_dict(self) -> dict:
        return {
            "languages": self.languages,
            "technologies": self.technologies,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
        }


def extract_project_name(repo_url: str) -> str:
    """Extrae `owner/repo` (o solo `repo`) a partir de la URL de GitHub."""
    cleaned = repo_url.strip().rstrip("/")
    # Rutas locales (Windows o POSIX): usamos el nombre de la carpeta.
    local = Path(cleaned)
    if local.exists() or re.match(r"^[A-Za-z]:[\\/]", cleaned):
        return local.name or "repo"
    # git@github.com:owner/repo.git
    if cleaned.startswith("git@"):
        match = re.search(r"[:/]([\w.-]+/[\w.-]+?)(?:\.git)?$", cleaned)
        if match:
            return match.group(1).replace("/", "__")
    else:
        path = urlparse(cleaned).path.strip("/")
        if path:
            parts = [p for p in path.split("/") if p]
            name = "__".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            return re.sub(r"\.git$", "", name)
    return "repo"


def _remove_readonly(func, path, exc_info):
    """Handler de `shutil.rmtree` para poder borrar archivos de solo lectura.

    En Windows los objetos de `.git` se crean con el atributo de solo
    lectura, lo que hace que `rmtree` falle en silencio y deje carpetas
    residuales que luego impiden volver a clonar.
    """
    del exc_info
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def clone_repository(repo_url: str, destination: Path) -> None:
    """Clona el repositorio (o copia una carpeta local) en `destination`."""
    if destination.exists():
        shutil.rmtree(destination, onerror=_remove_readonly)

    source = Path(repo_url)
    if source.is_dir():
        # Permite analizar una carpeta local sin red (útil para pruebas).
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git"))
        return

    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(destination)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Mostramos el error real de git para que el usuario sepa qué pasó
        # (repo inexistente/privado, sin red, ruta ocupada, etc.).
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git clone falló: {detail or 'error desconocido'}")


def file_language(path: Path) -> str:
    """Devuelve el lenguaje a partir del nombre del archivo."""
    name = path.name.lower()
    if name == "dockerfile":
        return "Dockerfile"
    if name == "makefile":
        return "Makefile"
    return LANGUAGE_BY_EXT.get(path.suffix.lower(), "Desconocido")


def is_relevant_file(path: Path, max_file_kb: int) -> bool:
    """Decide si un archivo merece ser leído e indexado."""
    if any(part in IGNORED_DIRS for part in path.parts):
        return False
    if path.name in IGNORED_FILENAMES:
        return False
    if path.suffix.lower() in IGNORED_EXTENSIONS:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return size <= max_file_kb * 1024


def read_files(root: Path, max_file_kb: int, max_files: int = 1500) -> list[FileInfo]:
    """Recorre el repositorio y devuelve los archivos relevantes leídos."""
    files: list[FileInfo] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_relevant_file(path, max_file_kb):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        relative = path.relative_to(root).as_posix()
        files.append(
            FileInfo(
                path=relative,
                language=file_language(path),
                extension=path.suffix.lower(),
                size=path.stat().st_size,
                lines=content.count("\n") + 1,
                content=content,
                is_doc=path.suffix.lower() in {".md", ".rst"},
            )
        )
        if len(files) >= max_files:
            break
    return files


def detect_technologies(root: Path, files: list[FileInfo]) -> TechSummary:
    """Detecta lenguajes y frameworks de forma heurística (sin LLM)."""
    summary = TechSummary()

    for file in files:
        if file.is_doc:
            continue
        summary.languages[file.language] = summary.languages.get(file.language, 0) + 1
        summary.total_lines += file.lines
    summary.total_files = len(files)

    # Lenguajes ordenados por número de archivos.
    summary.languages = dict(
        sorted(summary.languages.items(), key=lambda item: item[1], reverse=True)
    )

    # Marcos/herramientas a partir de la presencia de ficheros o carpetas.
    for marker, label in TECH_MARKERS.items():
        candidate = root / marker
        if candidate.exists():
            summary.technologies.append(label)

    # Heurísticas rápidas sobre dependencias declaradas.
    summary.technologies.extend(_tech_from_requirements(root))
    summary.technologies.extend(_tech_from_package_json(root))

    # Sin duplicados, conservando el orden.
    summary.technologies = list(dict.fromkeys(summary.technologies))
    return summary


def _tech_from_requirements(root: Path) -> list[str]:
    """Busca frameworks Python conocidos en requirements*.txt / pyproject."""
    patterns = {
        r"\bdjango\b": "Django",
        r"\bflask\b": "Flask",
        r"\bfastapi\b": "FastAPI",
        r"\bsqlalchemy\b": "SQLAlchemy",
        r"\bcelery\b": "Celery",
        r"\bpandas\b": "Pandas",
        r"\bnumpy\b": "NumPy",
        r"\bscikit-learn\b": "scikit-learn",
        r"\btensorflow\b": "TensorFlow",
        r"\btorch\b": "PyTorch",
    }
    found: list[str] = []
    for name in ("requirements.txt", "requirements.in", "pyproject.toml"):
        path = root / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        for pattern, label in patterns.items():
            if re.search(pattern, text) and label not in found:
                found.append(label)
    return found


def _tech_from_package_json(root: Path) -> list[str]:
    """Busca frameworks JS conocidos en package.json."""
    patterns = {
        r'"react"': "React",
        r'"vue"': "Vue",
        r'"svelte"': "Svelte",
        r'"next"': "Next.js",
        r'"express"': "Express",
        r'"nestjs"': "NestJS",
        r'"@angular/core"': "Angular",
        r'"typescript"': "TypeScript",
        r'"tailwindcss"': "Tailwind CSS",
    }
    path = root / "package.json"
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return []
    return [label for pattern, label in patterns.items() if re.search(pattern, text)]


def summarize_structure(files: list[FileInfo], max_dirs: int = 30) -> list[str]:
    """Resume la estructura del proyecto como una lista `directorio/ (n archivos)`."""
    counts: dict[str, int] = {}
    for file in files:
        parts = file.path.split("/")
        top = parts[0] if len(parts) > 1 else "(raíz)"
        counts[top] = counts.get(top, 0) + 1

    lines = [f"{directory}/ ({count} archivos)" for directory, count in sorted(counts.items())]
    return lines[:max_dirs]
