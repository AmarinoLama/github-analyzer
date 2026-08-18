"""Configuración de la aplicación.

Toda la configuración se lee de variables de entorno (con valores por
defecto razonables). De esta forma el código no tiene "números mágicos"
ni rutas hardcodeadas y puede configurarse mediante un archivo `.env`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Raíz del proyecto (un nivel por encima del paquete).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Ajustes inmutables cargados desde el entorno."""

    # --- OpenCode Server ---
    opencode_base_url: str
    opencode_provider_id: str | None
    opencode_model_id: str | None
    opencode_agent: str | None
    opencode_username: str | None
    opencode_password: str | None
    opencode_timeout: float

    # --- Embeddings ---
    embeddings_provider: str
    hf_model_name: str

    # --- Indexación / RAG ---
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    max_file_kb: int

    # --- Semgrep ---
    semgrep_config: str
    semgrep_timeout: float

    # --- Rutas ---
    workspace_dir: Path
    repos_dir: Path
    reports_dir: Path


def _env(name: str, default: str | None = None) -> str | None:
    """Lee una variable de entorno recortando espacios en blanco."""
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_settings() -> Settings:
    """Construye la configuración leyendo las variables de entorno."""
    root = PROJECT_ROOT
    workspace = root / "workspace"
    reports = root / "reports"
    repos = workspace / "repos"

    # Crea las carpetas de trabajo si no existen.
    for directory in (workspace, repos, reports):
        directory.mkdir(parents=True, exist_ok=True)

    return Settings(
        opencode_base_url=_env("OPENCODE_BASE_URL", "http://127.0.0.1:4096"),
        opencode_provider_id=_env("OPENCODE_PROVIDER_ID"),
        opencode_model_id=_env("OPENCODE_MODEL_ID"),
        opencode_agent=_env("OPENCODE_AGENT"),
        opencode_username=_env("OPENCODE_SERVER_USERNAME", "opencode"),
        opencode_password=_env("OPENCODE_SERVER_PASSWORD"),
        opencode_timeout=float(_env("OPENCODE_TIMEOUT", "300")),
        embeddings_provider=(_env("EMBEDDINGS_PROVIDER", "auto") or "auto").lower(),
        hf_model_name=_env("HF_MODEL_NAME", "all-MiniLM-L6-v2"),
        chunk_size=int(_env("CHUNK_SIZE", "1000")),
        chunk_overlap=int(_env("CHUNK_OVERLAP", "200")),
        retrieval_k=int(_env("RETRIEVAL_K", "6")),
        max_file_kb=int(_env("MAX_FILE_KB", "512")),
        semgrep_config=_env("SEMGREP_CONFIG", "auto") or "auto",
        semgrep_timeout=float(_env("SEMGREP_TIMEOUT", "600")),
        workspace_dir=workspace,
        repos_dir=repos,
        reports_dir=reports,
    )
