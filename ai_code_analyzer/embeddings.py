"""Generación de embeddings.

Un *embedding* es una representación numérica (vector) de un texto. Dos
textos semánticamente parecidos producen vectores cercanos, lo que permite
buscar por significado en lugar de por palabras exactas.

Usamos la interfaz `Embeddings` de LangChain, de modo que el vector store y
el retriever funcionan igual sea cual sea el motor concreto:

* `huggingface` -> sentence-transformers (embeddings semánticos reales,
  requiere descargar un modelo pequeño la primera vez).
* `hash`        -> fallback determinista en Python puro (n-gramas de
  caracteres convertidos en vector). NO es semántico, pero permite probar
  todo el pipeline sin descargas ni claves de API.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """Embeddings deterministas y sin dependencias (solo para pruebas).

    Convierte n-gramas de caracteres y palabras en un vector usando una
    función hash. Textos que comparten muchos términos producen vectores
    cercanos (similitud léxica), aunque no entienden sinónimos.
    """

    dimension: int = 256

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension

        # Palabras completas + n-gramas de caracteres (2, 3 y 4).
        tokens = [word.lower() for word in text.split()]
        for n in (2, 3, 4):
            tokens.extend(text[i : i + n].lower() for i in range(len(text) - n + 1))

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
            value = int(digest, 16)
            index = value % self.dimension
            sign = 1.0 if (value >> 8) & 1 else -1.0
            vector[index] += sign

        # Normalizamos para que la similitud coseno sea estable.
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def _try_huggingface(settings) -> Optional[Embeddings]:
    """Intenta crear embeddings semánticos con sentence-transformers."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.hf_model_name)
    except Exception as exc:  # ImportError, descarga fallida, falta de torch...
        print(f"  [!] No se pudieron cargar los embeddings de HuggingFace: {exc}")
        print("      Usando fallback de hashing (sin semántica real).")
        return None


def build_embeddings(settings) -> tuple[Embeddings, str]:
    """Crea el motor de embeddings según la configuración.

    Devuelve una tupla `(embeddings, nombre_del_proveedor)`.
    """
    provider = settings.embeddings_provider
    if provider in ("auto", "huggingface"):
        embeddings = _try_huggingface(settings)
        if embeddings is not None:
            return embeddings, "huggingface"
        if provider == "huggingface":
            # El usuario pidió HuggingFace explícitamente pero no está
            # disponible: degradamos igualmente para que la app funcione.
            print("      Degradando a fallback de hashing.")
    print("  [i] Embeddings: fallback de hashing (sin descargas).")
    return HashEmbeddings(), "hash"
