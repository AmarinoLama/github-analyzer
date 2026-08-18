"""Indicadores de progreso para la consola.

Muestra una barra de carga indeterminada mientras se ejecuta una operación
bloqueante (clonado, embeddings, llamada al LLM...). La animación corre en
un hilo en segundo plano y se detiene en cuanto termina el trabajo.

Si la salida no es una terminal (p. ej. redirigida a un archivo), se imprime
solo una línea estática para no ensuciar la salida.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

_BAR_WIDTH = 24
_TICK_SECONDS = 0.08
_FILL = "█"
_EMPTY = "░"


def _is_tty() -> bool:
    stream = getattr(sys, "stdout", None)
    return bool(stream and hasattr(stream, "isatty") and stream.isatty())


def _frame(elapsed: float) -> str:
    """Barra indeterminada que crece y decrece de forma cíclica."""
    cycle = 2 * _BAR_WIDTH
    position = int(elapsed / _TICK_SECONDS) % cycle
    filled = position if position <= _BAR_WIDTH else cycle - position
    return _FILL * filled + _EMPTY * (_BAR_WIDTH - filled)


@contextmanager
def progress(label: str) -> Iterator[None]:
    """Muestra una barra de carga animada mientras se ejecuta el bloque.

    Ejemplo:
        with progress("Analizando repositorio"):
            resultado = trabajo_largo()
    """
    if not _is_tty():
        print(f"  {label}...")
        yield
        return

    stop = threading.Event()
    started = time.monotonic()

    def _animate() -> None:
        while not stop.is_set():
            elapsed = time.monotonic() - started
            sys.stdout.write(f"\r  {label}... [{_frame(elapsed)}] {elapsed:3.0f}s")
            sys.stdout.flush()
            stop.wait(_TICK_SECONDS)

    thread = threading.Thread(target=_animate, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        # Limpia la línea dejando el cursor al principio para la salida real.
        # 2 (indent) + label + 3 ("...") + 1 + [24] + 1 + 1 + 4 ("NNNs") = label + 37
        width = len(label) + _BAR_WIDTH + 13
        sys.stdout.write("\r" + " " * width + "\r")
        sys.stdout.flush()
