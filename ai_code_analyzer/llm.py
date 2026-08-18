"""Conexión con OpenCode Server a través de LangChain.

`opencode serve` expone una API HTTP *propia* (no compatible con OpenAI) en
http://127.0.0.1:4096. El endpoint relevante para nosotros es:

    POST /session            -> crea una sesión (devuelve { "id": ... })
    POST /session/{id}/message -> envía un mensaje y espera la respuesta

Para integrarlo con LangChain implementamos `BaseChatModel`, la clase base
que LangChain usa para "hablar" con cualquier modelo de chat. De este modo
el resto de la aplicación (cadenas RAG, prompts, LangGraph) no necesita
saber nada de los detalles de OpenCode: solo ve un `llm` estándar.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class OpenCodeLLM(BaseChatModel):
    """Modelo de chat de LangChain respaldado por OpenCode Server."""

    base_url: str = "http://127.0.0.1:4096"
    provider_id: Optional[str] = None
    model_id: Optional[str] = None
    agent: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: float = 300.0

    # Identificador de la sesión de OpenCode. El prefijo "_" hace que
    # LangChain lo trate como atributo privado (no como parámetro del modelo).
    _session_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # API interna de LangChain (obligatoria)                             #
    # ------------------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        """Identificador que LangChain usa para logging/serialización."""
        return "opencode-server"

    @property
    def _identifying_params(self) -> dict:
        """Parámetros que identifican a este modelo."""
        return {
            "base_url": self.base_url,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Convierte los mensajes de LangChain en una llamada a OpenCode."""
        del stop, run_manager, kwargs  # no usados en esta integración

        # OpenCode distingue un "system" global de las "parts" del mensaje.
        system_lines: list[str] = []
        text_lines: list[str] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                system_lines.append(str(message.content))
            else:
                role = "Assistant" if message.type == "ai" else "User"
                text_lines.append(f"{role}: {message.content}")

        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": "\n\n".join(text_lines)}],
        }
        if system_lines:
            body["system"] = "\n\n".join(system_lines)

        model = self._resolve_model()
        if model:
            body["model"] = model
        if self.agent:
            body["agent"] = self.agent

        session_id = self._ensure_session()
        response = self._post(f"/session/{session_id}/message", body)
        response.raise_for_status()
        data = response.json()

        text = self._extract_text(data)
        info = data.get("info") or {}
        tokens = info.get("tokens") or {}
        usage = {
            "input_tokens": tokens.get("input"),
            "output_tokens": tokens.get("output"),
            "total_tokens": (tokens.get("input") or 0) + (tokens.get("output") or 0),
        }

        generation_info = {"session_id": session_id, "model": model, **usage}
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text),
                    generation_info=generation_info,
                )
            ],
            llm_output=usage,
        )

    # ------------------------------------------------------------------ #
    # Comunicación HTTP con OpenCode Server                              #
    # ------------------------------------------------------------------ #

    def _auth(self) -> Optional[tuple[str, str]]:
        """Credenciales para la autenticación básica opcional del servidor."""
        username = self.username or os.getenv("OPENCODE_SERVER_USERNAME")
        password = self.password or os.getenv("OPENCODE_SERVER_PASSWORD")
        if password:
            return (username or "opencode", password)
        return None

    def _get(self, path: str) -> requests.Response:
        return requests.get(
            f"{self.base_url.rstrip('/')}{path}",
            auth=self._auth(),
            timeout=self.timeout,
        )

    def _post(self, path: str, json: dict) -> requests.Response:
        return requests.post(
            f"{self.base_url.rstrip('/')}{path}",
            json=json,
            auth=self._auth(),
            timeout=self.timeout,
        )

    def check_health(self) -> dict:
        """Consulta /global/health. Lanza una excepción si no hay conexión."""
        response = self._get("/global/health")
        response.raise_for_status()
        return response.json()

    def _ensure_session(self) -> str:
        """Crea (una sola vez) la sesión de OpenCode que usará el modelo."""
        if self._session_id:
            return self._session_id
        response = self._post("/session", {"title": "ai-code-analyzer"})
        response.raise_for_status()
        self._session_id = response.json().get("id")
        if not self._session_id:
            raise RuntimeError("OpenCode Server no devolvió un id de sesión.")
        return self._session_id

    def _resolve_model(self) -> Optional[dict]:
        """Devuelve {providerID, modelID} del modelo a usar.

        Prioridad:
        1. `provider_id` + `model_id` configurados explícitamente.
        2. El modelo por defecto anunciado por /config/providers.
        3. El primer modelo disponible del primer provider.
        """
        if self.provider_id and self.model_id:
            return {"providerID": self.provider_id, "modelID": self.model_id}
        try:
            data = self._get("/config/providers").json()
            defaults = data.get("default") or {}
            if defaults:
                provider_id, model_id = next(iter(defaults.items()))
                return {"providerID": provider_id, "modelID": model_id}
            for provider in data.get("providers", []):
                models = provider.get("models") or {}
                if models:
                    model_id = next(iter(models))
                    return {"providerID": provider.get("id"), "modelID": model_id}
        except Exception:
            # Si no podemos descubrir el modelo, enviamos sin `model` y
            # dejamos que el servidor use su configuración por defecto.
            pass
        return None

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extrae el texto de la respuesta (concatenando las parts de texto)."""
        parts = data.get("parts") or []
        texts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(text for text in texts if text).strip()


def build_llm(settings) -> OpenCodeLLM:
    """Factoría que crea el LLM a partir de la configuración."""
    return OpenCodeLLM(
        base_url=settings.opencode_base_url,
        provider_id=settings.opencode_provider_id,
        model_id=settings.opencode_model_id,
        agent=settings.opencode_agent,
        username=settings.opencode_username,
        password=settings.opencode_password,
        timeout=settings.opencode_timeout,
    )
