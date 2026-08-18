"""Pruebas del wrapper `OpenCodeLLM` contra una API HTTP simulada.

Verifican el contrato real de OpenCode Server:
  * `GET  /config/providers`  -> descubrimiento del modelo por defecto.
  * `POST /session`           -> creación de sesión.
  * `POST /session/{id}/message` -> envío de mensaje y extracción de texto.

Ejecutar:  python -m tests.test_llm
"""
from __future__ import annotations

import sys
import unittest.mock as mock

from langchain_core.messages import HumanMessage, SystemMessage

from ai_code_analyzer.llm import OpenCodeLLM


def test_generate_discovers_model_and_posts_expected_body() -> None:
    llm = OpenCodeLLM(base_url="http://127.0.0.1:4096")

    with mock.patch("ai_code_analyzer.llm.requests.get") as get, mock.patch(
        "ai_code_analyzer.llm.requests.post"
    ) as post:
        # 1) GET /config/providers -> modelo por defecto.
        get.return_value.json.return_value = {"default": {"openai": "gpt-4o"}}

        # 2) POST /session -> {"id": "s1"} y luego el mensaje con la respuesta.
        post.return_value.raise_for_status = lambda: None
        post.return_value.json.side_effect = [
            {"id": "s1"},
            {
                "info": {"tokens": {"input": 3, "output": 2}},
                "parts": [
                    {"type": "reasoning", "text": "no debería contarse"},
                    {"type": "text", "text": "respuesta simulada"},
                ],
            },
        ]

        result = llm.invoke([SystemMessage(content="instrucciones"), HumanMessage(content="hola")])

        assert result.content == "respuesta simulada"
        assert get.call_args.args[0] == "http://127.0.0.1:4096/config/providers"

        session_call = post.call_args_list[0]
        message_call = post.call_args_list[1]
        assert session_call.args[0] == "http://127.0.0.1:4096/session"
        assert message_call.args[0] == "http://127.0.0.1:4096/session/s1/message"

        body = message_call.kwargs["json"]
        assert body["model"] == {"providerID": "openai", "modelID": "gpt-4o"}
        assert body["system"] == "instrucciones"
        assert body["parts"][0] == {"type": "text", "text": "User: hola"}
        print("[OK ] Cuerpo de la petición y extracción de texto correctos")


def test_generate_uses_explicit_model_without_discovery() -> None:
    llm = OpenCodeLLM(
        base_url="http://x", provider_id="anthropic", model_id="claude-3"
    )
    with mock.patch("ai_code_analyzer.llm.requests.get") as get, mock.patch(
        "ai_code_analyzer.llm.requests.post"
    ) as post:
        post.return_value.raise_for_status = lambda: None
        post.return_value.json.side_effect = [
            {"id": "s2"},
            {"info": {}, "parts": [{"type": "text", "text": "ok"}]},
        ]
        llm.invoke([HumanMessage(content="x")])

        get.assert_not_called()
        body = post.call_args_list[1].kwargs["json"]
        assert body["model"] == {"providerID": "anthropic", "modelID": "claude-3"}
        print("[OK ] Modelo explícito enviado sin llamar a /config/providers")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    test_generate_discovers_model_and_posts_expected_body()
    test_generate_uses_explicit_model_without_discovery()
    print("  Todas las pruebas de OpenCodeLLM pasaron.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
