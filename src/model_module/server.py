"""Protocol adapter: ModuleApp + ToolSpec. No request logic lives here.

The only jobs:
  1. declare `chat` with M6 §四 metadata (`risk=low`, 120 s, `exclusive`,
     `priority=100`) and let the SDK derive the JSON Schema from the function
     signature;
  2. translate `tools.ToolFailure` into `vade_sdk.ToolError`, keeping the C-04
     code and retriable flag;
  3. emit one structured log line per call — message count, `messages_digest`,
     token usage and duration. Never the prompt, never the completion.

`tools.py` never imports the SDK; this file contains no validation, no HTTP and
no response shaping. A reviewer can read `tools.py` + `provider.py` and know
exactly what goes on the wire.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from . import __version__
from . import tools as tools_module
from .provider import ModelClient, config_from_env

__all__ = ["MODULE_ID", "MODULE_VERSION", "PRIORITY", "build_app", "main"]

MODULE_ID = "com.example.model"
MODULE_VERSION = __version__

#: FR-6.3: with several model modules online, core picks the highest priority.
#: 100 is the top of the SDK's [0, 100] range; `exclusive` means one instance
#: owns the upstream, so two replicas cannot double-bill the same turn.
PRIORITY = 100


def _adapt(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap the tool so `ToolFailure` becomes `ToolError` with the same code."""
    from vade_sdk import ToolError

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except tools_module.ToolFailure as exc:
            raise ToolError(exc.code, exc.message, retriable=exc.retriable) from None

    return call


def _make_chat(client_factory: Callable[[], Any], log: Any) -> Callable[..., dict[str, Any]]:
    """Build the `chat` tool around a client factory (injectable in tests)."""

    def chat(
        messages: list,
        tools: list = [],  # noqa: B006 - never mutated; [] is the schema default M6 asks for
        temperature: float = 0.7,
        model: str = "",
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            validated = tools_module.validate_chat(messages, tools, temperature, model)
        except tools_module.ToolFailure as exc:
            log.info(
                "model chat rejected",
                tool="chat",
                code=exc.code,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            raise

        messages_in = validated["messages"]
        digest = tools_module.messages_digest(messages_in)
        body = tools_module.build_request_body(
            messages_in,
            validated["tools"],
            validated["temperature"],
            validated["model"],
        )

        attempts = 0
        try:
            client = client_factory()
            payload, attempts = client.chat(body)
            # The client returns the M6-shaped result; only the size budget is
            # the adapter's job (it is about the frame, not about the vendor).
            result = tools_module.fit_result(payload)
        except tools_module.ToolFailure as exc:
            if exc.attempts is None:
                exc.attempts = attempts
            log.info(
                "model chat failed",
                tool="chat",
                code=exc.code,
                attempts=exc.attempts,
                messages=len(messages_in),
                messages_digest=digest,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            raise

        usage = result.get("usage") or {}
        log.info(
            "model chat",
            tool="chat",
            messages=len(messages_in),
            messages_digest=digest,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=result.get("finish_reason"),
            attempts=attempts,
            truncated=bool(result.get("truncated")),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return result

    return chat


def build_app(client: Any = None, logger: Any = None) -> Any:
    """Build the ModuleApp. Importing the SDK happens here, not at import time.

    `client` is for tests only: a duck-typed object with
    `chat(body) -> (payload, attempts)`. Production always builds a
    `ModelClient` from the environment, per call, so a changed key or base URL
    needs no module restart.
    """
    from vade_sdk import ModuleApp, ToolSpec

    app = ModuleApp(
        module_id=MODULE_ID,
        version=MODULE_VERSION,
        # One upstream, one instance: two model modules with `exclusive` cannot
        # both serve the same conversation, and priority breaks the tie.
        concurrency="exclusive",
        display_name="模型模块",
        priority=PRIORITY,
        capabilities=["tools"],
        logger=logger,
    )

    factory: Callable[[], Any]
    if client is not None:
        factory = lambda: client  # noqa: E731 - a closure, not a style choice
    else:
        factory = config_and_client

    app.add_tool(
        ToolSpec(
            name=tools_module.CHAT_META["name"],
            fn=_adapt(_make_chat(factory, app.log)),
            description=tools_module.CHAT_META["description"],
            risk=tools_module.CHAT_META["risk"],
            timeout_ms=tools_module.CHAT_META["timeout_ms"],
            idempotent=tools_module.CHAT_META["idempotent"],
        )
    )
    return app


def config_and_client() -> ModelClient:
    """Resolve `MODEL_*` now and build one client. Import-time-free on purpose."""
    return ModelClient(config_from_env())


def main(argv: list[str] | None = None) -> int:
    return build_app().run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
