"""Model tools — pure request/response logic, stdlib only, ZERO SDK imports.

This is the module's "self-sufficiency" layer (design §15.1): everything in
this file can be written, reviewed and tested before `vade-sdk` exists, and
`pytest tests/test_tools.py` runs on a bare Python.

Three jobs, all pure functions:

1. **Validate** the `chat` arguments against the M6 limits (200 messages,
   four roles, 64 KiB per message, 512 KiB total).
2. **Build** the OpenAI `/chat/completions` request body and **normalise** the
   reply into M6 §五's frozen shape. `tool_calls[].function.arguments` is
   ALWAYS a string; an object there is silently dropped by core's adapter, so
   a non-OpenAI upstream is converted here rather than passed through.
3. **Fit** the result into the 48 KiB result budget by truncating `content`
   and setting `truncated=true`. Tool calls are never dropped — a dropped call
   is a silently skipped action.

No provider-specific field survives normalisation: only `id` / `type` /
`function` per call, and only `prompt_tokens` / `completion_tokens` for usage.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

__all__ = [
    "ToolFailure",
    "CHAT_META",
    "ROLES",
    "MAX_MESSAGES",
    "MAX_MESSAGE_BYTES",
    "MAX_MESSAGES_BYTES",
    "MAX_RESULT_BYTES",
    "MAX_TOOLS",
    "MAX_MODEL_CHARS",
    "messages_digest",
    "validate_chat",
    "validate_messages",
    "validate_tools",
    "validate_temperature",
    "validate_model",
    "build_request_body",
    "normalize_response",
    "fit_result",
    "result_bytes",
]

#: M6 §四: `chat(messages, tools=[], temperature=0.7)`, risk low, 120 s.
CHAT_META: dict[str, Any] = {
    "name": "chat",
    "description": (
        "调用 OpenAI 兼容的 /chat/completions（v1 非流式）：返回 message"
        "（tool_calls 的 arguments 为字符串）、usage、finish_reason；不执行工具"
    ),
    "risk": "low",
    "timeout_ms": 120_000,
    # A completion is sampled: asking twice costs twice and may answer
    # differently. Never let the router retry it silently.
    "idempotent": False,
}

ROLES = ("system", "user", "assistant", "tool")

#: FR-6.2 / M6 §四. Every number is a *ceiling*, not a target: a request past
#: it is refused before a socket is opened, so a 200-message pasted transcript
#: cannot turn into a 40 MB HTTP body.
MAX_MESSAGES = 200
MAX_MESSAGE_BYTES = 64 * 1024
MAX_MESSAGES_BYTES = 512 * 1024

#: M6 §十 asks the whole result JSON to stay under one frame's worth of space.
#: Measured on the *result object* (compact JSON, UTF-8), exactly the way the
#: SDK serialises it. The remaining ~16 KiB of the 65519-byte frame is the
#: JSON-RPC envelope, `duration_ms` and the Noise tag.
MAX_RESULT_BYTES = 48 * 1024

#: A `tools` array larger than this is a caller bug, not a model request.
MAX_TOOLS = 128

#: Model ids are short; a 10 KiB "model" is a probe, not a name.
MAX_MODEL_CHARS = 200

#: Codes this module can produce, with C-04's retriable flags. Kept local so
#: `tools.py` stays dependency-free; `server.py` hands the same pair to
#: `vade_sdk.ToolError`, which validates the code against the real table.
_RETRIABLE = {
    "INVALID_ARGUMENT": False,
    "NOT_FOUND": False,
    "PERMISSION_DENIED": False,
    "TIMEOUT": True,
    "CANCELLED": False,
    "UNAVAILABLE": True,
    "OVERLOADED": True,
    "PAYLOAD_TOO_LARGE": False,
    "UNSUPPORTED": False,
    "INTERNAL": False,
}


class ToolFailure(Exception):
    """A structured failure raised by a pure helper.

    Deliberately NOT `vade_sdk.ToolError`: importing the SDK here would break
    the "test the module without the SDK" rule. `server.py` translates this
    into a `ToolError` with the same code and retriable flag.
    """

    def __init__(self, code: str, message: str, retriable: bool | None = None) -> None:
        if code not in _RETRIABLE:
            raise ValueError(f"{code!r} is not a code this module may emit")
        self.code = code
        self.message = str(message)[:300]
        self.retriable = _RETRIABLE[code] if retriable is None else bool(retriable)
        #: Filled in by the client; `None` when the failure happened before
        #: the first HTTP attempt.
        self.attempts: int | None = None
        super().__init__(f"{code}: {self.message}")


# --------------------------------------------------------------------- encoding


def _compact(value: Any, *, sort_keys: bool = False) -> str:
    """The exact serialisation used for size checks and the digest."""
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def _json_bytes(value: Any, *, sort_keys: bool = False) -> int:
    return len(_compact(value, sort_keys=sort_keys).encode("utf-8"))


def messages_digest(messages: list[Any]) -> str:
    """sha256 of the canonical message array, first 16 hex chars.

    Logs carry this instead of the prompt: two identical prompts share a
    digest, so a support case can be correlated without the transcript ever
    reaching a log file.
    """
    try:
        canonical = _compact(messages, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolFailure("INVALID_ARGUMENT", f"messages must be JSON-serialisable: {exc}") from None
    return hashlib.sha256(canonical).hexdigest()[:16]


# ------------------------------------------------------------------- validation


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolFailure("INVALID_ARGUMENT", f"{field} must be an integer, got {type(value).__name__}")
    return value


def validate_messages(messages: Any) -> list[dict[str, Any]]:
    """Check the M6 §四 limits. Returns the list unchanged (validated)."""
    if not isinstance(messages, list):
        raise ToolFailure("INVALID_ARGUMENT", f"messages must be an array, got {type(messages).__name__}")
    if not messages:
        raise ToolFailure("INVALID_ARGUMENT", "messages must not be empty")
    if len(messages) > MAX_MESSAGES:
        raise ToolFailure(
            "INVALID_ARGUMENT", f"messages has {len(messages)} entries; the limit is {MAX_MESSAGES}"
        )

    total = 0
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if not isinstance(message, Mapping):
            raise ToolFailure("INVALID_ARGUMENT", f"{where} must be an object, got {type(message).__name__}")
        role = message.get("role")
        if role not in ROLES:
            raise ToolFailure(
                "INVALID_ARGUMENT", f"{where}.role must be one of {list(ROLES)}, got {role!r}"
            )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            # v1 is text-only. An image/parts array would be forwarded to a
            # provider that may silently drop it, so it is refused loudly.
            raise ToolFailure(
                "INVALID_ARGUMENT", f"{where}.content must be a string or null, got {type(content).__name__}"
            )
        try:
            size = _json_bytes(message, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ToolFailure("INVALID_ARGUMENT", f"{where} is not JSON-serialisable: {exc}") from None
        if size > MAX_MESSAGE_BYTES:
            raise ToolFailure(
                "INVALID_ARGUMENT",
                f"{where} is {size} bytes; the per-message limit is {MAX_MESSAGE_BYTES}",
            )
        total += size

    # Compact array = '[' + items joined by ',' + ']'.
    total += max(0, len(messages) - 1) + 2
    if total > MAX_MESSAGES_BYTES:
        raise ToolFailure(
            "INVALID_ARGUMENT",
            f"messages is {total} bytes; the total limit is {MAX_MESSAGES_BYTES}",
        )
    return messages


def validate_tools(tools: Any) -> list[Mapping[str, Any]]:
    """`tools` is forwarded to the provider; only its outer shape is checked."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ToolFailure("INVALID_ARGUMENT", f"tools must be an array, got {type(tools).__name__}")
    if len(tools) > MAX_TOOLS:
        raise ToolFailure("INVALID_ARGUMENT", f"tools has {len(tools)} entries; the limit is {MAX_TOOLS}")
    for index, spec in enumerate(tools):
        if not isinstance(spec, Mapping):
            raise ToolFailure(
                "INVALID_ARGUMENT", f"tools[{index}] must be an object, got {type(spec).__name__}"
            )
    return tools


def validate_temperature(temperature: Any) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ToolFailure(
            "INVALID_ARGUMENT", f"temperature must be a number, got {type(temperature).__name__}"
        )
    value = float(temperature)
    if not 0.0 <= value <= 2.0:
        raise ToolFailure("INVALID_ARGUMENT", f"temperature must be within [0, 2], got {value}")
    return value


def validate_model(model: Any) -> str:
    """`model` is the optional per-call override core sends on a pinned turn.

    Empty means "use MODEL_NAME". It is not a vendor field: the OpenAI body
    has had a `model` key since v1, and core's ModuleLLMProvider sends it
    whenever a turn is pinned (see `core/vade/llm/module.py`).
    """
    if model is None:
        return ""
    if not isinstance(model, str):
        raise ToolFailure("INVALID_ARGUMENT", f"model must be a string, got {type(model).__name__}")
    value = model.strip()
    if len(value) > MAX_MODEL_CHARS:
        raise ToolFailure("INVALID_ARGUMENT", f"model is longer than {MAX_MODEL_CHARS} characters")
    return value


def validate_chat(
    messages: Any,
    tools: Any = None,
    temperature: Any = 0.7,
    model: Any = "",
) -> dict[str, Any]:
    """Validate every argument once and return the normalised call."""
    return {
        "messages": validate_messages(messages),
        "tools": validate_tools(tools),
        "temperature": validate_temperature(temperature),
        "model": validate_model(model),
    }


def build_request_body(
    messages: list[Mapping[str, Any]],
    tools: list[Mapping[str, Any]],
    temperature: float,
    model: str,
) -> dict[str, Any]:
    """The OpenAI `/chat/completions` body. `stream` is explicit, never implied."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # v1 does not stream (M6 §六). Sending `false` explicitly means a
        # provider default cannot turn this into an SSE response we cannot read.
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    return body


# -------------------------------------------------------------------- response


def _upstream_kind(error: Any) -> str:
    """A provider's error *type/code*, never its message or the prompt echo."""
    if isinstance(error, Mapping):
        for key in ("type", "code", "status"):
            value = error.get(key)
            if isinstance(value, str) and value:
                return value[:80]
        return "object"
    if isinstance(error, str):
        return error[:80]
    return "unknown"


def _content_text(value: Any) -> str:
    """Flatten OpenAI string content or a text-parts array into one string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, Mapping):
                raise ToolFailure("INTERNAL", "upstream returned a malformed content part")
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif part.get("type") in ("tool_use", "tool_call"):
                continue  # handled by _tool_calls
            else:
                raise ToolFailure(
                    "INTERNAL", f"upstream returned a non-text content part ({_upstream_kind(part)})"
                )
        return "".join(parts)
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            return text
    raise ToolFailure("INTERNAL", f"upstream returned non-text content ({type(value).__name__})")


def _arguments_text(value: Any) -> str:
    """M6 §五: arguments is a STRING on the wire, even when the model sent JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    raise ToolFailure("INTERNAL", f"upstream returned tool arguments of type {type(value).__name__}")


def _tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = message.get("tool_calls")
    if raw is None:
        content = message.get("content")
        if isinstance(content, list):
            # Anthropic-shaped content blocks. `input` is an object by design;
            # it becomes a JSON string here.
            raw = [
                part
                for part in content
                if isinstance(part, Mapping) and part.get("type") in ("tool_use", "tool_call")
            ]
        else:
            return []
    if not isinstance(raw, (list, tuple)):
        raise ToolFailure("INTERNAL", "upstream returned tool_calls that is not an array")

    calls: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ToolFailure("INTERNAL", "upstream returned a malformed tool call")
        function = item.get("function")
        if function is None and ("name" in item or "input" in item):
            function = {"name": item.get("name"), "arguments": item.get("input", item.get("arguments"))}
        if not isinstance(function, Mapping):
            raise ToolFailure("INTERNAL", "upstream returned a tool call without a function object")
        name = function.get("name") or item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolFailure("INTERNAL", "upstream returned a tool call without a function name")
        call_id = item.get("id")
        if call_id is None or call_id == "":
            call_id = f"call_{index + 1}"
        calls.append(
            {
                "id": str(call_id),
                # Only "function" exists in v1; the upstream `type` is not
                # forwarded (a vendor value core does not know must not leak).
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _arguments_text(function.get("arguments", function.get("input"))),
                },
            }
        )
    return calls


_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "eos": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "content_filter": "content_filter",
}


def _finish_reason(choice: Mapping[str, Any], payload: Mapping[str, Any], has_calls: bool) -> str:
    nested = choice.get("message")
    nested_reason = nested.get("finish_reason") if isinstance(nested, Mapping) else None
    raw = (
        choice.get("finish_reason")
        or payload.get("finish_reason")
        or payload.get("stop_reason")
        or nested_reason
    )
    if isinstance(raw, str) and raw:
        return _FINISH_REASONS.get(raw, raw)
    # The contract always has a reason; "tool_calls" is the truthful default
    # when the message carries calls and the provider said nothing.
    return "tool_calls" if has_calls else "stop"


def _usage(payload: Mapping[str, Any]) -> dict[str, int]:
    raw = payload.get("usage")
    usage = raw if isinstance(raw, Mapping) else {}

    def count(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return 0

    # Only the two frozen fields travel; `total_tokens` and vendor detail
    # objects are not part of M6 §五 and would be noise for core.
    return {
        "prompt_tokens": count("prompt_tokens", "input_tokens"),
        "completion_tokens": count("completion_tokens", "output_tokens"),
    }


def _pick_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ToolFailure("INTERNAL", "upstream returned a malformed choice")
        return first
    # Flat shape: some OpenAI-compatible servers answer `{message, usage}`.
    if isinstance(payload.get("message"), Mapping):
        return payload
    # Anthropic-shaped reply: `{content: [...], stop_reason, usage}`.
    if "content" in payload:
        return payload
    raise ToolFailure("INTERNAL", "upstream response has no choices")


def normalize_response(payload: Any) -> dict[str, Any]:
    """Translate any supported upstream shape into M6 §五's frozen result."""
    if not isinstance(payload, Mapping):
        raise ToolFailure("INTERNAL", "upstream returned a non-object JSON body")
    if "error" in payload and not payload.get("choices") and "message" not in payload:
        raise ToolFailure(
            "INTERNAL", f"upstream returned an error object ({_upstream_kind(payload.get('error'))})"
        )

    choice = _pick_choice(payload)
    message = choice.get("message")
    if not isinstance(message, Mapping):
        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            message = delta
        elif "content" in choice:
            message = choice
        else:
            message = {}

    calls = _tool_calls(message)
    content = _content_text(message.get("content"))
    return {
        "message": {"role": "assistant", "content": content, "tool_calls": calls},
        "usage": _usage(payload),
        "finish_reason": _finish_reason(choice, payload, bool(calls)),
    }


# ---------------------------------------------------------------- result budget


def result_bytes(result: Any) -> int:
    """Byte size of the result as the SDK will serialise it."""
    try:
        return _json_bytes(result)
    except (TypeError, ValueError) as exc:
        raise ToolFailure("INTERNAL", f"result is not JSON-serialisable: {exc}") from None


def _safe_prefix(text: str, length: int) -> str:
    """Never cut a surrogate pair in half (json/UTF-8 would reject the tail)."""
    if length and 0xD800 <= ord(text[length - 1]) <= 0xDBFF:
        return text[: length - 1]
    return text[:length]


def fit_result(result: Mapping[str, Any], budget: int = MAX_RESULT_BYTES) -> dict[str, Any]:
    """Keep the result under `budget` by trimming `content`, never tool calls.

    A tool call is an action; truncating its `arguments` string would make the
    JSON unparseable and dropping the call would silently skip the action. So
    if the calls alone already exceed the budget the call fails with
    `PAYLOAD_TOO_LARGE` instead of lying about what the model asked for.
    """
    size = result_bytes(result)
    if size <= budget:
        return dict(result)

    message = result.get("message")
    if not isinstance(message, Mapping):
        raise ToolFailure("INTERNAL", "result has no message object")
    content = message.get("content")
    if not isinstance(content, str):
        content = ""

    fitted: dict[str, Any] = dict(result)
    fitted["message"] = {**message, "content": ""}
    fitted["truncated"] = True
    if result_bytes(fitted) > budget:
        raise ToolFailure(
            "PAYLOAD_TOO_LARGE",
            f"result exceeds the {budget}-byte budget even with empty content; tool_calls are too large",
        )

    low, high, best = 0, len(content), 0
    while low <= high:
        mid = (low + high) // 2
        candidate = _safe_prefix(content, mid)
        fitted["message"]["content"] = candidate
        if result_bytes(fitted) <= budget:
            best = len(candidate)
            low = mid + 1
        else:
            high = mid - 1

    fitted["message"]["content"] = content[:best]
    return fitted
