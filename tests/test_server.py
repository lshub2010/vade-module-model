"""Adapter tests: manifest identity, schema, error mapping, logging and the CLI.

Skipped entirely when the SDK is not installed, so `pytest tests` still runs on
a bare Python (that is the module's own gate). The SDK is looked up in the
usual three places: `VADE_SDK_PATH`, a vendored `sdk/python`, and the sibling
Vade checkout used during development.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import http.server
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))


def _sdk_candidates() -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("VADE_SDK_PATH"):
        candidates.append(Path(os.environ["VADE_SDK_PATH"]))
    candidates.append(MODULE_ROOT / "sdk" / "python")
    candidates.append(MODULE_ROOT.parent / "Vade" / "sdk" / "python")
    candidates.append(MODULE_ROOT.parents[1] / "sdk" / "python")
    return candidates


SDK_DIR = next((path for path in _sdk_candidates() if (path / "vade_sdk").is_dir()), None)
if SDK_DIR is not None:
    sys.path.insert(0, str(SDK_DIR))

pytest.importorskip("vade_sdk", reason="vade-sdk is required for the adapter tests")

from vade_sdk.logging import JsonLogger  # noqa: E402

from model_module import server, tools  # noqa: E402

SECRET = "USER_PROMPT_绝不进日志_9f3a"

RAW_PAYLOAD = {
    "id": "chatcmpl-1",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        # Deliberately NOT a string: the module must fix it.
                        "function": {"name": "fs__read_file", "arguments": {"path": "/a"}},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 34},
    "vendor_extension": {"trace": "drop-me"},
}


class FakeClient:
    def __init__(self, payload: dict | None = None, *, attempts: int = 1, error: Exception | None = None) -> None:
        self.payload = payload if payload is not None else RAW_PAYLOAD
        self.attempts = attempts
        self.error = error
        self.bodies: list[dict] = []

    def chat(self, body: dict) -> tuple[dict, int]:
        self.bodies.append(body)
        if self.error is not None:
            raise self.error
        # The real provider normalises before returning; the fake must honour
        # the same contract or the adapter is being tested against a fiction.
        return tools.normalize_response(self.payload), self.attempts


def _logger(stream: io.StringIO) -> JsonLogger:
    return JsonLogger(server.MODULE_ID, "main", stream=stream)


@pytest.fixture()
def captured() -> io.StringIO:
    return io.StringIO()


@pytest.fixture()
def app(captured):
    return server.build_app(client=FakeClient(), logger=_logger(captured))


def _chat_args(content: str = SECRET) -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def _records(captured: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in captured.getvalue().splitlines() if line.strip()]


# ---------------------------------------------------------------------- identity


def test_module_identity_and_priority_are_stable(app):
    assert server.MODULE_ID == "com.example.model"
    assert server.MODULE_VERSION == "1.0.0"
    assert server.PRIORITY == 100
    assert app.priority == 100
    assert app.concurrency == "exclusive"


def test_committed_manifest_matches_the_app(app):
    committed = json.loads((MODULE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert app.manifest() == committed


def test_check_is_clean(app):
    assert app.check() == []


def test_schema_comes_from_the_chat_signature(app):
    schema = app.tool("chat").input_schema
    assert schema["required"] == ["messages"]
    assert schema["properties"]["messages"] == {"type": "array"}
    assert schema["properties"]["tools"] == {"type": "array", "default": []}
    assert schema["properties"]["temperature"] == {"type": "number", "default": 0.7}
    assert schema["properties"]["model"] == {"type": "string", "default": ""}
    assert schema["additionalProperties"] is False


def test_chat_metadata_is_low_risk_and_120_seconds(app):
    spec = app.tool("chat")
    assert spec.risk == "low"
    assert spec.timeout_ms == 120_000
    assert spec.idempotent is False
    assert spec.description == tools.CHAT_META["description"]


# ---------------------------------------------------------------------- contract


async def test_chat_returns_the_m6_contract_and_normalises_arguments(app):
    envelope = await app.call_tool("chat", _chat_args())

    assert envelope["ok"] is True, envelope
    result = envelope["result"]
    assert result["message"]["content"] == "ok"
    assert result["message"]["role"] == "assistant"
    call = result["message"]["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "fs__read_file"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"path": "/a"}
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 34}
    assert result["finish_reason"] == "tool_calls"
    assert set(result) == {"message", "usage", "finish_reason"}


async def test_the_provider_receives_the_validated_body(captured):
    client = FakeClient()
    app = server.build_app(client=client, logger=_logger(captured))
    await app.call_tool("chat", {"messages": [{"role": "user", "content": SECRET}], "temperature": 0.1})

    body = client.bodies[0]
    assert body["stream"] is False
    assert body["temperature"] == 0.1
    assert body["messages"] == [{"role": "user", "content": SECRET}]


async def test_empty_response_content_is_an_empty_string(captured):
    payload = {
        "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}],
        "usage": {},
    }
    app = server.build_app(client=FakeClient(payload), logger=_logger(captured))
    envelope = await app.call_tool("chat", _chat_args())
    assert envelope["result"]["message"]["content"] == ""
    assert envelope["result"]["message"]["tool_calls"] == []


# ------------------------------------------------------------------- validation


async def test_oversized_messages_are_rejected_before_any_request(captured):
    client = FakeClient()
    app = server.build_app(client=client, logger=_logger(captured))
    messages = [{"role": "user", "content": "x" * (60 * 1024)} for _ in range(9)]

    envelope = await app.call_tool("chat", {"messages": messages})

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "INVALID_ARGUMENT"
    assert client.bodies == []


async def test_more_than_200_messages_is_invalid(captured):
    app = server.build_app(client=FakeClient(), logger=_logger(captured))
    messages = [{"role": "user", "content": f"m{i}"} for i in range(201)]
    envelope = await app.call_tool("chat", {"messages": messages})
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "INVALID_ARGUMENT"


async def test_unknown_argument_is_rejected_by_the_sdk(app):
    envelope = await app.call_tool("chat", {**_chat_args(), "stream": True})
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "INVALID_ARGUMENT"


async def test_unknown_tool_is_not_found(app):
    assert (await app.call_tool("complete", _chat_args()))["error"]["code"] == "NOT_FOUND"


# ----------------------------------------------------------------------- errors


async def test_provider_failure_keeps_its_code_and_retriable_flag(captured):
    failure = tools.ToolFailure("OVERLOADED", "upstream rate limit was still exhausted after 3 attempt(s)")
    failure.attempts = 3
    app = server.build_app(client=FakeClient(error=failure), logger=_logger(captured))

    envelope = await app.call_tool("chat", _chat_args())

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "OVERLOADED"
    assert envelope["error"]["retriable"] is True
    assert "3 attempt(s)" in envelope["error"]["message"]


async def test_configuration_errors_are_internal_and_readable(captured, monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "ftp://model.test")
    app = server.build_app(logger=_logger(captured))  # no fake client: really read env

    envelope = await app.call_tool("chat", _chat_args())

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "INTERNAL"
    assert "MODEL_BASE_URL" in envelope["error"]["message"]


# ----------------------------------------------------------------------- logging


async def test_logs_carry_digest_counts_and_tokens_but_never_content(app, captured):
    await app.call_tool("chat", _chat_args())

    raw = captured.getvalue()
    assert SECRET not in raw
    assert tools.messages_digest(_chat_args()["messages"]) in raw

    records = _records(captured)
    chat = next(record for record in records if record["msg"] == "model chat")
    assert chat["tool"] == "chat"
    assert chat["messages"] == 1
    assert chat["prompt_tokens"] == 12
    assert chat["completion_tokens"] == 34
    assert chat["finish_reason"] == "tool_calls"
    assert chat["attempts"] == 1
    assert chat["truncated"] is False
    assert isinstance(chat["duration_ms"], int)
    assert chat["module_id"] == "com.example.model"
    assert chat["instance_id"] == "main"


async def test_rejected_calls_are_logged_without_content(captured):
    app = server.build_app(client=FakeClient(), logger=_logger(captured))
    await app.call_tool("chat", {"messages": [{"role": "user", "content": SECRET}], "temperature": 9})

    raw = captured.getvalue()
    assert SECRET not in raw
    record = next(record for record in _records(captured) if record["msg"] == "model chat rejected")
    assert record["code"] == "INVALID_ARGUMENT"


async def test_failures_are_logged_with_digest_and_no_content(captured):
    failure = tools.ToolFailure("UNAVAILABLE", "cannot reach the model endpoint")
    app = server.build_app(client=FakeClient(error=failure), logger=_logger(captured))

    await app.call_tool("chat", _chat_args())

    raw = captured.getvalue()
    assert SECRET not in raw
    record = next(record for record in _records(captured) if record["msg"] == "model chat failed")
    assert record["code"] == "UNAVAILABLE"
    assert record["messages"] == 1
    assert record["messages_digest"] == tools.messages_digest(_chat_args()["messages"])


# ---------------------------------------------------------------- result budget


async def test_a_large_answer_is_truncated_in_place_of_overflowing(captured):
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "y" * 60_000}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 60_000},
    }
    app = server.build_app(client=FakeClient(payload), logger=_logger(captured))

    envelope = await app.call_tool("chat", _chat_args())

    assert envelope["ok"] is True
    result = envelope["result"]
    assert result["truncated"] is True
    assert len(result["message"]["content"]) < 60_000
    assert tools.result_bytes(result) <= tools.MAX_RESULT_BYTES
    # The SDK envelope has its own `truncated`, always false (it never trims).
    assert envelope["truncated"] is False


# --------------------------------------------------------------------------- CLI


def _subprocess_env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("MODEL_BASE_URL", "MODEL_NAME", "MODEL_API_KEY", "MODEL_API_KEY_ENV",
                "MODEL_TIMEOUT_S", "MODEL_MAX_RETRIES", "MODEL_RETRY_BASE_S"):
        env.pop(key, None)
    paths = [str(MODULE_ROOT / "src")]
    if SDK_DIR is not None:
        paths.append(str(SDK_DIR))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(extra)
    return env


class _Upstream(http.server.BaseHTTPRequestHandler):
    payload: dict = {}
    status: int = 200
    last_body: dict | None = None
    last_headers: dict | None = None

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        type(self).last_body = json.loads(raw or b"{}")
        type(self).last_headers = dict(self.headers.items())
        body = json.dumps(type(self).payload).encode("utf-8")
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass


@pytest.fixture()
def upstream():
    _Upstream.payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "from the upstream"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    _Upstream.status = 200
    _Upstream.last_body = None
    _Upstream.last_headers = None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_standalone_chat_against_a_local_upstream(upstream):
    """The real CLI + real urllib + real JSON, in a fresh interpreter."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_module",
            "--standalone",
            "chat",
            "--args",
            json.dumps({"messages": [{"role": "user", "content": SECRET}]}),
        ],
        cwd=MODULE_ROOT,
        env=_subprocess_env(
            MODEL_BASE_URL=upstream,
            MODEL_NAME="standalone-model",
            MODEL_API_KEY="sk-standalone",
            MODEL_TIMEOUT_S="5",
            MODEL_MAX_RETRIES="0",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    envelope = json.loads(completed.stdout)
    assert envelope["ok"] is True
    assert envelope["result"] == {
        "message": {"role": "assistant", "content": "from the upstream", "tool_calls": []},
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
        "finish_reason": "stop",
    }

    # The wire body is OpenAI-shaped, non-streaming, and carries the override.
    assert _Upstream.last_body is not None
    assert _Upstream.last_body["model"] == "standalone-model"
    assert _Upstream.last_body["stream"] is False
    assert _Upstream.last_body["messages"] == [{"role": "user", "content": SECRET}]
    assert _Upstream.last_headers["Authorization"] == "Bearer sk-standalone"

    # Neither stdout nor stderr may contain the prompt.
    assert SECRET not in completed.stdout
    assert SECRET not in completed.stderr
    assert tools.messages_digest([{"role": "user", "content": SECRET}]) in completed.stderr


def test_standalone_rejects_an_oversized_prompt_without_a_network_call():
    # One 70 KiB message: over the 64 KiB per-message ceiling but under Linux's
    # 128 KiB MAX_ARG_STRLEN, so it can still be passed through --args.
    messages = [{"role": "user", "content": "z" * (70 * 1024)}]
    completed = subprocess.run(
        [sys.executable, "-m", "model_module", "--standalone", "chat", "--args",
         json.dumps({"messages": messages})],
        cwd=MODULE_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 1
    envelope = json.loads(completed.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "INVALID_ARGUMENT"
    assert "65536" in envelope["error"]["message"]


def test_print_manifest_is_byte_identical_to_the_committed_file():
    completed = subprocess.run(
        [sys.executable, "-m", "model_module", "--print-manifest"],
        cwd=MODULE_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (MODULE_ROOT / "manifest.json").read_text(encoding="utf-8")
