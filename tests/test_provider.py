"""Provider tests — a fake transport, no socket, no SDK.

Every retry rule from M6 §六 has one test here: 429 retried twice then success,
400 never retried, 5xx exhausted to `UNAVAILABLE`, timeout to `TIMEOUT`, and a
connection error retried. The key rule is checked by inspecting the headers the
client would send.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_module import provider  # noqa: E402
from model_module.tools import ToolFailure  # noqa: E402


class FakeTransport:
    """Returns/raises the queued items and records every request."""

    def __init__(self, items: list) -> None:
        self.items = list(items)
        self.requests: list[provider.HttpRequest] = []

    def __call__(self, request: provider.HttpRequest) -> provider.HttpResponse:
        self.requests.append(request)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def warning(self, message: str, **fields) -> None:
        self.calls.append((message, fields))


def _response(status: int, payload: dict | None = None, headers: dict | None = None) -> provider.HttpResponse:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    return provider.HttpResponse(status=status, headers=headers or {}, body=body)


def _success(content: str = "hi") -> provider.HttpResponse:
    return _response(
        200,
        {
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
    )


def _config(**overrides) -> provider.ModelConfig:
    base = dict(
        base_url="http://model.test/v1",
        model="mock-model",
        api_key_env="MODEL_API_KEY",
        timeout_s=5.0,
        max_retries=2,
        retry_base_s=0.5,
    )
    base.update(overrides)
    return provider.ModelConfig(**base)


def _client(transport: FakeTransport, *, sleeps: list | None = None, logger=None, **overrides) -> provider.ModelClient:
    recorded = sleeps if sleeps is not None else []
    return provider.ModelClient(
        _config(**overrides),
        transport=transport,
        sleep=recorded.append,
        clock=lambda: 1000.0,
        jitter=lambda: 0.0,
        logger=logger,
    )


def _body() -> dict:
    return {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "stream": False,
    }


# ------------------------------------------------------------------- happy path


def test_success_posts_once_with_bearer_auth_and_stream_false(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-123")
    transport = FakeTransport([_success("hello")])
    client = _client(transport)

    payload, attempts = client.chat(_body())

    assert attempts == 1
    assert payload["message"]["content"] == "hello"
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.url == "http://model.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-test-123"
    assert request.timeout_s == 5.0
    sent = json.loads(request.body)
    assert sent["stream"] is False
    assert sent["model"] == "mock-model"


def test_empty_model_is_filled_from_config():
    transport = FakeTransport([_success()])
    body = _body()
    body["model"] = ""
    client = _client(transport, model="from-config")
    client.chat(body)
    assert json.loads(transport.requests[0].body)["model"] == "from-config"


def test_a_full_chat_completions_url_is_not_doubled():
    transport = FakeTransport([_success()])
    client = _client(transport, base_url="http://model.test/v1/chat/completions")
    client.chat(_body())
    assert transport.requests[0].url == "http://model.test/v1/chat/completions"


# ------------------------------------------------------------------------- retry


def test_429_is_retried_twice_then_succeeds():
    rate_limited = _response(429, {"error": {"type": "rate_limit"}})
    transport = FakeTransport([rate_limited, rate_limited, _success()])
    sleeps: list[float] = []
    client = _client(transport, sleeps=sleeps)

    payload, attempts = client.chat(_body())

    assert attempts == 3
    assert payload["message"]["content"] == "hi"
    assert sleeps == [0.5, 1.0]  # exponential, jitter pinned to 0 in tests


def test_429_retry_after_is_honoured():
    transport = FakeTransport([_response(429, headers={"Retry-After": "2"}), _success()])
    sleeps: list[float] = []
    client = _client(transport, sleeps=sleeps)
    assert client.chat(_body())[1] == 2
    assert sleeps == [2.0]


def test_400_is_never_retried():
    transport = FakeTransport(
        [_response(400, {"error": {"type": "invalid_request_error", "message": "bad"}})]
    )
    sleeps: list[float] = []
    client = _client(transport, sleeps=sleeps)

    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert excinfo.value.retriable is False
    assert excinfo.value.attempts == 1
    assert len(transport.requests) == 1
    assert sleeps == []


def test_401_is_permission_denied_without_retry():
    transport = FakeTransport([_response(401, {"error": {"type": "authentication_error"}})])
    client = _client(transport)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert excinfo.value.code == "PERMISSION_DENIED"
    assert len(transport.requests) == 1


def test_5xx_exhausts_retries_and_reports_unavailable():
    transport = FakeTransport([_response(503), _response(500), _response(502)])
    sleeps: list[float] = []
    client = _client(transport, sleeps=sleeps)

    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())

    assert excinfo.value.code == "UNAVAILABLE"
    assert excinfo.value.retriable is True
    assert excinfo.value.attempts == 3
    assert sleeps == [0.5, 1.0]


def test_429_exhausted_reports_overloaded():
    transport = FakeTransport([_response(429), _response(429), _response(429)])
    client = _client(transport)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert excinfo.value.code == "OVERLOADED"


def test_max_retries_zero_means_one_attempt():
    transport = FakeTransport([_response(429)])
    client = _client(transport, max_retries=0)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert len(transport.requests) == 1
    assert excinfo.value.attempts == 1


def test_upstream_error_message_is_not_echoed():
    transport = FakeTransport(
        [_response(400, {"error": {"type": "bad", "message": "PROMPT_ECHO_SHOULD_NOT_LEAK"}})]
    )
    client = _client(transport)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert "PROMPT_ECHO_SHOULD_NOT_LEAK" not in excinfo.value.message
    assert "bad" in excinfo.value.message


# --------------------------------------------------------------- timeout / connect


def test_timeout_maps_to_timeout_and_is_not_retried():
    transport = FakeTransport([TimeoutError("slow")])
    sleeps: list[float] = []
    client = _client(transport, sleeps=sleeps)

    with pytest.raises(ToolFailure) as excinfo:
        client.chat({**_body(), "model": ""})

    assert excinfo.value.code == "TIMEOUT"
    assert excinfo.value.retriable is True
    assert excinfo.value.attempts == 1
    assert len(transport.requests) == 1
    assert sleeps == []


def test_connection_error_is_retried_then_succeeds():
    transport = FakeTransport(
        [
            urllib.error.URLError(ConnectionRefusedError("refused")),
            urllib.error.URLError(ConnectionRefusedError("refused")),
            _success(),
        ]
    )
    sleeps: list[float] = []
    client = _client(transport, sleeps=sleeps)
    assert client.chat(_body())[1] == 3
    assert sleeps == [0.5, 1.0]


def test_connection_error_exhausted_is_unavailable():
    transport = FakeTransport(
        [
            urllib.error.URLError(ConnectionRefusedError("refused")),
            urllib.error.URLError(ConnectionRefusedError("refused")),
            urllib.error.URLError(ConnectionRefusedError("refused")),
        ]
    )
    client = _client(transport)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert excinfo.value.code == "UNAVAILABLE"
    assert excinfo.value.retriable is True


# ------------------------------------------------------------------------ output


def test_non_json_success_body_is_internal():
    transport = FakeTransport([provider.HttpResponse(200, {}, b"<html>not json</html>")])
    client = _client(transport)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert excinfo.value.code == "INTERNAL"


def test_non_object_json_success_body_is_internal():
    transport = FakeTransport([provider.HttpResponse(200, {}, b"[1,2,3]")])
    client = _client(transport)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert excinfo.value.code == "INTERNAL"


def test_oversized_upstream_body_is_payload_too_large():
    transport = FakeTransport([provider.HttpResponse(200, {}, b"x" * (provider.MAX_RESPONSE_BYTES + 1))])
    client = _client(transport)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert excinfo.value.code == "PAYLOAD_TOO_LARGE"


def test_a_redirect_is_not_followed():
    transport = FakeTransport([provider.HttpResponse(302, {"Location": "http://evil.test/"}, b"")])
    client = _client(transport)
    with pytest.raises(ToolFailure) as excinfo:
        client.chat(_body())
    assert excinfo.value.code == "INTERNAL"
    assert len(transport.requests) == 1


def test_urllib_transport_returns_a_redirect_instead_of_chasing_it():
    """The real transport must not carry Authorization to a second host."""
    import http.server
    import threading

    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        response = provider.urllib_transport(
            provider.HttpRequest(
                url=f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                headers={"Authorization": "Bearer sk-secret", "Content-Type": "application/json"},
                body=b"{}",
                timeout_s=2.0,
            )
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert response.status == 302
    assert hits == []


# -------------------------------------------------------------------- credentials


def test_missing_key_sends_no_authorization_header(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    transport = FakeTransport([_success()])
    _client(transport).chat(_body())
    assert "Authorization" not in transport.requests[0].headers


def test_key_env_name_is_configurable(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-default")
    monkeypatch.setenv("MY_MODEL_KEY", "sk-custom")
    transport = FakeTransport([_success()])
    _client(transport, api_key_env="MY_MODEL_KEY").chat(_body())
    assert transport.requests[0].headers["Authorization"] == "Bearer sk-custom"


# ------------------------------------------------------------------------ config


def test_config_defaults():
    config = provider.config_from_env({})
    assert config.base_url == provider.DEFAULT_BASE_URL
    assert config.model == provider.DEFAULT_MODEL
    assert config.api_key_env == "MODEL_API_KEY"
    assert config.timeout_s == provider.DEFAULT_TIMEOUT_S
    assert config.max_retries == provider.DEFAULT_MAX_RETRIES


def test_config_overrides():
    config = provider.config_from_env(
        {
            "MODEL_BASE_URL": " http://127.0.0.1:8080/v1/ ",
            "MODEL_NAME": "qwen",
            "MODEL_API_KEY_ENV": "QWEN_KEY",
            "MODEL_TIMEOUT_S": "7.5",
            "MODEL_MAX_RETRIES": "0",
        }
    )
    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.model == "qwen"
    assert config.api_key_env == "QWEN_KEY"
    assert config.timeout_s == 7.5
    assert config.max_retries == 0


@pytest.mark.parametrize(
    "env",
    [
        {"MODEL_BASE_URL": "ftp://model.test"},
        {"MODEL_BASE_URL": "http://"},
        {"MODEL_NAME": "   "},
        {"MODEL_TIMEOUT_S": "soon"},
        {"MODEL_TIMEOUT_S": "0"},
        {"MODEL_MAX_RETRIES": "9"},
        {"MODEL_MAX_RETRIES": "two"},
        {"MODEL_RETRY_BASE_S": "-1"},
    ],
)
def test_config_rejects_invalid_values(env):
    with pytest.raises(ToolFailure) as excinfo:
        provider.config_from_env(env)
    assert excinfo.value.code == "INTERNAL"


# -------------------------------------------------------------------------- logs


def test_retry_log_carries_no_request_or_prompt():
    logger = RecordingLogger()
    transport = FakeTransport([_response(429, {"error": {"type": "rate_limit"}}), _success()])
    _client(transport, logger=logger).chat(_body())

    assert len(logger.calls) == 1
    message, fields = logger.calls[0]
    assert message == "model retry"
    assert fields["attempt"] == 1
    assert fields["status"] == "http 429"
    assert fields["delay_ms"] == 500
    assert "PROMPT" not in repr(logger.calls)
