"""OpenAI-compatible `/chat/completions` client — stdlib urllib only.

Why not `httpx`/`requests`: the module is a separate deployable and its only
I/O is one POST. A dependency here would be a dependency in every image that
runs a model module, for a client that is 80 lines of `urllib`.

Rules (M6 §六):
  * the API key is read ONLY from the environment variable named by
    `MODEL_API_KEY_ENV` (default `MODEL_API_KEY`) — never from config, never
    logged, never in an error message;
  * retry 429 / 5xx / connection errors at most `MODEL_MAX_RETRIES` times with
    exponential backoff + jitter; 4xx is never retried;
  * a socket timeout is NOT retried: one attempt can consume the whole 120 s
    tool budget, so a second attempt would only guarantee the SDK deadline
    fires first. It surfaces as `TIMEOUT` with `retriable: true` — core may
    decide to start a new turn;
  * the total wall clock is capped at `MAX_TOTAL_S` (118 s), 2 s below the
    tool's 120 s deadline, so the module's own structured `TIMEOUT` wins the
    race against the SDK's generic one.

`transport` is injectable: every retry/status/timeout test runs without a
socket, and `tests/test_server.py` exercises the real urllib path against a
local `http.server`.
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from . import __version__
from .tools import ToolFailure, normalize_response

__all__ = [
    "ModelConfig",
    "config_from_env",
    "HttpRequest",
    "HttpResponse",
    "ModelClient",
    "urllib_transport",
    "RETRIABLE_STATUS_NOTE",
    "MAX_TOTAL_S",
]

#: The tool's own deadline is 120 s (manifest). Leave the SDK/core 2 s to
#: serialise the error instead of being killed by `asyncio.wait_for`.
MAX_TOTAL_S = 118.0

#: A `Retry-After` larger than this is ignored: waiting 60 s inside a 120 s
#: tool call is indistinguishable from a timeout, only slower.
MAX_RETRY_AFTER_S = 10.0

#: A completion body larger than this is not an answer we can use; reading it
#: whole would let a broken upstream exhaust the module's memory.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

#: 408 and 429 plus every 5xx. Anything else in 4xx is the caller's fault and
#: replaying it would burn the prompt twice for the same answer.
RETRIABLE_STATUS_NOTE = "408, 429, 5xx"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_S = 0.5


@dataclass(frozen=True)
class ModelConfig:
    """Resolved configuration. The API key is NOT a field: only its env name."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key_env: str = "MODEL_API_KEY"
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_base_s: float = DEFAULT_RETRY_BASE_S

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"


def _env_float(env: Mapping[str, str], name: str, default: float, *, low: float, high: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ToolFailure("INTERNAL", f"{name} must be a number, got {raw!r}") from None
    if not low <= value <= high:
        raise ToolFailure("INTERNAL", f"{name} must be within [{low}, {high}], got {value}")
    return value


def _env_int(env: Mapping[str, str], name: str, default: int, *, low: int, high: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ToolFailure("INTERNAL", f"{name} must be an integer, got {raw!r}") from None
    if not low <= value <= high:
        raise ToolFailure("INTERNAL", f"{name} must be within [{low}, {high}], got {value}")
    return value


def config_from_env(env: Mapping[str, str] | None = None) -> ModelConfig:
    """Read `MODEL_*` from the environment (per call, so a restart is not needed)."""
    environ = os.environ if env is None else env
    base_url = (environ.get("MODEL_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ToolFailure(
            "INTERNAL", "MODEL_BASE_URL must be an absolute http(s) URL (e.g. https://api.openai.com/v1)"
        )
    model = (environ.get("MODEL_NAME") or DEFAULT_MODEL).strip()
    if not model:
        raise ToolFailure("INTERNAL", "MODEL_NAME must not be empty")
    api_key_env = (environ.get("MODEL_API_KEY_ENV") or "MODEL_API_KEY").strip() or "MODEL_API_KEY"
    return ModelConfig(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        timeout_s=_env_float(environ, "MODEL_TIMEOUT_S", DEFAULT_TIMEOUT_S, low=0.1, high=600.0),
        max_retries=_env_int(environ, "MODEL_MAX_RETRIES", DEFAULT_MAX_RETRIES, low=0, high=5),
        retry_base_s=_env_float(environ, "MODEL_RETRY_BASE_S", DEFAULT_RETRY_BASE_S, low=0.0, high=10.0),
    )


@dataclass(frozen=True)
class HttpRequest:
    url: str
    headers: Mapping[str, str]
    body: bytes
    timeout_s: float


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects instead of carrying `Authorization` to another host.

    A 3xx becomes an `HTTPError` and is returned as a response, so the caller
    sees `INTERNAL` rather than a completion from wherever the upstream pointed.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def urllib_transport(request: HttpRequest) -> HttpResponse:
    """One POST. HTTPError is a response (4xx/5xx), not an exception to retry."""
    req = urllib.request.Request(
        request.url,
        data=request.body,
        headers=dict(request.headers),
        method="POST",
    )
    try:
        with _OPENER.open(req, timeout=request.timeout_s) as response:
            return HttpResponse(
                status=int(getattr(response, "status", 200)),
                headers=dict(response.headers.items()),
                # +1 is enough to detect "over the cap" without buffering it.
                body=response.read(MAX_RESPONSE_BYTES + 1),
            )
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except OSError:  # a broken error body is still an error response
            body = b""
        headers = dict(exc.headers.items()) if exc.headers else {}
        return HttpResponse(status=int(exc.code), headers=headers, body=body)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout))


def _is_retriable_status(status: int) -> bool:
    return status == 408 or status == 429 or 500 <= status <= 599


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    lowered = {str(key).lower(): value for key, value in headers.items()}
    raw = lowered.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except ValueError:  # HTTP-date form is legal but not worth parsing here
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_S)


def _error_kind(body: bytes) -> str:
    """Only a provider's error type/code — never its message (it may echo input)."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return "non-json body"
    if not isinstance(payload, Mapping):
        return "non-object body"
    error = payload.get("error")
    if isinstance(error, Mapping):
        for key in ("type", "code", "status"):
            value = error.get(key)
            if isinstance(value, str) and value:
                return value[:80]
        return "error object"
    if isinstance(error, str):
        return error[:80]
    return "error"


class ModelClient:
    """Synchronous client (the SDK runs sync tools in a worker thread)."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        transport: Callable[[HttpRequest], HttpResponse] | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        jitter: Callable[[], float] | None = None,
        logger: Any = None,
    ) -> None:
        self.config = config
        self._transport = transport or urllib_transport
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._jitter = jitter or random.random
        self._logger = logger

    # ------------------------------------------------------------------ public
    def chat(self, body: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        """POST the body; return `(normalised result, attempts)`.

        Normalisation happens here, before the payload can reach `server.py`:
        a vendor-shaped `message`/`content` never travels past the provider
        boundary (M6 §五). `tools.normalize_response` is the pure function that
        does the work; this method only decides when a body is an answer.
        """
        if not body.get("model"):
            # `model` is a per-call override: an empty value means MODEL_NAME.
            body = {**body, "model": self.config.model}
        data = self._encode(body)
        deadline = self._clock() + MAX_TOTAL_S
        attempts = 0
        while True:
            attempts += 1
            remaining = deadline - self._clock()
            if remaining <= 0.05:
                failure = self._failure(
                    "TIMEOUT", f"model call exceeded the module's {MAX_TOTAL_S:.0f} s budget", attempts
                )
                raise failure
            request = HttpRequest(
                url=self.config.endpoint,
                headers=self._headers(),
                body=data,
                # Never let one attempt outlive the total budget.
                timeout_s=min(self.config.timeout_s, remaining),
            )
            try:
                response = self._transport(request)
            except (TimeoutError, socket.timeout) as exc:
                raise self._failure(
                    "TIMEOUT",
                    f"upstream did not answer within {request.timeout_s:.1f} s",
                    attempts,
                ) from None
            except urllib.error.URLError as exc:
                if _is_timeout(exc):
                    raise self._failure(
                        "TIMEOUT",
                        f"upstream did not answer within {request.timeout_s:.1f} s",
                        attempts,
                    ) from None
                delay = (
                    self._retry_delay(attempts, None, deadline)
                    if attempts <= self.config.max_retries
                    else None
                )
                if delay is not None:
                    self._log_retry(attempts, "connect", delay)
                    if delay > 0:
                        self._sleep(delay)
                    continue
                raise self._failure(
                    "UNAVAILABLE", f"cannot reach the model endpoint ({type(exc.reason).__name__})", attempts
                ) from None
            except OSError as exc:
                delay = (
                    self._retry_delay(attempts, None, deadline)
                    if attempts <= self.config.max_retries
                    else None
                )
                if delay is not None:
                    self._log_retry(attempts, "connect", delay)
                    if delay > 0:
                        self._sleep(delay)
                    continue
                raise self._failure(
                    "UNAVAILABLE", f"cannot reach the model endpoint ({type(exc).__name__})", attempts
                ) from None

            if 200 <= response.status < 300:
                if len(response.body) > MAX_RESPONSE_BYTES:
                    raise self._failure(
                        "PAYLOAD_TOO_LARGE",
                        f"upstream response exceeds {MAX_RESPONSE_BYTES} bytes",
                        attempts,
                    )
                return normalize_response(self._decode(response)), attempts

            if _is_retriable_status(response.status) and attempts <= self.config.max_retries:
                retry_after = _retry_after_seconds(response.headers)
                delay = self._retry_delay(attempts, retry_after, deadline)
                if delay is not None:
                    self._log_retry(attempts, f"http {response.status}", delay)
                    if delay > 0:
                        self._sleep(delay)
                    continue
            raise self._status_failure(response.status, response.body, attempts)

    # ----------------------------------------------------------------- helpers
    def _encode(self, body: Mapping[str, Any]) -> bytes:
        try:
            return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ToolFailure("INVALID_ARGUMENT", f"request body is not JSON-serialisable: {exc}") from None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"vade-model-module/{__version__}",
        }
        key = os.environ.get(self.config.api_key_env)
        if key and key.strip():
            headers["Authorization"] = f"Bearer {key.strip()}"
        return headers

    def _delay(self, attempts: int) -> float:
        # attempts is 1-based: after the first failure wait base * 2^0.
        base = self.config.retry_base_s * (2 ** max(0, attempts - 1))
        return base + self._jitter() * self.config.retry_base_s

    def _retry_delay(self, attempts: int, retry_after: float | None, deadline: float) -> float | None:
        """The wait before the next attempt, or None when the budget cannot fit it."""
        delay = retry_after if retry_after is not None else self._delay(attempts)
        if delay >= deadline - self._clock():
            return None
        return max(0.0, delay)

    def _decode(self, response: HttpResponse) -> dict[str, Any]:
        try:
            payload = json.loads(response.body)
        except (ValueError, UnicodeDecodeError):
            raise ToolFailure("INTERNAL", "upstream returned a non-JSON body") from None
        if not isinstance(payload, dict):
            raise ToolFailure("INTERNAL", "upstream returned a non-object JSON body")
        return payload

    def _failure(self, code: str, message: str, attempts: int) -> ToolFailure:
        failure = ToolFailure(code, message)
        failure.attempts = attempts
        return failure

    def _status_failure(self, status: int, body: bytes, attempts: int) -> ToolFailure:
        kind = _error_kind(body)
        if status in (401, 403):
            return self._failure(
                "PERMISSION_DENIED",
                f"upstream rejected the credentials (HTTP {status}); check the env var "
                f"{self.config.api_key_env}",
                attempts,
            )
        if status == 404:
            return self._failure(
                "NOT_FOUND", f"upstream has no such model or endpoint (HTTP 404, {kind})", attempts
            )
        if status == 413:
            return self._failure("PAYLOAD_TOO_LARGE", f"upstream rejected the body as too large (HTTP 413)", attempts)
        if status == 408:
            return self._failure("TIMEOUT", f"upstream timed out (HTTP 408, {kind})", attempts)
        if status == 429:
            return self._failure(
                "OVERLOADED",
                f"upstream rate limit was still exhausted after {attempts} attempt(s) (HTTP 429)",
                attempts,
            )
        if 500 <= status:
            return self._failure(
                "UNAVAILABLE",
                f"upstream failed after {attempts} attempt(s) (HTTP {status}, {kind})",
                attempts,
            )
        if 400 <= status:
            return self._failure("INVALID_ARGUMENT", f"upstream rejected the request (HTTP {status}, {kind})", attempts)
        return self._failure("INTERNAL", f"upstream returned unexpected HTTP {status}", attempts)

    def _log_retry(self, attempts: int, status: str, delay: float) -> None:
        """Structured retry line. Fields only — never the request or prompt."""
        logger = self._logger
        if logger is None:
            return
        try:
            logger.warning(
                "model retry",
                tool="chat",
                attempt=attempts,
                status=status,
                delay_ms=int(delay * 1000),
                max_attempts=self.config.max_retries + 1,
            )
        except Exception:  # a broken logger must not fail a model call
            pass
