"""Parity with core's builtin OpenAI adapter (M6 gate 1).

The point of M6 is "core changes one config line and behaves the same". This
test pins that: the same OpenAI payload goes through our `normalize_response`
and through core's `parse_chunk` (the builtin adapter's pure translator), and
the content / tool-call structure / finish reason must match exactly.

Skipped when the core checkout is not reachable, so the module's own suite still
runs on a bare Python. In the Vade dev tree it runs by default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))


def _core_candidates() -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("VADE_CORE_PATH"):
        candidates.append(Path(os.environ["VADE_CORE_PATH"]))
    candidates.append(MODULE_ROOT.parent / "Vade" / "core")
    candidates.append(MODULE_ROOT / "core")
    return candidates


CORE_DIR = next((path for path in _core_candidates() if (path / "vade" / "llm").is_dir()), None)
if CORE_DIR is not None:
    sys.path.insert(0, str(CORE_DIR))

pytest.importorskip("vade.llm.openai_compat", reason="core's builtin adapter is not on sys.path")

from vade.llm.base import flatten_usage  # noqa: E402
from vade.llm.openai_compat import parse_chunk  # noqa: E402

from model_module import tools  # noqa: E402

PAYLOAD = {
    "id": "chatcmpl-parity",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I will read the file.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fs__read_file", "arguments": '{"path":"/a"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
}


def test_message_and_tool_calls_match_the_builtin_adapter():
    ours = tools.normalize_response(PAYLOAD)
    message = PAYLOAD["choices"][0]["message"]
    theirs = parse_chunk(
        {"choices": [{"delta": message, "finish_reason": PAYLOAD["choices"][0]["finish_reason"]}]}
    )

    assert ours["message"]["content"] == theirs.text
    assert [
        (call["id"], call["function"]["name"], call["function"]["arguments"])
        for call in ours["message"]["tool_calls"]
    ] == [(call.id, call.name, call.arguments) for call in theirs.tool_calls]
    assert ours["finish_reason"] == theirs.finish_reason


def test_usage_matches_the_builtin_flattening():
    ours = tools.normalize_response(PAYLOAD)
    flat = flatten_usage(PAYLOAD["usage"])
    assert ours["usage"] == {
        "prompt_tokens": flat["prompt_tokens"],
        "completion_tokens": flat["completion_tokens"],
    }


def test_plain_text_answer_matches():
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    ours = tools.normalize_response(payload)
    theirs = parse_chunk(
        {"choices": [{"delta": payload["choices"][0]["message"], "finish_reason": "stop"}]}
    )
    assert ours["message"]["content"] == theirs.text
    assert ours["message"]["tool_calls"] == []
    assert ours["finish_reason"] == theirs.finish_reason
