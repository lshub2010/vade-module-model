"""Pure-layer tests — stdlib + pytest only, NO SDK import.

If this file ever needs `vade_sdk`, the "write and test the module before the
SDK" guarantee is gone.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_module import tools  # noqa: E402


def _openai_payload(
    content: str | None = "hello",
    *,
    tool_calls: list | None = None,
    usage: dict | None = None,
    finish_reason: str = "stop",
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage if usage is not None else {"prompt_tokens": 12, "completion_tokens": 34},
        "system_fingerprint": "fp_vendor",
    }


def _message(text: str = "hi", role: str = "user") -> dict:
    return {"role": role, "content": text}


# --------------------------------------------------------------- self-sufficiency


def test_no_sdk_import_in_tools():
    """AST check, not a substring scan: docstrings may *mention* the SDK."""
    tree = ast.parse(Path(tools.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if name == "vade_sdk" or name.startswith("vade_sdk.")]
    assert not [name for name in imported if name == "vade" or name.startswith("vade.")]


# ------------------------------------------------------------------------ digest


def test_messages_digest_is_short_and_hides_content():
    digest = tools.messages_digest([_message("SUPER_SECRET_PROMPT")])
    assert len(digest) == 16
    assert "SUPER_SECRET" not in digest
    assert all(char in "0123456789abcdef" for char in digest)


def test_messages_digest_is_key_order_independent_and_content_sensitive():
    first = tools.messages_digest([{"role": "user", "content": "a"}])
    reordered = tools.messages_digest([{"content": "a", "role": "user"}])
    changed = tools.messages_digest([{"role": "user", "content": "b"}])
    assert first == reordered
    assert first != changed


def test_messages_digest_rejects_non_serialisable_content():
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.messages_digest([{"role": "user", "content": object()}])
    assert excinfo.value.code == "INVALID_ARGUMENT"


# -------------------------------------------------------------------- validation


def test_accepts_a_minimal_chat():
    validated = tools.validate_chat([_message()])
    assert validated["messages"] == [_message()]
    assert validated["tools"] == []
    assert validated["temperature"] == 0.7
    assert validated["model"] == ""


def test_rejects_messages_that_are_not_an_array():
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.validate_messages({"role": "user", "content": "hi"})
    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert "array" in excinfo.value.message


def test_rejects_empty_and_none_messages():
    for value in ([], None):
        with pytest.raises(tools.ToolFailure) as excinfo:
            tools.validate_messages(value)
        assert excinfo.value.code == "INVALID_ARGUMENT"


def test_rejects_more_than_200_messages():
    messages = [_message(f"m{i}") for i in range(tools.MAX_MESSAGES + 1)]
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.validate_messages(messages)
    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert "200" in excinfo.value.message


def test_accepts_exactly_200_messages():
    messages = [_message(f"m{i}") for i in range(tools.MAX_MESSAGES)]
    assert len(tools.validate_messages(messages)) == 200


@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_accepts_every_declared_role(role):
    assert tools.validate_messages([_message(role=role)])[0]["role"] == role


def test_rejects_an_unknown_role():
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.validate_messages([_message(role="developer")])
    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert "developer" in excinfo.value.message


def test_rejects_non_string_content_and_non_object_messages():
    with pytest.raises(tools.ToolFailure):
        tools.validate_messages([{"role": "user", "content": {"text": "hi"}}])
    with pytest.raises(tools.ToolFailure):
        tools.validate_messages(["user: hi"])


def test_rejects_a_single_message_over_64k():
    oversized = "_" * (tools.MAX_MESSAGE_BYTES + 1)
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.validate_messages([_message(oversized)])
    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert "65536" in excinfo.value.message


def test_rejects_more_than_512k_in_total():
    # 9 x 60 KiB = 540 KiB: each message is legal, the array is not.
    messages = [_message("x" * (60 * 1024)) for _ in range(9)]
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.validate_messages(messages)
    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert "524288" in excinfo.value.message


def test_rejects_non_serialisable_message_values():
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.validate_messages([{"role": "user", "content": "hi", "extra": object()}])
    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_tools_must_be_an_array_of_objects():
    assert tools.validate_tools(None) == []
    assert tools.validate_tools([]) == []
    with pytest.raises(tools.ToolFailure):
        tools.validate_tools({"type": "function"})
    with pytest.raises(tools.ToolFailure):
        tools.validate_tools(["not-a-tool"])
    with pytest.raises(tools.ToolFailure):
        tools.validate_tools([{} for _ in range(tools.MAX_TOOLS + 1)])


def test_temperature_is_bounded_and_rejects_bool():
    assert tools.validate_temperature(0) == 0.0
    assert tools.validate_temperature(1.5) == 1.5
    for bad in (-0.1, 2.1, True, "0.7"):
        with pytest.raises(tools.ToolFailure) as excinfo:
            tools.validate_temperature(bad)
        assert excinfo.value.code == "INVALID_ARGUMENT"


def test_model_override_is_trimmed_and_bounded():
    assert tools.validate_model("  gpt-4o  ") == "gpt-4o"
    assert tools.validate_model(None) == ""
    with pytest.raises(tools.ToolFailure):
        tools.validate_model("m" * (tools.MAX_MODEL_CHARS + 1))
    with pytest.raises(tools.ToolFailure):
        tools.validate_model(123)


def test_build_request_body_is_openai_shaped_and_non_streaming():
    body = tools.build_request_body([_message()], [], 0.2, "gpt-4o")
    assert body == {
        "model": "gpt-4o",
        "messages": [_message()],
        "temperature": 0.2,
        "stream": False,
    }


def test_build_request_body_includes_tools_only_when_present():
    spec = {"type": "function", "function": {"name": "fs_read", "parameters": {"type": "object"}}}
    body = tools.build_request_body([_message()], [spec], 0.7, "m")
    assert body["tools"] == [spec]
    assert "tools" not in tools.build_request_body([_message()], [], 0.7, "m")


# --------------------------------------------------------------------- normalise


def test_normalizes_the_openai_shape_into_the_m6_contract():
    result = tools.normalize_response(_openai_payload("hi"))
    assert result == {
        "message": {"role": "assistant", "content": "hi", "tool_calls": []},
        "usage": {"prompt_tokens": 12, "completion_tokens": 34},
        "finish_reason": "stop",
    }


def test_arguments_are_always_a_string():
    payload = _openai_payload(
        None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fs__read_file", "arguments": {"path": "/a", "n": 2}},
            }
        ],
        finish_reason="tool_calls",
    )
    call = tools.normalize_response(payload)["message"]["tool_calls"][0]
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"path": "/a", "n": 2}
    assert call["function"]["arguments"] == '{"n":2,"path":"/a"}'


def test_null_content_becomes_an_empty_string_and_missing_ids_are_filled():
    payload = _openai_payload(
        None,
        tool_calls=[
            {"type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"type": "function", "function": {"name": "b", "arguments": ""}},
        ],
        finish_reason="tool_calls",
    )
    result = tools.normalize_response(payload)
    assert result["message"]["content"] == ""
    assert [c["id"] for c in result["message"]["tool_calls"]] == ["call_1", "call_2"]
    assert all(c["type"] == "function" for c in result["message"]["tool_calls"])


def test_flat_non_openai_shape_is_normalised():
    payload = {
        "message": {"role": "assistant", "content": "flat"},
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        "finish_reason": "stop",
    }
    assert tools.normalize_response(payload)["message"]["content"] == "flat"


def test_content_blocks_and_tool_use_are_normalised():
    payload = {
        "content": [
            {"type": "text", "text": "Let me look"},
            {"type": "tool_use", "id": "tu_1", "name": "fs_read", "input": {"path": "/a"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    result = tools.normalize_response(payload)
    assert result["message"]["content"] == "Let me look"
    assert result["message"]["tool_calls"] == [
        {
            "id": "tu_1",
            "type": "function",
            "function": {"name": "fs_read", "arguments": '{"path":"/a"}'},
        }
    ]
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}
    assert result["finish_reason"] == "tool_calls"


def test_top_level_tool_call_fields_are_accepted_but_not_forwarded():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "x", "name": "f", "input": {"a": 1}, "vendor": "y"}],
                },
                "finish_reason": "end_turn",
            }
        ],
        "usage": {},
    }
    call = tools.normalize_response(payload)["message"]["tool_calls"][0]
    assert call == {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}


def test_vendor_fields_are_dropped_everywhere():
    payload = _openai_payload(
        "hi",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "x_vendor": True,
                "function": {"name": "f", "arguments": "{}", "parsed": {"k": 1}},
            }
        ],
        finish_reason="tool_calls",
    )
    result = tools.normalize_response(payload)
    assert set(result) == {"message", "usage", "finish_reason"}
    assert set(result["message"]) == {"role", "content", "tool_calls"}
    assert set(result["message"]["tool_calls"][0]) == {"id", "type", "function"}
    assert set(result["message"]["tool_calls"][0]["function"]) == {"name", "arguments"}


def test_finish_reason_and_usage_aliases_are_mapped():
    assert (
        tools.normalize_response(_openai_payload("x", finish_reason="max_tokens"))["finish_reason"] == "length"
    )
    assert (
        tools.normalize_response(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "end_turn"}]}
        )["finish_reason"]
        == "stop"
    )
    no_reason = tools.normalize_response({"choices": [{"message": {"content": "x"}}]})
    assert no_reason["finish_reason"] == "stop"
    assert no_reason["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}


def test_normalize_rejects_malformed_upstream_shapes():
    for payload in ("not-an-object", {}, {"choices": ["nope"]}, {"choices": []}):
        with pytest.raises(tools.ToolFailure) as excinfo:
            tools.normalize_response(payload)
        assert excinfo.value.code == "INTERNAL"

    with pytest.raises(tools.ToolFailure):
        tools.normalize_response({"choices": [{"message": {"tool_calls": {"bad": 1}}}]})
    with pytest.raises(tools.ToolFailure):
        tools.normalize_response({"choices": [{"message": {"tool_calls": [{"function": {}}]}}]})


def test_error_objects_are_not_mistaken_for_answers():
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.normalize_response({"error": {"type": "invalid_request_error", "message": "nope"}})
    assert excinfo.value.code == "INTERNAL"
    assert "invalid_request_error" in excinfo.value.message


# ------------------------------------------------------------------ result budget


def test_a_result_within_budget_is_returned_untouched():
    result = tools.normalize_response(_openai_payload("small"))
    assert tools.fit_result(result) == result
    assert "truncated" not in tools.fit_result(result)


def test_oversized_content_is_truncated_to_the_budget():
    result = tools.normalize_response(_openai_payload("x" * 60_000))
    assert tools.result_bytes(result) > tools.MAX_RESULT_BYTES
    fitted = tools.fit_result(result)
    assert fitted["truncated"] is True
    assert tools.result_bytes(fitted) <= tools.MAX_RESULT_BYTES
    assert len(fitted["message"]["content"]) < 60_000
    assert fitted["message"]["content"] == "x" * len(fitted["message"]["content"])
    # Everything except content survives.
    assert fitted["usage"] == result["usage"]
    assert fitted["finish_reason"] == result["finish_reason"]


def test_budget_counts_json_escapes_not_source_characters():
    # 30k quotes serialise to 60k bytes once escaped; a naive len() would
    # happily return 30k characters and overflow the frame.
    result = tools.normalize_response(_openai_payload('"' * 30_000))
    fitted = tools.fit_result(result)
    assert len(fitted["message"]["content"]) < 25_000
    assert tools.result_bytes(fitted) <= tools.MAX_RESULT_BYTES


def test_truncation_never_splits_a_surrogate_pair():
    result = tools.normalize_response(_openai_payload("a" * 100 + "\U0001f600" * 30_000))
    fitted = tools.fit_result(result)
    content = fitted["message"]["content"]
    content.encode("utf-8")  # raises UnicodeEncodeError on a lone surrogate
    assert not (content and 0xD800 <= ord(content[-1]) <= 0xDBFF)


def test_tool_calls_are_never_dropped_or_mangled():
    arguments = json.dumps({"blob": "x" * 60_000})
    result = {
        "message": {
            "role": "assistant",
            "content": "prefix",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": arguments}}
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        "finish_reason": "tool_calls",
    }
    with pytest.raises(tools.ToolFailure) as excinfo:
        tools.fit_result(result)
    assert excinfo.value.code == "PAYLOAD_TOO_LARGE"
