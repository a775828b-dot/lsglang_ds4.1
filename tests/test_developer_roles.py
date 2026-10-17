"""OpenAI developer-role compatibility for the patched 1017 encoder."""
import copy
import importlib.util
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def enc():
    root = os.environ.get("LSGLANG")
    if not root:
        pytest.fail("Set LSGLANG to the patched Lsglang checkout")
    path = Path(root) / "python/sglang/srt/entrypoints/openai/encoding_dsv41.py"
    spec = importlib.util.spec_from_file_location("developer_role_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native(messages):
    return [dict(m, role="system") if m.get("role") == "developer" else m for m in messages]


@pytest.mark.parametrize("mode", ["chat", "thinking"])
@pytest.mark.parametrize("blocks", [False, True])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("position", ["leading", "mid"])
def test_developer_matches_native_system_without_losing_instructions(enc, mode, blocks, tools, position):
    app = {"role": "developer", "name": "application", "content": [{"type": "text", "text": "APP_MARKER"}] if blocks else "APP_MARKER"}
    if tools:
        app["tools"] = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
    messages = [app, {"role": "user", "content": "QUESTION"}]
    if position == "mid":
        messages = [{"role": "system", "content": "SYSTEM_MARKER"}, {"role": "user", "content": "BEFORE"}, {"role": "assistant", "content": "ANSWER", "reasoning_content": "OLD_REASONING"}] + messages
    before = copy.deepcopy(messages)
    actual = enc.encode_messages(messages, thinking_mode=mode)
    assert actual == enc.encode_messages(native(messages), thinking_mode=mode)
    assert "APP_MARKER" in actual
    if position == "mid":
        assert actual.index("SYSTEM_MARKER") < actual.index("APP_MARKER")
    if tools:
        assert "lookup" in actual
    assert messages == before


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_context_developer_is_normalized_without_mutation(enc, mode):
    context = [{"role": "developer", "content": "APP_CONTEXT"}, {"role": "user", "content": "FIRST"}, {"role": "assistant", "content": "DONE"}]
    messages = [{"role": "developer", "content": "APP_CURRENT"}, {"role": "user", "content": "NEXT"}]
    before = copy.deepcopy((messages, context))
    assert enc.encode_messages(messages, mode, context=context) == enc.encode_messages(native(messages), mode, context=native(context))
    assert (messages, context) == before


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_existing_system_and_developer_both_survive(enc, mode):
    messages = [{"role": "system", "content": "SYSTEM_FIRST"}, {"role": "developer", "content": "APP_SECOND"}, {"role": "user", "content": "QUESTION"}]
    text = enc.encode_messages(messages, mode)
    assert text.index("SYSTEM_FIRST") < text.index("APP_SECOND")
    assert text == enc.encode_messages(native(messages), mode)


def test_unknown_roles_are_not_promoted(enc):
    with pytest.raises(NotImplementedError, match="Unknown role"):
        enc.encode_messages([{"role": "unknown", "content": "X"}], "chat")


def test_developer_keeps_placeholder_validation(enc):
    with pytest.raises(ValueError, match="image"):
        enc.encode_messages([{"role": "developer", "content": enc.IMAGE_PLACEHOLDER}, {"role": "user", "content": "QUESTION"}], "chat")
