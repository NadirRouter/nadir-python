"""Tests for Nadir SDK response types."""

from nadir.types import ChatCompletion, ChatCompletionChunk, Message, Choice, Usage, NadirMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMPLETION_DICT = {
    "id": "req-abc123",
    "object": "chat.completion",
    "created": 1711234567,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "The answer is 4."},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    },
    "nadir_metadata": {
        "strategy": "smart-routing",
        "selected_model": "gpt-4o-mini",
        "tier": "simple",
        "confidence": 0.92,
        "complexity_score": 0.15,
        "analyzer_used": "trained",
    },
}

CHUNK_DICT = {
    "id": "req-abc123",
    "object": "chat.completion.chunk",
    "created": 1711234567,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": "Hello"},
            "finish_reason": None,
        }
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chat_completion_from_dict():
    comp = ChatCompletion.from_dict(COMPLETION_DICT)
    assert comp.id == "req-abc123"
    assert comp.model == "gpt-4o-mini"
    assert len(comp.choices) == 1
    assert comp.choices[0].message.content == "The answer is 4."
    assert comp.choices[0].finish_reason == "stop"


def test_chat_completion_usage():
    comp = ChatCompletion.from_dict(COMPLETION_DICT)
    assert comp.usage is not None
    assert comp.usage.prompt_tokens == 12
    assert comp.usage.completion_tokens == 5
    assert comp.usage.total_tokens == 17


def test_chat_completion_metadata():
    comp = ChatCompletion.from_dict(COMPLETION_DICT)
    meta = comp.nadir_metadata
    assert meta is not None
    assert meta.tier == "simple"
    assert meta.confidence == 0.92
    assert meta.complexity_score == 0.15
    assert meta.selected_model == "gpt-4o-mini"
    assert meta.analyzer_used == "trained"
    assert meta.strategy == "smart-routing"


def test_chat_completion_no_metadata():
    data = {**COMPLETION_DICT}
    del data["nadir_metadata"]
    comp = ChatCompletion.from_dict(data)
    assert comp.nadir_metadata is None


def test_chat_completion_no_usage():
    data = {**COMPLETION_DICT}
    del data["usage"]
    comp = ChatCompletion.from_dict(data)
    assert comp.usage is None


def test_chat_completion_chunk():
    chunk = ChatCompletionChunk.from_dict(CHUNK_DICT)
    assert chunk.id == "req-abc123"
    assert chunk.object == "chat.completion.chunk"
    assert len(chunk.choices) == 1
    assert chunk.choices[0].delta is not None
    assert chunk.choices[0].delta.content == "Hello"
    assert chunk.choices[0].message is None


def test_message_to_dict():
    msg = Message(role="user", content="hi")
    d = msg.to_dict()
    assert d == {"role": "user", "content": "hi"}


def test_message_to_dict_no_content():
    msg = Message(role="assistant")
    d = msg.to_dict()
    assert d == {"role": "assistant"}


def test_usage_from_dict_with_extras():
    usage = Usage.from_dict({
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "reasoning_tokens": 5,
        "cache_creation_input_tokens": 8,
        "cache_read_input_tokens": 2,
    })
    assert usage.reasoning_tokens == 5
    assert usage.cache_creation_input_tokens == 8
    assert usage.cache_read_input_tokens == 2


def test_nadir_metadata_extra_fields():
    meta = NadirMetadata.from_dict({
        "tier": "complex",
        "custom_field": "hello",
        "another": 42,
    })
    assert meta.tier == "complex"
    assert meta.extra == {"custom_field": "hello", "another": 42}


def test_chat_completion_from_empty_dict():
    comp = ChatCompletion.from_dict({})
    assert comp.id == ""
    assert comp.model == ""
    assert comp.choices == []
    assert comp.usage is None
    assert comp.nadir_metadata is None


def test_choice_with_tool_calls():
    data = {
        **COMPLETION_DICT,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    comp = ChatCompletion.from_dict(data)
    assert comp.choices[0].message.content is None
    assert len(comp.choices[0].message.tool_calls) == 1
    assert comp.choices[0].finish_reason == "tool_calls"
