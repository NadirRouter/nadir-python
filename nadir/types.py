"""Nadir SDK response types — lightweight dataclasses, no pydantic dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core message / choice types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class Choice:
    index: int = 0
    message: Optional[Message] = None
    delta: Optional[Message] = None
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Usage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens"),
            cache_creation_input_tokens=data.get("cache_creation_input_tokens"),
            cache_read_input_tokens=data.get("cache_read_input_tokens"),
        )


# ---------------------------------------------------------------------------
# Nadir-specific metadata
# ---------------------------------------------------------------------------

@dataclass
class NadirMetadata:
    """Routing metadata returned by the Nadir API."""

    model_selection_type: Optional[str] = None
    strategy: Optional[str] = None
    selected_model: Optional[str] = None
    complexity_score: Optional[float] = None
    analyzer_used: Optional[str] = None
    tier: Optional[str] = None
    confidence: Optional[float] = None
    cost: Optional[Dict[str, Any]] = None
    benchmark_comparison: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NadirMetadata":
        known_keys = {
            "model_selection_type", "strategy", "selected_model",
            "complexity_score", "analyzer_used", "tier", "confidence",
            "cost", "benchmark_comparison",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            model_selection_type=data.get("model_selection_type"),
            strategy=data.get("strategy"),
            selected_model=data.get("selected_model"),
            complexity_score=data.get("complexity_score"),
            analyzer_used=data.get("analyzer_used"),
            tier=data.get("tier"),
            confidence=data.get("confidence"),
            cost=data.get("cost"),
            benchmark_comparison=data.get("benchmark_comparison"),
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Chat completion responses
# ---------------------------------------------------------------------------

def _parse_message(data: Dict[str, Any]) -> Message:
    return Message(
        role=data.get("role", "assistant"),
        content=data.get("content"),
        tool_calls=data.get("tool_calls"),
    )


def _parse_choices(data: List[Dict[str, Any]], *, streaming: bool = False) -> List[Choice]:
    choices: List[Choice] = []
    for c in data:
        choices.append(Choice(
            index=c.get("index", 0),
            message=_parse_message(c["message"]) if "message" in c else None,
            delta=_parse_message(c["delta"]) if "delta" in c else None,
            finish_reason=c.get("finish_reason"),
        ))
    return choices


@dataclass
class ChatCompletion:
    """Non-streaming chat completion response."""

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None
    nadir_metadata: Optional[NadirMetadata] = None

    # Keep the raw dict for fields we don't model explicitly
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletion":
        usage_data = data.get("usage")
        meta_data = data.get("nadir_metadata")
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=_parse_choices(data.get("choices", [])),
            usage=Usage.from_dict(usage_data) if usage_data else None,
            nadir_metadata=NadirMetadata.from_dict(meta_data) if meta_data else None,
            _raw=data,
        )


@dataclass
class ChatCompletionChunk:
    """A single streamed chunk (SSE event)."""

    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletionChunk":
        usage_data = data.get("usage")
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=_parse_choices(data.get("choices", []), streaming=True),
            usage=Usage.from_dict(usage_data) if usage_data else None,
        )
