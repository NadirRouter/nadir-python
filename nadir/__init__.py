"""Nadir Python SDK — OpenAI-compatible client for the Nadir LLM router."""

__version__ = "0.1.0"

from .client import NadirClient
from .async_client import AsyncNadirClient
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Message,
    NadirMetadata,
    Usage,
)
from .errors import (
    APIError,
    AuthenticationError,
    ConnectionError,
    NadirError,
    RateLimitError,
)

__all__ = [
    "NadirClient",
    "AsyncNadirClient",
    "ChatCompletion",
    "ChatCompletionChunk",
    "Choice",
    "Message",
    "NadirMetadata",
    "Usage",
    "NadirError",
    "APIError",
    "AuthenticationError",
    "ConnectionError",
    "RateLimitError",
]
