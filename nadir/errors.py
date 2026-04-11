"""Nadir SDK error classes."""

from __future__ import annotations

from typing import Any, Optional


class NadirError(Exception):
    """Base error for all Nadir API errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        body: Any = None,
        error_type: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.error_type = error_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(status_code={self.status_code}, message={str(self)!r})"


class AuthenticationError(NadirError):
    """Raised on 401 Unauthorized — invalid or missing API key."""


class RateLimitError(NadirError):
    """Raised on 429 Too Many Requests."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        body: Any = None,
        error_type: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, status_code=status_code, body=body, error_type=error_type)
        self.retry_after = retry_after


class APIError(NadirError):
    """Raised on server errors (4xx/5xx) not covered by a more specific class."""


class ConnectionError(NadirError):
    """Raised when the SDK cannot reach the Nadir API."""
