"""Tests for Nadir SDK client construction and error handling."""

import os
import pytest
from unittest.mock import patch

from nadir import NadirClient, AsyncNadirClient
from nadir.errors import NadirError, AuthenticationError, RateLimitError, APIError, ConnectionError


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def test_client_default_base_url():
    client = NadirClient(api_key="ndr_test")
    assert client._base_url == "https://api.getnadir.com"
    client.close()


def test_client_custom_base_url():
    client = NadirClient(api_key="ndr_test", base_url="http://localhost:8856/")
    assert client._base_url == "http://localhost:8856"  # trailing slash stripped
    client.close()


def test_client_env_api_key():
    with patch.dict(os.environ, {"NADIR_API_KEY": "ndr_from_env"}):
        client = NadirClient()
        assert client._api_key == "ndr_from_env"
        client.close()


def test_client_env_base_url():
    with patch.dict(os.environ, {"NADIR_BASE_URL": "http://custom:9000"}):
        client = NadirClient(api_key="ndr_test")
        assert client._base_url == "http://custom:9000"
        client.close()


def test_client_explicit_overrides_env():
    with patch.dict(os.environ, {"NADIR_API_KEY": "ndr_env", "NADIR_BASE_URL": "http://env:9000"}):
        client = NadirClient(api_key="ndr_explicit", base_url="http://explicit:8000")
        assert client._api_key == "ndr_explicit"
        assert client._base_url == "http://explicit:8000"
        client.close()


def test_client_context_manager():
    with NadirClient(api_key="ndr_test") as client:
        assert client._api_key == "ndr_test"


def test_client_namespaces():
    client = NadirClient(api_key="ndr_test")
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client, "smart_export")
    assert hasattr(client, "clustering")
    assert hasattr(client, "organizations")
    client.close()


def test_client_headers():
    client = NadirClient(api_key="ndr_test123")
    headers = client._headers()
    assert headers["X-API-Key"] == "ndr_test123"
    assert headers["Content-Type"] == "application/json"
    assert "nadir-python" in headers["User-Agent"]
    client.close()


# ---------------------------------------------------------------------------
# Async client construction
# ---------------------------------------------------------------------------

def test_async_client_default_base_url():
    client = AsyncNadirClient(api_key="ndr_test")
    assert client._base_url == "https://api.getnadir.com"


def test_async_client_namespaces():
    client = AsyncNadirClient(api_key="ndr_test")
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client, "smart_export")
    assert hasattr(client, "clustering")
    assert hasattr(client, "organizations")


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------

def test_nadir_error_base():
    err = NadirError("something failed", status_code=500)
    assert str(err) == "something failed"
    assert err.status_code == 500


def test_auth_error():
    err = AuthenticationError("invalid key", status_code=401)
    assert err.status_code == 401
    assert isinstance(err, NadirError)


def test_rate_limit_error():
    err = RateLimitError("too fast", status_code=429, retry_after=30.0)
    assert err.retry_after == 30.0
    assert isinstance(err, NadirError)


def test_api_error():
    err = APIError("server error", status_code=500, body={"detail": "oops"})
    assert err.status_code == 500
    assert err.body == {"detail": "oops"}


def test_connection_error():
    err = ConnectionError("cannot connect", status_code=0)
    assert err.status_code == 0
    assert isinstance(err, NadirError)


def test_error_repr():
    err = APIError("bad request", status_code=400)
    r = repr(err)
    assert "APIError" in r
    assert "400" in r
