"""
Nadir Python SDK — OpenAI-compatible client for the Nadir LLM router.

Usage::

    from nadir import NadirClient

    client = NadirClient(api_key="ndr_...")

    # Drop-in replacement for OpenAI
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}],
        model="auto",
    )
    print(response.choices[0].message.content)

    # Streaming
    for chunk in client.chat.completions.create(
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True,
    ):
        print(chunk.choices[0].delta.content or "", end="", flush=True)

    # Access Nadir routing metadata
    print(response.nadir_metadata)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx

from .errors import APIError, AuthenticationError, ConnectionError, NadirError, RateLimitError
from .types import ChatCompletion, ChatCompletionChunk

DEFAULT_BASE_URL = "https://api.getnadir.com"


# ---------------------------------------------------------------------------
# Namespace classes (mirrors openai.chat.completions)
# ---------------------------------------------------------------------------

class _Completions:
    """Handles /v1/chat/completions."""

    def __init__(self, client: "NadirClient"):
        self._client = client

    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        route: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        transforms: Optional[List[str]] = None,
        layers: Optional[Dict[str, bool]] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Create a chat completion via the Nadir router.

        When ``stream=True``, returns an iterator of ``ChatCompletionChunk``.
        When ``stream=False`` (default), returns a ``ChatCompletion``.
        """
        body: Dict[str, Any] = {"messages": messages}
        if model is not None:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if response_format is not None:
            body["response_format"] = response_format
        if reasoning is not None:
            body["reasoning"] = reasoning
        if provider is not None:
            body["provider"] = provider
        if route is not None:
            body["route"] = route
        if fallback_models is not None:
            body["fallback_models"] = fallback_models
        if transforms is not None:
            body["transforms"] = transforms
        if layers is not None:
            body["layers"] = layers
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if stream:
            body["stream"] = True
        body.update(kwargs)

        if stream:
            return self._stream(body)

        data = self._client._post("/v1/chat/completions", body)
        return ChatCompletion.from_dict(data)

    def _stream(self, body: Dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        """Stream SSE chunks from the server."""
        with self._client._http.stream(
            "POST",
            f"{self._client._base_url}/v1/chat/completions",
            json=body,
            headers=self._client._headers(),
        ) as response:
            _raise_for_status(response)
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                yield ChatCompletionChunk.from_dict(json.loads(payload))


class _Chat:
    """Namespace: ``client.chat.completions``."""

    def __init__(self, client: "NadirClient"):
        self.completions = _Completions(client)


# ---------------------------------------------------------------------------
# Resource namespaces
# ---------------------------------------------------------------------------

class _SmartExport:
    """Smart Export / distillation endpoints."""

    def __init__(self, client: "NadirClient"):
        self._client = client

    def status(self) -> Dict[str, Any]:
        """Get Smart Export overview."""
        return self._client._get("/v1/smart-export/status")

    def train(self, cluster_id: str, *, base_model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Trigger fine-tuning for a cluster."""
        return self._client._post(
            f"/v1/smart-export/clusters/{cluster_id}/train",
            {"base_model": base_model},
        )

    def export_data(self, cluster_id: str, *, format: str = "jsonl") -> Dict[str, Any]:
        """Export training data for a cluster."""
        return self._client._post(
            f"/v1/smart-export/clusters/{cluster_id}/export",
            {"format": format},
        )

    def list_jobs(self) -> Dict[str, Any]:
        """List export/training jobs."""
        return self._client._get("/v1/smart-export/jobs")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get details of a specific job."""
        return self._client._get(f"/v1/smart-export/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job."""
        return self._client._delete(f"/v1/smart-export/jobs/{job_id}")


class _Clustering:
    """Prompt clustering endpoints."""

    def __init__(self, client: "NadirClient"):
        self._client = client

    def classify(self, prompt: str) -> Dict[str, Any]:
        """Classify a prompt into a cluster."""
        return self._client._post("/v1/clustering/classify", {"prompt": prompt})

    def list_clusters(self) -> Dict[str, Any]:
        """List all clusters."""
        return self._client._get("/v1/clustering/clusters")

    def create_cluster(
        self,
        name: str,
        description: str = "",
        *,
        examples: Optional[List[str]] = None,
        classification_criteria: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new cluster."""
        body: Dict[str, Any] = {"name": name, "description": description}
        if examples is not None:
            body["examples"] = examples
        if classification_criteria is not None:
            body["classification_criteria"] = classification_criteria
        return self._client._post("/v1/clustering/clusters", body)


class _Organizations:
    """Organization management endpoints."""

    def __init__(self, client: "NadirClient"):
        self._client = client

    def list(self) -> Dict[str, Any]:
        """List organizations."""
        return self._client._get("/v1/organizations")

    def create(self, name: str, slug: str, *, plan_type: Optional[str] = None) -> Dict[str, Any]:
        """Create an organization."""
        body: Dict[str, Any] = {"name": name, "slug": slug}
        if plan_type is not None:
            body["plan_type"] = plan_type
        return self._client._post("/v1/organizations", body)

    def get(self, org_id: str) -> Dict[str, Any]:
        """Get organization details."""
        return self._client._get(f"/v1/organizations/{org_id}")


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _raise_for_status(response: httpx.Response) -> None:
    """Raise a typed NadirError on non-2xx responses."""
    if response.is_success:
        return

    body: Any = None
    try:
        body = response.json()
    except Exception:
        pass

    detail = ""
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error", {}).get("message", "")
    if not detail:
        detail = response.text or f"HTTP {response.status_code}"

    if response.status_code == 401:
        raise AuthenticationError(detail, status_code=401, body=body)
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")
        raise RateLimitError(
            detail,
            status_code=429,
            body=body,
            retry_after=float(retry_after) if retry_after else None,
        )
    raise APIError(detail, status_code=response.status_code, body=body)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class NadirClient:
    """Synchronous client for the Nadir LLM router API.

    Args:
        api_key: Nadir API key (``ndr_...``). Falls back to ``NADIR_API_KEY`` env var.
        base_url: API base URL. Falls back to ``NADIR_BASE_URL`` env var.
        timeout: Request timeout in seconds (default 120).

    Usage::

        client = NadirClient(api_key="ndr_...")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self._api_key = api_key or os.getenv("NADIR_API_KEY", "")
        self._base_url = (base_url or os.getenv("NADIR_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self._timeout = timeout
        self._http = httpx.Client(timeout=timeout)

        # Namespaces
        self.chat = _Chat(self)
        self.smart_export = _SmartExport(self)
        self.clustering = _Clustering(self)
        self.organizations = _Organizations(self)

    # ── Convenience methods ───────────────────────────────────────────

    def recommend(
        self,
        prompt: str,
        *,
        system_message: Optional[str] = None,
        models: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        max_models: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get a model recommendation without making an LLM call."""
        body: Dict[str, Any] = {"prompt": prompt}
        if system_message is not None:
            body["system_message"] = system_message
        if models is not None:
            body["models"] = models
        if providers is not None:
            body["providers"] = providers
        if max_models is not None:
            body["max_models"] = max_models
        return self._post("/v1/public/recommendation", body)

    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        return self._get("/v1/models")

    def list_presets(self) -> Dict[str, Any]:
        """List user presets."""
        return self._get("/v1/user/presets")

    def health(self) -> Dict[str, Any]:
        """Check API health."""
        return self._get("/health")

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "NadirClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── HTTP internals ────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self._api_key,
            "User-Agent": "nadir-python/0.1.1",
        }

    def _get(self, path: str) -> Dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", path, body)

    def _delete(self, path: str) -> Dict[str, Any]:
        return self._request("DELETE", path)

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._http.request(
                method,
                url,
                json=body,
                headers=self._headers(),
            )
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Failed to connect to {self._base_url}: {exc}",
                status_code=0,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Request timed out after {self._timeout}s: {exc}",
                status_code=0,
            ) from exc

        _raise_for_status(response)
        return response.json()
