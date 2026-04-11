"""
Nadir async Python SDK — AsyncIO client for the Nadir LLM router.

Usage::

    import asyncio
    from nadir import AsyncNadirClient

    async def main():
        async with AsyncNadirClient(api_key="ndr_...") as client:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello!"}],
            )
            print(response.choices[0].message.content)

            # Streaming
            async for chunk in await client.chat.completions.create(
                messages=[{"role": "user", "content": "Tell me a story"}],
                stream=True,
            ):
                print(chunk.choices[0].delta.content or "", end="", flush=True)

    asyncio.run(main())
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from .errors import APIError, AuthenticationError, ConnectionError, NadirError, RateLimitError
from .types import ChatCompletion, ChatCompletionChunk

DEFAULT_BASE_URL = "https://api.getnadir.dev"


# ---------------------------------------------------------------------------
# Error helper (shared logic with sync client)
# ---------------------------------------------------------------------------

def _raise_for_status(response: httpx.Response) -> None:
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
# Async namespace classes
# ---------------------------------------------------------------------------

class _AsyncCompletions:
    def __init__(self, client: "AsyncNadirClient"):
        self._client = client

    async def create(
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
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
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

        data = await self._client._post("/v1/chat/completions", body)
        return ChatCompletion.from_dict(data)

    async def _stream(self, body: Dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        async with self._client._http.stream(
            "POST",
            f"{self._client._base_url}/v1/chat/completions",
            json=body,
            headers=self._client._headers(),
        ) as response:
            _raise_for_status(response)
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                yield ChatCompletionChunk.from_dict(json.loads(payload))


class _AsyncChat:
    def __init__(self, client: "AsyncNadirClient"):
        self.completions = _AsyncCompletions(client)


class _AsyncSmartExport:
    def __init__(self, client: "AsyncNadirClient"):
        self._client = client

    async def status(self) -> Dict[str, Any]:
        return await self._client._get("/v1/smart-export/status")

    async def train(self, cluster_id: str, *, base_model: str = "gpt-4o-mini") -> Dict[str, Any]:
        return await self._client._post(
            f"/v1/smart-export/clusters/{cluster_id}/train",
            {"base_model": base_model},
        )

    async def export_data(self, cluster_id: str, *, format: str = "jsonl") -> Dict[str, Any]:
        return await self._client._post(
            f"/v1/smart-export/clusters/{cluster_id}/export",
            {"format": format},
        )

    async def list_jobs(self) -> Dict[str, Any]:
        return await self._client._get("/v1/smart-export/jobs")

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        return await self._client._get(f"/v1/smart-export/jobs/{job_id}")

    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return await self._client._delete(f"/v1/smart-export/jobs/{job_id}")


class _AsyncClustering:
    def __init__(self, client: "AsyncNadirClient"):
        self._client = client

    async def classify(self, prompt: str) -> Dict[str, Any]:
        return await self._client._post("/v1/clustering/classify", {"prompt": prompt})

    async def list_clusters(self) -> Dict[str, Any]:
        return await self._client._get("/v1/clustering/clusters")

    async def create_cluster(
        self,
        name: str,
        description: str = "",
        *,
        examples: Optional[List[str]] = None,
        classification_criteria: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "description": description}
        if examples is not None:
            body["examples"] = examples
        if classification_criteria is not None:
            body["classification_criteria"] = classification_criteria
        return await self._client._post("/v1/clustering/clusters", body)


class _AsyncOrganizations:
    def __init__(self, client: "AsyncNadirClient"):
        self._client = client

    async def list(self) -> Dict[str, Any]:
        return await self._client._get("/v1/organizations")

    async def create(self, name: str, slug: str, *, plan_type: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "slug": slug}
        if plan_type is not None:
            body["plan_type"] = plan_type
        return await self._client._post("/v1/organizations", body)

    async def get(self, org_id: str) -> Dict[str, Any]:
        return await self._client._get(f"/v1/organizations/{org_id}")


# ---------------------------------------------------------------------------
# Main async client
# ---------------------------------------------------------------------------

class AsyncNadirClient:
    """Async client for the Nadir LLM router API.

    Args:
        api_key: Nadir API key (``ndr_...``). Falls back to ``NADIR_API_KEY`` env var.
        base_url: API base URL. Falls back to ``NADIR_BASE_URL`` env var.
        timeout: Request timeout in seconds (default 120).
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
        self._http = httpx.AsyncClient(timeout=timeout)

        self.chat = _AsyncChat(self)
        self.smart_export = _AsyncSmartExport(self)
        self.clustering = _AsyncClustering(self)
        self.organizations = _AsyncOrganizations(self)

    # ── Convenience methods ───────────────────────────────────────────

    async def recommend(
        self,
        prompt: str,
        *,
        system_message: Optional[str] = None,
        models: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        max_models: Optional[int] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"prompt": prompt}
        if system_message is not None:
            body["system_message"] = system_message
        if models is not None:
            body["models"] = models
        if providers is not None:
            body["providers"] = providers
        if max_models is not None:
            body["max_models"] = max_models
        return await self._post("/v1/public/recommendation", body)

    async def list_models(self) -> Dict[str, Any]:
        return await self._get("/v1/models")

    async def list_presets(self) -> Dict[str, Any]:
        return await self._get("/v1/user/presets")

    async def health(self) -> Dict[str, Any]:
        return await self._get("/health")

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncNadirClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ── HTTP internals ────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self._api_key,
            "User-Agent": "nadir-python/0.1.0",
        }

    async def _get(self, path: str) -> Dict[str, Any]:
        return await self._request("GET", path)

    async def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", path, body)

    async def _delete(self, path: str) -> Dict[str, Any]:
        return await self._request("DELETE", path)

    async def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = await self._http.request(
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
