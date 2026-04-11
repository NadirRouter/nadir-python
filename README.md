# Nadir Python SDK

Official Python client for the [Nadir](https://getnadir.dev) LLM router — intelligent model routing that cuts API costs 30-60%.

## Installation

```bash
pip install nadir-sdk
```

## Quick Start

```python
from nadir import NadirClient

client = NadirClient(api_key="ndr_...")

# Chat completion — Nadir picks the optimal model
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(response.choices[0].message.content)

# See which model was selected and why
print(response.nadir_metadata.tier)             # "simple"
print(response.nadir_metadata.selected_model)   # "gpt-4o-mini"
print(response.nadir_metadata.complexity_score)  # 0.12
```

## Streaming

```python
stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Model Recommendation (no LLM call)

```python
rec = client.recommend("Explain quantum entanglement in detail")
print(rec)  # {"recommended_model": "claude-sonnet-4-20250514", "complexity": ...}
```

## Async

```python
import asyncio
from nadir import AsyncNadirClient

async def main():
    async with AsyncNadirClient(api_key="ndr_...") as client:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

## Advanced: Fallback & Routing Control

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Complex analysis..."}],
    route="fallback",                          # enable auto-fallback
    fallback_models=["claude-sonnet-4-20250514", "gpt-4o"],  # explicit fallback chain
    layers={"routing": True, "optimize": True},  # per-request feature toggles
    reasoning={"effort": "high"},              # reasoning token support
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NADIR_API_KEY` | API key (fallback if not passed to constructor) | — |
| `NADIR_BASE_URL` | API base URL | `https://api.getnadir.dev` |

## OpenAI Drop-in Compatibility

Nadir's API is OpenAI-compatible. You can also use the OpenAI SDK directly:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.getnadir.dev/v1",
    api_key="ndr_...",
)
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The Nadir SDK adds typed access to routing metadata, recommendations, smart export, and clustering that the OpenAI SDK can't reach.
