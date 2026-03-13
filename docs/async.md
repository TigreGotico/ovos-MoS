
# Async Support

## Overview

`ovos_MoS._async` provides asyncio-based alternatives to the ThreadPoolExecutor-based `gather_concurrent()`. Two functions are available depending on whether your workers are natively async or synchronous.

## Native Async Workers

`gather_async` — `ovos_MoS/_async.py:18`

For workers with native async support (e.g. aiohttp-based API calls), this avoids thread pool overhead entirely.

```python
import asyncio
from ovos_MoS._async import gather_async

async def main():
    async def query_worker(w):
        return await w.async_get_response("What is AI?")

    answers = await gather_async(
        workers=[async_worker1, async_worker2],
        fn=query_worker,
        query="What is AI?",
        timeout=30.0,
    )
```

## Sync Workers in Async Context

`gather_sync_in_executor` — `ovos_MoS/_async.py:69`

For standard synchronous ChatEngine plugins running in an async application, this wraps calls in `asyncio`'s default thread executor.

```python
import asyncio
from ovos_MoS._async import gather_sync_in_executor

async def main():
    answers = await gather_sync_in_executor(
        workers=[chat1, chat2],
        fn=lambda w: w.get_response("What is AI?"),
        query="What is AI?",
        timeout=30.0,
    )
```

## Error Handling

Both functions handle individual worker failures gracefully — failed workers are logged and skipped. Timed-out tasks are cancelled. Empty/None responses are filtered out.

## When to Use

| Scenario | Function |
|----------|----------|
| Workers have native `async` methods | `gather_async` |
| Standard sync ChatEngine plugins in an async app | `gather_sync_in_executor` |
| Standard sync app (no asyncio) | `gather_concurrent` from `ovos_MoS._concurrent` |
