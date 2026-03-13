"""Asyncio-based concurrent worker querying for MoS strategies.

Provides an async alternative to the ThreadPoolExecutor-based
gather_concurrent() for use in async frameworks.
"""
import asyncio
from typing import Optional, List, Callable, Any, Awaitable

from ovos_utils.log import LOG


async def gather_async(
    workers: List[Any],
    fn: Callable[[Any], Awaitable[str]],
    query: str,
    lang: Optional[str] = None,
    units: Optional[str] = None,
    timeout: float = 30.0,
) -> List[str]:
    """Query multiple workers concurrently using asyncio.

    For workers with native async support (e.g. aiohttp-based API calls),
    this avoids the thread pool overhead entirely.

    Args:
        workers: List of worker objects to query.
        fn: Async function to call on each worker.
        query: The query string (for logging only).
        lang: Optional language code.
        units: Optional unit system.
        timeout: Total timeout in seconds.

    Returns:
        List of non-empty response strings.
    """
    if not workers:
        return []

    async def _call_worker(worker: Any) -> Optional[str]:
        try:
            result = await fn(worker)
            return result if result else None
        except Exception as e:
            LOG.error(f"Async worker {worker} failed for query '{query}': {e}")
            return None

    tasks = [asyncio.create_task(_call_worker(w)) for w in workers]
    done, pending = await asyncio.wait(tasks, timeout=timeout)

    for task in pending:
        task.cancel()
        LOG.warning(f"Async worker timed out for query '{query}'")

    answers = []
    for task in done:
        try:
            result = task.result()
            if result:
                answers.append(result)
        except Exception as e:
            LOG.error(f"Async worker result failed: {e}")

    if not answers:
        LOG.warning(f"No answers gathered from {len(workers)} async workers.")
    return answers


async def gather_sync_in_executor(
    workers: List[Any],
    fn: Callable[[Any], str],
    query: str,
    lang: Optional[str] = None,
    units: Optional[str] = None,
    timeout: float = 30.0,
) -> List[str]:
    """Run synchronous worker calls in asyncio's default executor.

    Use this when workers are synchronous (e.g. standard ChatEngine plugins)
    but you want to integrate with an async application.

    Args:
        workers: List of worker objects to query.
        fn: Synchronous function to call on each worker.
        query: The query string (for logging only).
        lang: Optional language code.
        units: Optional unit system.
        timeout: Total timeout in seconds.

    Returns:
        List of non-empty response strings.
    """
    if not workers:
        return []

    loop = asyncio.get_running_loop()

    async def _call_worker(worker: Any) -> Optional[str]:
        try:
            result = await loop.run_in_executor(None, fn, worker)
            return result if result else None
        except Exception as e:
            LOG.error(f"Worker {worker} failed for query '{query}': {e}")
            return None

    tasks = [asyncio.create_task(_call_worker(w)) for w in workers]
    done, pending = await asyncio.wait(tasks, timeout=timeout)

    for task in pending:
        task.cancel()

    answers = []
    for task in done:
        try:
            result = task.result()
            if result:
                answers.append(result)
        except Exception as e:
            LOG.error(f"Worker result failed: {e}")

    if not answers:
        LOG.warning(f"No answers gathered from {len(workers)} workers.")
    return answers
