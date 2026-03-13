"""Concurrent worker querying for MoS strategies."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Callable, Any

from ovos_utils.log import LOG


def gather_concurrent(workers: List[Any],
                      fn: Callable,
                      query: str,
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      max_workers: Optional[int] = None,
                      timeout: float = 30.0) -> List[str]:
    """Query multiple workers concurrently using a thread pool.

    Workers are I/O-bound (API/network calls), so threads are appropriate.
    Individual worker exceptions are logged and skipped (graceful failure).

    Args:
        workers: List of worker objects to query.
        fn: Function to call on each worker, e.g. lambda w: w.get_response(query).
        query: The query string (for logging only).
        lang: Optional language code.
        units: Optional unit system.
        max_workers: Max concurrent threads. Defaults to len(workers).
        timeout: Per-worker timeout in seconds. Defaults to 30.
    Returns:
        List of non-empty response strings.
    """
    if not workers:
        return []

    max_workers = max_workers or len(workers)
    answers = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_worker = {
            executor.submit(fn, worker): worker
            for worker in workers
        }
        for future in as_completed(future_to_worker, timeout=timeout):
            worker = future_to_worker[future]
            try:
                result = future.result(timeout=0)
                if result:
                    answers.append(result)
            except Exception as e:
                LOG.error(f"Worker {worker} failed for query '{query}': {e}")

    if not answers:
        LOG.warning(f"No answers gathered from {len(workers)} workers.")
    return answers
