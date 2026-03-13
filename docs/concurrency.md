
# Concurrency

## Overview

Worker querying is I/O-bound (API calls, network requests), so `ovos-MoS` uses Python's `ThreadPoolExecutor` for concurrent execution.

## Implementation

`gather_concurrent` — `ovos_MoS/_concurrent.py:8`

```python
def gather_concurrent(workers, fn, query, lang, units, max_workers, timeout):
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `workers` | — | List of worker objects |
| `fn` | — | Callable that takes a worker and returns a response string |
| `query` | — | Query string (for logging) |
| `max_workers` | `len(workers)` | Max concurrent threads |
| `timeout` | `30.0` | Timeout for `as_completed()` in seconds |

### Behavior

1. Each worker is submitted to the thread pool via `executor.submit(fn, worker)` at `ovos_MoS/_concurrent.py:39`
2. Results are collected as they complete via `as_completed()` at `ovos_MoS/_concurrent.py:42`
3. Individual failures are caught and logged at `ovos_MoS/_concurrent.py:49` — they do not propagate
4. `None` and empty string responses are filtered at `ovos_MoS/_concurrent.py:46`
5. If all workers fail, an empty list is returned with a warning at `ovos_MoS/_concurrent.py:52`

### Usage in MoS Engines

`AbstractMoSEngine.gather_responses` — `ovos_MoS/agents.py:31`

Calls `gather_concurrent()` with `fn=lambda w: w.get_response(query, lang=lang, units=units)`, passing `max_workers` and `worker_timeout` from config.

### Config

Set in the MoS engine config:

```json
{
    "max_workers": 4,
    "worker_timeout": 30
}
```

- `max_workers` defaults to the number of workers (one thread per worker)
- `worker_timeout` defaults to 30 seconds
