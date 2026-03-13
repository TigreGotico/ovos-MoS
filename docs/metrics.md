
# Metrics and Observability

## Overview

`ovos_MoS.metrics` provides thread-safe timing and success rate tracking for MoS strategies. Use it to tune `max_workers`, `worker_timeout`, `threshold`, and `juror_weights`.

## Classes

`WorkerStats` — `ovos_MoS/metrics.py:14`

Per-worker statistics: total calls, successes, failures, min/max/avg response time, success rate.

`StrategyStats` — `ovos_MoS/metrics.py:52`

Per-strategy statistics: total calls, average time, and a dict of `WorkerStats` for each worker.

`MoSMetrics` — `ovos_MoS/metrics.py:70`

Thread-safe metrics collector. Provides context managers for timing strategies and workers.

## Usage

```python
from ovos_MoS.metrics import MoSMetrics

metrics = MoSMetrics()

# Time a full strategy invocation
with metrics.track_strategy("KingMoS"):
    # Time individual workers
    with metrics.track_worker("KingMoS", "ddg-solver"):
        answer1 = ddg.get_response(query)
    with metrics.track_worker("KingMoS", "wikipedia"):
        answer2 = wiki.get_response(query)

# Get structured report
report = metrics.report()
# {"KingMoS": {"total_calls": 1, "avg_time": 1.234, "workers": {...}}}

# Get human-readable text
print(metrics.report_text())
# MoS Metrics Report
# ========================================
# KingMoS:
#   Calls: 1  Avg time: 1.234s
#   Worker 'ddg-solver': 1/1 (100%) avg=0.500s
#   Worker 'wikipedia': 1/1 (100%) avg=0.734s

# Reset all metrics
metrics.reset()
```

## Context Managers

`MoSMetrics.track_strategy(name)` — `ovos_MoS/metrics.py:90`

Times the full strategy execution. Records elapsed time on exit.

`MoSMetrics.track_worker(strategy_name, worker_name)` — `ovos_MoS/metrics.py:95`

Times a single worker call. Records success on normal exit, failure on exception. The exception is not suppressed.

## Thread Safety

All methods use a `threading.Lock` for thread-safe access. Safe to use with `ThreadPoolExecutor`-based concurrent worker querying.
