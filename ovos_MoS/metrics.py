"""Metrics and observability for MoS strategies.

Provides timing, success rate tracking, and per-worker statistics
to help tune max_workers, worker_timeout, threshold, and juror_weights.
"""
import time
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from ovos_utils.log import LOG


@dataclass
class WorkerStats:
    """Statistics for a single worker."""
    name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.successes if self.successes > 0 else 0.0

    def record_success(self, elapsed: float) -> None:
        self.total_calls += 1
        self.successes += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

    def record_failure(self) -> None:
        self.total_calls += 1
        self.failures += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 3),
            "avg_time": round(self.avg_time, 3),
            "min_time": round(self.min_time, 3) if self.min_time != float("inf") else None,
            "max_time": round(self.max_time, 3),
        }


@dataclass
class StrategyStats:
    """Statistics for an entire MoS strategy invocation."""
    strategy_name: str
    total_calls: int = 0
    total_time: float = 0.0
    worker_stats: Dict[str, WorkerStats] = field(default_factory=dict)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total_calls if self.total_calls > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "total_calls": self.total_calls,
            "avg_time": round(self.avg_time, 3),
            "workers": {k: v.to_dict() for k, v in self.worker_stats.items()},
        }


class MoSMetrics:
    """Thread-safe metrics collector for MoS strategies.

    Usage:
        metrics = MoSMetrics()
        with metrics.track_strategy("KingMoS"):
            with metrics.track_worker("ddg-solver"):
                answer = worker.get_response(query)
        print(metrics.report())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._strategies: Dict[str, StrategyStats] = {}

    def _get_strategy(self, name: str) -> StrategyStats:
        if name not in self._strategies:
            self._strategies[name] = StrategyStats(strategy_name=name)
        return self._strategies[name]

    def _get_worker(self, strategy_name: str, worker_name: str) -> WorkerStats:
        strategy = self._get_strategy(strategy_name)
        if worker_name not in strategy.worker_stats:
            strategy.worker_stats[worker_name] = WorkerStats(name=worker_name)
        return strategy.worker_stats[worker_name]

    def track_strategy(self, strategy_name: str) -> "_StrategyTimer":
        """Context manager to time a full strategy invocation."""
        return _StrategyTimer(self, strategy_name)

    def track_worker(self, strategy_name: str,
                     worker_name: str) -> "_WorkerTimer":
        """Context manager to time a single worker call."""
        return _WorkerTimer(self, strategy_name, worker_name)

    def record_worker_success(self, strategy_name: str,
                              worker_name: str, elapsed: float) -> None:
        with self._lock:
            ws = self._get_worker(strategy_name, worker_name)
            ws.record_success(elapsed)

    def record_worker_failure(self, strategy_name: str,
                              worker_name: str) -> None:
        with self._lock:
            ws = self._get_worker(strategy_name, worker_name)
            ws.record_failure()

    def record_strategy_call(self, strategy_name: str,
                             elapsed: float) -> None:
        with self._lock:
            s = self._get_strategy(strategy_name)
            s.total_calls += 1
            s.total_time += elapsed

    def report(self) -> Dict[str, Any]:
        """Return a full metrics report as a dict."""
        with self._lock:
            return {
                name: stats.to_dict()
                for name, stats in self._strategies.items()
            }

    def report_text(self) -> str:
        """Return a human-readable metrics report."""
        lines = ["MoS Metrics Report", "=" * 40]
        for name, stats in self._strategies.items():
            lines.append(f"\n{stats.strategy_name}:")
            lines.append(f"  Calls: {stats.total_calls}  "
                         f"Avg time: {stats.avg_time:.3f}s")
            for wn, ws in stats.worker_stats.items():
                lines.append(
                    f"  Worker '{wn}': {ws.successes}/{ws.total_calls} "
                    f"({ws.success_rate:.0%}) avg={ws.avg_time:.3f}s"
                )
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._strategies.clear()


class _StrategyTimer:
    """Context manager for timing strategy calls."""

    def __init__(self, metrics: MoSMetrics, strategy_name: str) -> None:
        self._metrics = metrics
        self._name = strategy_name
        self._start = 0.0

    def __enter__(self) -> "_StrategyTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.monotonic() - self._start
        self._metrics.record_strategy_call(self._name, elapsed)
        return None


class _WorkerTimer:
    """Context manager for timing individual worker calls."""

    def __init__(self, metrics: MoSMetrics,
                 strategy_name: str, worker_name: str) -> None:
        self._metrics = metrics
        self._strategy = strategy_name
        self._worker = worker_name
        self._start = 0.0

    def __enter__(self) -> "_WorkerTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.monotonic() - self._start
        if exc_type is None:
            self._metrics.record_worker_success(
                self._strategy, self._worker, elapsed
            )
        else:
            self._metrics.record_worker_failure(
                self._strategy, self._worker
            )
        return None
