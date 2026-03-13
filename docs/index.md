
# ovos-MoS Documentation

Mixture of Solvers (MoS) orchestrates multiple OVOS agent plugins to produce better answers through eight strategies.

## Contents

- [Architecture](architecture.md) — design overview, class hierarchy, data flow
- [Strategies](strategies.md) — King, Democracy, Duopoly, Tournament, Cascade, Jury, Chain, Committee
- [Configuration](configuration.md) — config-driven OPM plugin setup
- [Concurrency](concurrency.md) — ThreadPoolExecutor-based parallel worker querying
- [Async Support](async.md) — asyncio-based alternatives for async frameworks
- [Streaming](streaming.md) — token/sentence streaming for generative strategies
- [Metrics](metrics.md) — timing, success rate tracking, observability
- [Composition Helpers](compose.md) — convenience functions for recursive MoS compositions
- [Search](search.md) — hyperparameter & architecture search (grid, random, genetic)
- [Migration Guide](migration.md) — migrating from legacy QuestionSolver API to ChatEngine API

## Quick Start

```python
from ovos_MoS.agents import KingMoSEngine

engine = KingMoSEngine(king=my_reranker, workers=[chat1, chat2])
answer = engine.get_response("What is the speed of light?")
```

## Module Map

| Module | Purpose |
|--------|---------|
| `ovos_MoS.agents` | Modern ChatEngine-based MoS classes |
| `ovos_MoS.factory` | Config-driven OPM plugin wrappers |
| `ovos_MoS._concurrent` | ThreadPoolExecutor gather utility |
| `ovos_MoS._async` | Asyncio-based gather utilities |
| `ovos_MoS._streaming` | Streaming mixins for generative strategies |
| `ovos_MoS.metrics` | Timing and observability |
| `ovos_MoS.compose` | Strategy composition helpers |
| `ovos_MoS.search` | Hyperparameter & architecture search |
| `ovos_MoS.__init__` | Legacy deprecated classes |
