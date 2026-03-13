
# ovos-MoS — Maintenance Report

## 2026-03-09 — Hyperparameter & Architecture Search

### Changes
- **New module**: `ovos_MoS/search.py` (~550 lines) — grid, random, and genetic search over the full MoS design space
- **Search space dimensions**: `CategoricalDim`, `IntRangeDim`, `FloatRangeDim`, `FloatListDim`, `IntListDim`
- **Default factory**: `default_mos_factory()` maps strategy name → engine class, loads sub-plugins via OPM
- **Plugin discovery**: `discover_search_space()` auto-generates search space from installed OPM plugins
- **Three search strategies**: `GridSearch` (exhaustive), `RandomSearch` (sampling), `GeneticSearch` (evolutionary)
- **MoS-as-judge evaluators**: `make_reranker_evaluator()`, `make_mos_judge_evaluator()`, `make_latency_evaluator()` — use ReRankers or MoS engines to score candidate configurations
- **Tests**: 43 new tests in `test/test_search.py` (168 total), all passing, zero network calls
- **Docs**: `docs/search.md` with full API reference and examples

### Transparency Report
- **AI Model**: Claude Opus 4.6
- **Actions Taken**: Implemented hyperparameter/architecture search module with three search strategies, default engine factory, plugin discovery, tests, and documentation
- **Human Oversight**: User-directed implementation from detailed plan

## 2026-03-09 — CI, Streaming, Metrics, Composition, Async

### Changes
- **CI workflows**: Added `license_tests.yml` (reusable license check) and `unit_tests.yml` (pytest on Python 3.9–3.12)
- **Streaming**: Streaming mixins (`_streaming.py`) for King, Duopoly, Chain, Committee — delegate `stream_tokens()`/`stream_sentences()` to inner generative engines
- **Metrics**: `metrics.py` with `MoSMetrics`, `WorkerStats`, `StrategyStats` — thread-safe timing and success rate tracking
- **Composition helpers**: `compose.py` with 5 functions: `democracy_of_kings()`, `cascade_then_committee()`, `chain_with_jury()`, `tournament_of_committees()`, `duopoly_with_cascade_workers()`
- **Async support**: `_async.py` with `gather_async()` (native async) and `gather_sync_in_executor()` (sync-in-async)
- **Tests**: 37 new tests (125 total), all passing
- **Docs**: 4 new doc pages (async.md, streaming.md, metrics.md, compose.md)

### Transparency Report
- **AI Model**: Claude Opus 4.6
- **Actions Taken**: Implemented 5 features — CI workflows, streaming mixins, metrics/observability, composition helpers, async support — with tests and documentation
- **Human Oversight**: User-directed implementation

## 2026-03-09 — Five New Strategies

### Changes
- **Tournament**: Bracket-style elimination via ReRanker — O(log N) comparisons
- **Cascade**: Sequential querying with early stopping on confidence threshold
- **Jury**: Weighted voting — each juror has a configurable weight
- **Chain**: Sequential refinement pipeline — each worker improves the previous answer
- **Committee**: Multi-round convergence — workers revise after seeing each other's answers
- **Factory plugins**: 5 new `opm.agents.chat` entry points
- **Tests**: 39 new tests (88 total), all passing

### Transparency Report
- **AI Model**: Claude Opus 4.6
- **Actions Taken**: Implemented 5 new MoS strategies with engine classes, factory plugins, entry points, and tests
- **Human Oversight**: User-directed implementation

## 2026-03-09 — Modern API Migration (v0.1.0a1)

### Changes
- **Bug fixes** (Phase 1): Fixed 3 bugs in legacy `__init__.py`:
  - `AbstractDuopolyMoS.__init__`: `self.founders` referenced before assignment
  - `ReRankerDuopolyMoS.discuss_answers`: `select_answer` missing `answers` arg
  - Both Duopoly `discuss_answers`: redundant `gather_responses` call overwrote `answers` param
- **Deprecation warning**: Legacy module emits `DeprecationWarning` on import
- **Concurrent querying** (Phase 2): `_concurrent.py` with `ThreadPoolExecutor`-based `gather_concurrent()`
- **Modern agent API** (Phase 3): `agents.py` with `KingMoSEngine`, `DemocracyMoSEngine`, `DuopolyMoSEngine` built on `ChatEngine`/`ReRankerEngine`
- **Config-driven factory** (Phase 4): `factory.py` with 5 OPM plugin wrappers + self-loading guard
- **Entry points**: 5 `opm.agents.chat` entry points in `pyproject.toml`
- **Tests** (Phase 5): 49 tests across 4 files, all passing, zero network calls
- **Version**: Bumped to 0.1.0a1

### Transparency Report
- **AI Model**: Claude Opus 4.6
- **Actions Taken**: Implemented 6-phase improvement plan — bug fixes, concurrency module, modern ChatEngine-based API, config-driven factory with OPM entry points, comprehensive test suite, documentation updates
- **Human Oversight**: User-directed implementation from detailed plan

## 2026-03-09 — Agent Plugins Audit

### Transparency Report
- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**: Audited plugin, fixed CI workflows, added missing docs and LICENSE
- **Oversight**: User-directed audit of all agent plugins
