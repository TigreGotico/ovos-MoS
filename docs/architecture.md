
# Architecture

## Class Hierarchy

```
ChatEngine (from ovos_plugin_manager.templates.agents)
  ├── AbstractMoSEngine
  │     │   gather_responses() → concurrent worker querying
  │     │   continue_chat() → abstract
  │     │
  │     ├── KingMoSEngine
  │     │     One king (ReRankerEngine or ChatEngine) + N workers
  │     │
  │     ├── DemocracyMoSEngine
  │     │     N voters (ReRankerEngine) + optional president + N workers
  │     │
  │     ├── DuopolyMoSEngine
  │     │     President + M founders + N workers
  │     │
  │     ├── TournamentMoSEngine
  │     │     Referee (ReRankerEngine) + N workers → bracket elimination
  │     │
  │     ├── JuryMoSEngine
  │     │     N weighted jurors + optional president + N workers
  │     │
  │     └── CommitteeMoSEngine
  │           N workers + optional president → multi-round convergence
  │
  ├── CascadeMoSEngine (extends ChatEngine directly)
  │     Scorer (ReRankerEngine) + ordered workers → early stopping
  │
  └── ChainMoSEngine (extends ChatEngine directly)
        Ordered workers + optional president → sequential refinement
```

Note: `CascadeMoSEngine` and `ChainMoSEngine` extend `ChatEngine` directly (not `AbstractMoSEngine`) because they use sequential worker querying rather than concurrent gathering.

Streaming mixins (`StreamingKingMixin`, `StreamingDuopolyMixin`, `StreamingChainMixin`, `StreamingCommitteeMixin`) from `ovos_MoS/_streaming.py` are applied via multiple inheritance to engines with generative kings/presidents.

All classes inherit from `ChatEngine`, so any MoS engine can be used as a worker, king, voter, founder, or president inside another MoS (recursive composition).

## Key Classes

### AbstractMoSEngine

`AbstractMoSEngine.__init__` — `ovos_MoS/agents.py:24`

Base class that holds the worker list and provides concurrent `gather_responses()`.

`AbstractMoSEngine.gather_responses` — `ovos_MoS/agents.py:31`

Queries all workers in parallel via `gather_concurrent()` from `ovos_MoS/_concurrent.py:8`. Returns a `List[str]` of non-empty responses.

### KingMoSEngine

`KingMoSEngine.continue_chat` — `ovos_MoS/agents.py:75`

1. Extracts query from the last user message
2. Gathers worker responses concurrently
3. If king is `ReRankerEngine`: calls `king.rerank()` and returns the top-scored answer
4. If king is `ChatEngine`: formats a prompt with worker answers and delegates to `king.continue_chat()`

### DemocracyMoSEngine

`DemocracyMoSEngine.gather_votes` — `ovos_MoS/agents.py:124`

Each voter calls `select_answer()` on the gathered worker answers. Results are tallied in a `Dict[str, int]`.

`DemocracyMoSEngine.continue_chat` — `ovos_MoS/agents.py:136`

1. Gathers worker responses
2. Voters vote → majority answer wins
3. If no president: returns the majority answer
4. If president is `ReRankerEngine`: reranks the voted answers
5. If president is `ChatEngine`: generates a synthesis from voted answers

### DuopolyMoSEngine

`DuopolyMoSEngine.continue_chat` — `ovos_MoS/agents.py:205`

1. Gathers worker responses
2. Founders discuss for `discussion_rounds` rounds — each founder gets a prompt with the query, answers, and prior discussion
3. If president is `ReRankerEngine`: founders generate final candidates, president reranks them
4. If president is `ChatEngine`: president generates a final answer from the discussion

## Factory Layer

`ovos_MoS/factory.py` provides 10 thin `ChatEngine` subclasses that read sub-plugin configs, instantiate them via OPM's `load_chat_plugin`/`load_reranker_plugin`, and delegate to the inner MoS engine.

`_MOS_ENTRY_POINTS` — `ovos_MoS/factory.py:17`

Self-loading guard: prevents loading any `ovos-mos-*` entry point as a sub-plugin to avoid infinite recursion.

## Concurrency

`gather_concurrent` — `ovos_MoS/_concurrent.py:8`

Uses `ThreadPoolExecutor` with configurable `max_workers` and `timeout`. Individual worker failures are logged and skipped. Workers are I/O-bound (API/network calls), making threads appropriate.

## Async Support

`gather_async` — `ovos_MoS/_async.py:18`

Asyncio-based alternative for natively async workers. Uses `asyncio.create_task` and `asyncio.wait` with timeout.

`gather_sync_in_executor` — `ovos_MoS/_async.py:69`

Wraps synchronous worker calls in `asyncio`'s default thread executor for integration with async applications.

## Streaming

Streaming mixins in `ovos_MoS/_streaming.py` override `stream_tokens()` and `stream_sentences()` on engines with generative kings/presidents. For ReRanker-based strategies (selection), the base class default is used.

## Metrics

`MoSMetrics` — `ovos_MoS/metrics.py:70`

Thread-safe metrics collector with context managers for timing strategies and workers. Tracks total calls, success rates, and min/max/avg response times.

## Composition

`ovos_MoS/compose.py` provides 5 helper functions for common recursive MoS compositions: `democracy_of_kings`, `cascade_then_committee`, `chain_with_jury`, `tournament_of_committees`, `duopoly_with_cascade_workers`.

## Search

`ovos_MoS/search.py` provides hyperparameter and architecture search over the full MoS design space.

`_BaseSearch` — `ovos_MoS/search.py:350`

Base class with `_sample_random()`, `_evaluate_params()`, `_sort_results()`.

Three search strategies:
- `GridSearch` — exhaustive `itertools.product` with configurable `float_steps` and `max_evaluations`
- `RandomSearch` — sample `n_trials` random configs with reproducible `seed`
- `GeneticSearch` — tournament selection, uniform crossover, per-dimension mutation, elite preservation

`default_mos_factory` — `ovos_MoS/search.py:140`

Default engine factory that maps `"strategy"` key to engine class and loads sub-plugins via `_load_chat_worker`/`_load_reranker_worker` from `factory.py`.

`discover_search_space` — `ovos_MoS/search.py:268`

Auto-generates a search space from installed OPM plugins, filtering out `ovos-mos-*` self-references.
