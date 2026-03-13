
# ovos-MoS — FAQ

## Q: What is ovos-MoS?
A: Mixture of Solvers framework for OpenVoiceOS. It orchestrates multiple agent plugins to produce better answers using three strategies: King, Democracy, and Duopoly.

## Q: How do I install ovos-MoS?
A: `pip install ovos-MoS`

## Q: What are the three MoS strategies?
A: **King** — workers answer, one king selects/generates the final answer. **Democracy** — voters vote on worker answers, majority wins. **Duopoly** — founders discuss worker answers in multiple rounds, president decides.

## Q: What is the difference between ReRanker and Generative variants?
A: ReRanker variants use a `ReRankerEngine` to score and select the best answer from candidates. Generative variants use a `ChatEngine` (LLM) to synthesize a new answer from the gathered information.

## Q: What entry points does ovos-MoS register?
A: Ten `opm.agents.chat` entry points: `ovos-mos-king-reranker`, `ovos-mos-king-generative`, `ovos-mos-democracy`, `ovos-mos-duopoly-reranker`, `ovos-mos-duopoly-generative`, `ovos-mos-tournament`, `ovos-mos-cascade`, `ovos-mos-jury`, `ovos-mos-chain`, `ovos-mos-committee`.

## Q: How do I use MoS from config?
A: Set `"module": "ovos-mos-king-reranker"` (or any entry point) in your agent config, with `"king"`, `"workers"`, `"voters"`, `"founders"`, or `"president"` sub-configs specifying the sub-plugin modules and their configs.

## Q: Are worker queries concurrent?
A: Yes. The modern API (`ovos_MoS.agents`) uses `ThreadPoolExecutor` for concurrent worker querying. Configure `max_workers` and `worker_timeout` in config.

## Q: What bugs were fixed in 0.1.0?
A: Three bugs: (1) `AbstractDuopolyMoS.__init__` referenced `self.founders` before assignment — fixed to use parameter `founders`. (2) `ReRankerDuopolyMoS.discuss_answers` called `self.president.select_answer(query, lang=lang)` without the `answers` argument. (3) Both Duopoly `discuss_answers` methods called `self.gather_responses()` which overwrote the `answers` parameter, causing workers to be queried twice.

## Q: Is the legacy API still available?
A: Yes, but deprecated. Importing `ovos_MoS` emits a `DeprecationWarning`. The legacy classes will be removed in ovos-MoS 1.0. Use `ovos_MoS.agents` instead.

## Q: Can MoS be used recursively?
A: Yes. Any MoS engine is itself a `ChatEngine`, so it can serve as a worker, king, voter, founder, or president in another MoS.

## Q: What is the self-loading guard?
A: The factory module prevents loading any `ovos-mos-*` entry point as a sub-plugin, which would cause infinite recursion.

## Q: What is the Tournament strategy?
A: Bracket-style elimination. Workers provide answers, then pairs compete head-to-head via a ReRanker. Winners advance until one remains. Requires only log(N) reranker calls instead of N.

## Q: What is the Cascade strategy?
A: Sequential querying with early stopping. Workers are queried one by one; after each response, the scorer evaluates. If the top score exceeds a threshold, stop early. Saves cost when the first good answer is often sufficient.

## Q: What is the Jury strategy?
A: Weighted voting. Like Democracy but each juror (voter) has a numeric weight. The answer with the highest weighted vote total wins. Weights can represent model quality or historical accuracy.

## Q: What is the Chain strategy?
A: Sequential refinement pipeline. Worker 1 answers the query directly. Worker 2 receives the query + Worker 1's answer and improves it. Worker 3 improves Worker 2's output, and so on.

## Q: What is the Committee strategy?
A: Multi-round convergence. All workers answer independently in round 1. In subsequent rounds, each worker sees all other answers and revises. Repeats until convergence or max_rounds is reached.

## Q: How many tests does ovos-MoS have?
A: 125 tests across 9 test files: `test_legacy.py` (bug regression + smoke), `test_agents.py` (original 3 strategies), `test_new_strategies.py` (5 new strategies), `test_factory.py` (config-driven loading), `test_concurrent.py` (ThreadPoolExecutor), `test_metrics.py` (observability), `test_compose.py` (composition helpers), `test_streaming.py` (streaming support), `test_async.py` (asyncio).

## Q: Does ovos-MoS support streaming?
A: Yes. Engines with generative kings/presidents (King, Duopoly, Chain) delegate `stream_tokens()` and `stream_sentences()` to the inner ChatEngine. ReRanker-based strategies fall back to the base class default.

## Q: How do I track performance of MoS strategies?
A: Use `MoSMetrics` from `ovos_MoS.metrics`. It provides thread-safe context managers `track_strategy()` and `track_worker()` for timing. Call `report()` for structured data or `report_text()` for human-readable output.

## Q: Can I compose MoS strategies recursively?
A: Yes. Every MoS engine is a `ChatEngine`. The `ovos_MoS.compose` module provides helpers: `democracy_of_kings()`, `cascade_then_committee()`, `chain_with_jury()`, `tournament_of_committees()`, `duopoly_with_cascade_workers()`.

## Q: Does ovos-MoS support asyncio?
A: Yes. `ovos_MoS._async` provides `gather_async()` for natively async workers and `gather_sync_in_executor()` for wrapping sync workers in asyncio's executor.

## Q: What CI workflows does ovos-MoS have?
A: Six workflows: `release_workflow.yml` (alpha release), `publish_stable.yml` (stable release), `build_tests.yml` (build validation), `unit_tests.yml` (pytest on Python 3.9–3.12), `license_tests.yml` (license check), `conventional-label.yaml` (PR labels).

## Q: Can I search for the best MoS configuration automatically?
A: Yes. `ovos_MoS.search` provides `GridSearch`, `RandomSearch`, and `GeneticSearch` over the full design space — strategy type, worker/king/reranker plugins, and numeric parameters. Define a search space with dimension types (`CategoricalDim`, `IntRangeDim`, `FloatRangeDim`, `FloatListDim`, `IntListDim`), provide an `engine_factory` and `evaluate` function, then run search.

## Q: What is the default_mos_factory?
A: A helper that builds a MoS engine from a params dict. It maps the `"strategy"` key to the right engine class (King, Democracy, etc.) and loads sub-plugins via OPM's `_load_chat_worker`/`_load_reranker_worker`.

## Q: How does discover_search_space work?
A: It uses `find_chat_plugins()` and `find_reranker_plugins()` from OPM to discover installed plugins, filters out `ovos-mos-*` self-references, and generates a search space with worker combinations (1–3 workers), king/reranker options, and numeric dims with sensible defaults.

## Q: What search strategies are available?
A: Three: **GridSearch** — exhaustive `itertools.product` with `float_steps` and `max_evaluations` cap. **RandomSearch** — sample `n_trials` random configs, reproducible via `seed`. **GeneticSearch** — tournament selection, uniform crossover, per-dimension mutation, elite preservation.

## Q: Does the search module require extra dependencies?
A: No. It uses only Python stdlib (`random`, `itertools`, `dataclasses`). No new dependencies.

## Q: Can a MoS judge other MoS configurations?
A: Yes. Use `make_mos_judge_evaluator()` — a high-quality MoS engine produces reference answers, then a ReRankerEngine scores candidate engines against them. This enables evolving MoS configurations where one MoS evaluates another.

## Q: How do I use a ReRanker as a search evaluator?
A: Use `make_reranker_evaluator(judge, benchmark)`. It scores each engine's answers against reference answers from a benchmark using the ReRankerEngine's `rerank()` method. Returns average score.

## Q: Can I optimize for latency instead of quality?
A: Yes. Use `make_latency_evaluator(queries)`. It returns negative average latency (since search maximizes by default). Combine with quality evaluators for multi-objective optimization.
