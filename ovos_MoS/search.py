"""Hyperparameter and architecture search for MoS configurations.

Provides grid, random, and genetic search over the full MoS design space:
strategy type, worker/king/reranker plugins, and numeric parameters.

No external dependencies — stdlib only (random, itertools, dataclasses).
"""
import itertools
import math
import random as _random
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

from ovos_utils.log import LOG

from ovos_MoS.agents import (
    KingMoSEngine, DemocracyMoSEngine, DuopolyMoSEngine,
    TournamentMoSEngine, CascadeMoSEngine, JuryMoSEngine,
    ChainMoSEngine, CommitteeMoSEngine,
)
from ovos_MoS.factory import (
    _load_chat_worker, _load_reranker_worker, _load_workers,
    _MOS_ENTRY_POINTS,
)

# ---------------------------------------------------------------------------
# Search-space dimension types
# ---------------------------------------------------------------------------

SearchSpace = Dict[str, "Dimension"]


@dataclass(frozen=True)
class CategoricalDim:
    """Finite set of choices (strategy names, plugin configs, etc.)."""
    choices: Sequence[Any]

    def sample(self, rng: _random.Random) -> Any:
        return rng.choice(self.choices)


@dataclass(frozen=True)
class IntRangeDim:
    """Integer range [low, high] inclusive."""
    low: int
    high: int

    def sample(self, rng: _random.Random) -> int:
        return rng.randint(self.low, self.high)


@dataclass(frozen=True)
class FloatRangeDim:
    """Continuous range [low, high]."""
    low: float
    high: float

    def sample(self, rng: _random.Random) -> float:
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class FloatListDim:
    """Fixed-length list of floats, each in [low, high]."""
    length: int
    low: float
    high: float

    def sample(self, rng: _random.Random) -> List[float]:
        return [rng.uniform(self.low, self.high) for _ in range(self.length)]


@dataclass(frozen=True)
class IntListDim:
    """Fixed-length list of ints, each in [low, high]."""
    length: int
    low: int
    high: int

    def sample(self, rng: _random.Random) -> List[int]:
        return [rng.randint(self.low, self.high) for _ in range(self.length)]


Dimension = Union[CategoricalDim, IntRangeDim, FloatRangeDim,
                   FloatListDim, IntListDim]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single evaluated configuration."""
    params: Dict[str, Any]
    score: float


@dataclass
class SearchReport:
    """Sorted search results with metadata."""
    results: List[SearchResult]
    best: SearchResult
    search_type: str
    total_evaluations: int

    def top_k(self, k: int = 5) -> List[SearchResult]:
        """Return the *k* best results."""
        return self.results[:k]


# ---------------------------------------------------------------------------
# Strategy name → engine class mapping
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "king": KingMoSEngine,
    "democracy": DemocracyMoSEngine,
    "duopoly": DuopolyMoSEngine,
    "tournament": TournamentMoSEngine,
    "cascade": CascadeMoSEngine,
    "jury": JuryMoSEngine,
    "chain": ChainMoSEngine,
    "committee": CommitteeMoSEngine,
}

# ---------------------------------------------------------------------------
# Default engine factory
# ---------------------------------------------------------------------------


def default_mos_factory(params: Dict[str, Any]) -> Any:
    """Build a MoS engine from a params dict with well-known keys.

    Expected keys:
      - ``"strategy"``: str — one of "king", "democracy", "duopoly",
        "tournament", "cascade", "jury", "chain", "committee"
      - ``"workers"``: List[Dict] — worker plugin configs
      - ``"king"``/``"referee"``/``"scorer"``: Dict — king/referee/scorer cfg
      - ``"voters"``/``"jurors"``: List[Dict] — voter/juror plugin configs
      - ``"founders"``: List[Dict] — founder plugin configs
      - ``"president"``: Dict — president plugin config
      - Plus any numeric params (threshold, max_rounds, etc.)

    Uses ``factory._load_chat_worker`` / ``_load_reranker_worker`` for
    sub-plugin instantiation.

    ``default_mos_factory`` — ``ovos_MoS/search.py:140``
    """
    strategy = params.get("strategy")
    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Expected one of: {sorted(_STRATEGY_MAP)}"
        )

    worker_cfgs = params.get("workers", [])
    workers = _load_workers(worker_cfgs)
    if not workers:
        raise ValueError("At least one worker must be loadable.")

    # Collect numeric config
    config: Dict[str, Any] = {}
    _NUMERIC_KEYS = {
        "threshold", "max_rounds", "discussion_rounds", "worker_timeout",
        "max_workers", "system_prompt", "prompt_template",
        "refinement_prompt", "revision_prompt", "discuss_prompt",
    }
    for k, v in params.items():
        if k in _NUMERIC_KEYS:
            config[k] = v

    engine_cls = _STRATEGY_MAP[strategy]

    if strategy == "king":
        king_cfg = params.get("king", {})
        # Try reranker first, fall back to chat
        try:
            king = _load_reranker_worker(king_cfg)
        except Exception:
            king = _load_chat_worker(king_cfg)
        return engine_cls(config=config, king=king, workers=workers)

    if strategy == "democracy":
        voter_cfgs = params.get("voters", [])
        voters = [_load_reranker_worker(c) for c in voter_cfgs]
        president_cfg = params.get("president")
        president = None
        if president_cfg:
            try:
                president = _load_reranker_worker(president_cfg)
            except Exception:
                president = _load_chat_worker(president_cfg)
        return engine_cls(config=config, voters=voters, workers=workers,
                          president=president)

    if strategy == "duopoly":
        founder_cfgs = params.get("founders", worker_cfgs)
        founders = [_load_chat_worker(c) for c in founder_cfgs]
        president_cfg = params.get("president", {})
        try:
            president = _load_reranker_worker(president_cfg)
        except Exception:
            president = _load_chat_worker(president_cfg)
        return engine_cls(config=config, president=president,
                          founders=founders, workers=workers)

    if strategy == "tournament":
        referee_cfg = params.get("referee", {})
        referee = _load_reranker_worker(referee_cfg)
        return engine_cls(config=config, referee=referee, workers=workers)

    if strategy == "cascade":
        scorer_cfg = params.get("scorer", {})
        scorer = _load_reranker_worker(scorer_cfg)
        return engine_cls(config=config, scorer=scorer, workers=workers)

    if strategy == "jury":
        juror_cfgs = params.get("jurors", [])
        jurors = [_load_reranker_worker(c) for c in juror_cfgs]
        juror_weights = params.get("juror_weights")
        president_cfg = params.get("president")
        president = None
        if president_cfg:
            try:
                president = _load_reranker_worker(president_cfg)
            except Exception:
                president = _load_chat_worker(president_cfg)
        kwargs: Dict[str, Any] = {
            "config": config, "jurors": jurors, "workers": workers,
            "president": president,
        }
        if juror_weights is not None:
            kwargs["juror_weights"] = juror_weights
        return engine_cls(**kwargs)

    if strategy == "chain":
        president_cfg = params.get("president")
        president = None
        if president_cfg:
            try:
                president = _load_reranker_worker(president_cfg)
            except Exception:
                president = _load_chat_worker(president_cfg)
        return engine_cls(config=config, workers=workers,
                          president=president)

    # committee
    president_cfg = params.get("president")
    president = None
    if president_cfg:
        try:
            president = _load_reranker_worker(president_cfg)
        except Exception:
            president = _load_chat_worker(president_cfg)
    return engine_cls(config=config, workers=workers, president=president)


# ---------------------------------------------------------------------------
# Plugin discovery helper
# ---------------------------------------------------------------------------


def discover_search_space(
    include_strategies: Optional[List[str]] = None,
    worker_pool: Optional[List[Dict[str, Any]]] = None,
    king_pool: Optional[List[Dict[str, Any]]] = None,
    reranker_pool: Optional[List[Dict[str, Any]]] = None,
) -> SearchSpace:
    """Auto-generate a search space from installed OPM plugins.

    If pools are not provided, uses ``find_chat_plugins()`` /
    ``find_reranker_plugins()`` to discover installed plugins (excluding
    ``ovos-mos-*`` self-references).

    Returns a ``SearchSpace`` with:
      - ``"strategy"``: CategoricalDim of strategy names
      - ``"workers"``: CategoricalDim of worker config combinations
      - ``"king"``: CategoricalDim of reranker/chat plugin configs
      - Strategy-specific numeric dims with sensible defaults

    ``discover_search_space`` — ``ovos_MoS/search.py:268``
    """
    from ovos_plugin_manager.agents import (
        find_chat_plugins, find_reranker_plugins,
    )

    # Discover installed plugins, filtering out MoS self-references
    if worker_pool is None:
        chat_names = [
            name for name in find_chat_plugins()
            if name not in _MOS_ENTRY_POINTS
        ]
        worker_pool = [{"module": name} for name in chat_names]

    if reranker_pool is None:
        rr_names = list(find_reranker_plugins())
        reranker_pool = [{"module": name} for name in rr_names]

    if king_pool is None:
        king_pool = list(reranker_pool)  # default: rerankers as kings

    strategies = include_strategies or list(_STRATEGY_MAP.keys())

    # Generate worker combinations (1, 2, 3 workers from pool)
    worker_combos: List[List[Dict[str, Any]]] = []
    for size in range(1, min(4, len(worker_pool) + 1)):
        for combo in itertools.combinations(worker_pool, size):
            worker_combos.append(list(combo))
    if not worker_combos and worker_pool:
        worker_combos = [worker_pool[:1]]

    space: SearchSpace = {
        "strategy": CategoricalDim(strategies),
        "workers": CategoricalDim(worker_combos) if worker_combos else CategoricalDim([[]]),
        "king": CategoricalDim(king_pool) if king_pool else CategoricalDim([{}]),
        "threshold": FloatRangeDim(0.5, 1.0),
        "max_rounds": IntRangeDim(1, 5),
        "discussion_rounds": IntRangeDim(1, 5),
        "worker_timeout": FloatRangeDim(10.0, 120.0),
    }

    if reranker_pool:
        space["scorer"] = CategoricalDim(reranker_pool)
        space["referee"] = CategoricalDim(reranker_pool)

        # Voter/juror combos (1–3 rerankers)
        rr_combos: List[List[Dict[str, Any]]] = []
        for size in range(1, min(4, len(reranker_pool) + 1)):
            for combo in itertools.combinations(reranker_pool, size):
                rr_combos.append(list(combo))
        if rr_combos:
            space["voters"] = CategoricalDim(rr_combos)
            space["jurors"] = CategoricalDim(rr_combos)

    return space


# ---------------------------------------------------------------------------
# Base search class
# ---------------------------------------------------------------------------


class _BaseSearch:
    """Common infrastructure for all search strategies."""

    def __init__(
        self,
        space: SearchSpace,
        engine_factory: Callable[[Dict[str, Any]], Any],
        evaluate: Callable[[Any], float],
        maximize: bool = True,
        seed: Optional[int] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self.space = space
        self.engine_factory = engine_factory
        self.evaluate = evaluate
        self.maximize = maximize
        self.rng = _random.Random(seed)
        self.metrics = metrics

    def _sample_random(self) -> Dict[str, Any]:
        """Sample a random point from the search space."""
        return {
            name: dim.sample(self.rng)
            for name, dim in self.space.items()
        }

    def _evaluate_params(self, params: Dict[str, Any]) -> SearchResult:
        """Build engine, evaluate, return result."""
        try:
            engine = self.engine_factory(params)
            score = self.evaluate(engine)
        except Exception as e:
            LOG.warning(f"Evaluation failed for params: {e}")
            score = float("-inf") if self.maximize else float("inf")
        return SearchResult(params=params, score=score)

    def _sort_results(
        self, results: List[SearchResult],
    ) -> List[SearchResult]:
        """Sort results best-first."""
        return sorted(results, key=lambda r: r.score,
                      reverse=self.maximize)

    def _make_report(
        self, results: List[SearchResult], search_type: str,
    ) -> SearchReport:
        """Build a SearchReport from evaluated results."""
        sorted_results = self._sort_results(results)
        return SearchReport(
            results=sorted_results,
            best=sorted_results[0],
            search_type=search_type,
            total_evaluations=len(sorted_results),
        )


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


class GridSearch(_BaseSearch):
    """Exhaustive grid search over the search space.

    ``FloatRangeDim`` values are discretised into ``float_steps`` evenly
    spaced points. ``FloatListDim`` elements are similarly discretised.
    ``max_evaluations`` caps the total number of configurations to
    evaluate (safety valve for combinatorial explosion).
    """

    def __init__(
        self,
        space: SearchSpace,
        engine_factory: Callable[[Dict[str, Any]], Any],
        evaluate: Callable[[Any], float],
        maximize: bool = True,
        seed: Optional[int] = None,
        metrics: Optional[Any] = None,
        float_steps: int = 10,
        max_evaluations: int = 10_000,
    ) -> None:
        super().__init__(space, engine_factory, evaluate, maximize,
                         seed, metrics)
        self.float_steps = float_steps
        self.max_evaluations = max_evaluations

    def _dim_values(self, dim: Dimension) -> List[Any]:
        """Enumerate all values for a dimension."""
        if isinstance(dim, CategoricalDim):
            return list(dim.choices)
        if isinstance(dim, IntRangeDim):
            return list(range(dim.low, dim.high + 1))
        if isinstance(dim, FloatRangeDim):
            if self.float_steps <= 1:
                return [dim.low]
            step = (dim.high - dim.low) / (self.float_steps - 1)
            return [dim.low + i * step for i in range(self.float_steps)]
        if isinstance(dim, FloatListDim):
            step = (dim.high - dim.low) / max(self.float_steps - 1, 1)
            element_vals = [dim.low + i * step
                            for i in range(self.float_steps)]
            return list(itertools.product(
                *[element_vals for _ in range(dim.length)]
            ))
        if isinstance(dim, IntListDim):
            element_vals = list(range(dim.low, dim.high + 1))
            return list(itertools.product(
                *[element_vals for _ in range(dim.length)]
            ))
        return [dim]  # pragma: no cover

    def run(self) -> SearchReport:
        """Execute grid search. Returns a ``SearchReport``."""
        names = list(self.space.keys())
        all_values = [self._dim_values(self.space[n]) for n in names]

        results: List[SearchResult] = []
        for combo in itertools.product(*all_values):
            if len(results) >= self.max_evaluations:
                LOG.warning(
                    f"GridSearch: hit max_evaluations cap "
                    f"({self.max_evaluations})"
                )
                break
            params = {}
            for name, val in zip(names, combo):
                if isinstance(val, tuple):
                    params[name] = list(val)
                else:
                    params[name] = val
            results.append(self._evaluate_params(params))

        return self._make_report(results, "grid")


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------


class RandomSearch(_BaseSearch):
    """Random sampling from the search space."""

    def run(self, n_trials: int = 50) -> SearchReport:
        """Evaluate ``n_trials`` random configurations."""
        results = [
            self._evaluate_params(self._sample_random())
            for _ in range(n_trials)
        ]
        return self._make_report(results, "random")


# ---------------------------------------------------------------------------
# Genetic search
# ---------------------------------------------------------------------------


class GeneticSearch(_BaseSearch):
    """Genetic algorithm over the search space.

    Uses tournament selection, uniform crossover, and per-dimension
    mutation. Elite individuals survive unchanged across generations.
    """

    def run(
        self,
        population_size: int = 20,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_count: int = 2,
    ) -> SearchReport:
        """Run the genetic search. Returns a ``SearchReport``."""
        # Initial population
        population = [self._sample_random()
                      for _ in range(population_size)]
        evaluated = [self._evaluate_params(p) for p in population]
        all_results: List[SearchResult] = list(evaluated)

        for gen in range(generations):
            sorted_pop = self._sort_results(evaluated)
            scores = [r.score for r in sorted_pop]

            # Elites survive
            next_gen_results: List[SearchResult] = sorted_pop[:elite_count]
            next_gen: List[Dict[str, Any]] = [
                r.params for r in next_gen_results
            ]

            # Fill the rest
            while len(next_gen) < population_size:
                parent_a = self._tournament_select(sorted_pop, scores)
                parent_b = self._tournament_select(sorted_pop, scores)

                if self.rng.random() < crossover_rate:
                    child = self._crossover(parent_a, parent_b)
                else:
                    child = dict(parent_a)

                if self.rng.random() < mutation_rate:
                    child = self._mutate(child)

                result = self._evaluate_params(child)
                next_gen.append(child)
                next_gen_results.append(result)
                all_results.append(result)

            evaluated = next_gen_results
            best = self._sort_results(evaluated)[0]
            LOG.debug(
                f"Generation {gen + 1}: best score = {best.score:.4f}"
            )

        return self._make_report(all_results, "genetic")

    def _tournament_select(
        self,
        population: List[SearchResult],
        scores: List[float],
        k: int = 3,
    ) -> Dict[str, Any]:
        """Select an individual via tournament selection."""
        candidates = self.rng.sample(
            range(len(population)),
            min(k, len(population)),
        )
        if self.maximize:
            winner = max(candidates, key=lambda i: scores[i])
        else:
            winner = min(candidates, key=lambda i: scores[i])
        return dict(population[winner].params)

    def _crossover(
        self,
        parent_a: Dict[str, Any],
        parent_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Uniform crossover: for each dimension, pick from either parent."""
        child: Dict[str, Any] = {}
        for name in self.space:
            if self.rng.random() < 0.5:
                child[name] = parent_a.get(name)
            else:
                child[name] = parent_b.get(name)
        return child

    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Per-dimension mutation."""
        result = dict(individual)
        for name, dim in self.space.items():
            if self.rng.random() > 0.3:
                continue  # only mutate ~30% of dimensions per call

            if isinstance(dim, CategoricalDim):
                result[name] = dim.sample(self.rng)

            elif isinstance(dim, IntRangeDim):
                delta = max(1, (dim.high - dim.low) // 5)
                val = result.get(name, dim.low)
                val += self.rng.randint(-delta, delta)
                result[name] = max(dim.low, min(dim.high, val))

            elif isinstance(dim, FloatRangeDim):
                spread = (dim.high - dim.low) * 0.1
                val = result.get(name, dim.low)
                val += self.rng.gauss(0, spread)
                result[name] = max(dim.low, min(dim.high, val))

            elif isinstance(dim, FloatListDim):
                lst = list(result.get(name, dim.sample(self.rng)))
                spread = (dim.high - dim.low) * 0.1
                for i in range(len(lst)):
                    if self.rng.random() < 0.5:
                        lst[i] += self.rng.gauss(0, spread)
                        lst[i] = max(dim.low, min(dim.high, lst[i]))
                result[name] = lst

            elif isinstance(dim, IntListDim):
                lst = list(result.get(name, dim.sample(self.rng)))
                delta = max(1, (dim.high - dim.low) // 5)
                for i in range(len(lst)):
                    if self.rng.random() < 0.5:
                        lst[i] += self.rng.randint(-delta, delta)
                        lst[i] = max(dim.low, min(dim.high, lst[i]))
                result[name] = lst

        return result


# ---------------------------------------------------------------------------
# MoS-as-judge evaluators
# ---------------------------------------------------------------------------


def make_reranker_evaluator(
    judge: "ReRankerEngine",
    benchmark: List[Tuple[str, str]],
) -> Callable[[Any], float]:
    """Create an evaluator that uses a ReRankerEngine to score answers.

    For each ``(query, reference)`` pair in the benchmark, the engine's
    answer is scored against the reference via the judge's ``rerank()``.
    Returns the average top score across all benchmark items.

    ``make_reranker_evaluator`` — ``ovos_MoS/search.py:640``

    Args:
        judge: A ``ReRankerEngine`` used to score engine answers against
            reference answers.
        benchmark: List of ``(query, reference_answer)`` pairs.

    Returns:
        A callable ``evaluate(engine) -> float`` suitable for search.
    """
    from ovos_plugin_manager.templates.agents import ReRankerEngine

    def evaluate(engine: Any) -> float:
        scores: List[float] = []
        for query, reference in benchmark:
            try:
                answer = engine.get_response(query)
                if not answer:
                    scores.append(0.0)
                    continue
                ranked = judge.rerank(query, [answer, reference])
                if ranked:
                    scores.append(ranked[0][0])
                else:
                    scores.append(0.0)
            except Exception as e:
                LOG.warning(f"Evaluation failed for '{query}': {e}")
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    return evaluate


def make_mos_judge_evaluator(
    judge_engine: Any,
    benchmark: List[Tuple[str, str]],
    score_reranker: "ReRankerEngine",
) -> Callable[[Any], float]:
    """Create an evaluator where a MoS engine judges candidate answers.

    The judge engine (itself a MoS) produces a reference-quality answer
    for each query. Then ``score_reranker`` scores the candidate engine's
    answer against the judge's answer. This enables evolving MoS
    configurations where one MoS evaluates another.

    ``make_mos_judge_evaluator`` — ``ovos_MoS/search.py:682``

    Args:
        judge_engine: A MoS engine (or any ``ChatEngine``) that produces
            high-quality reference answers.
        benchmark: List of ``(query, reference_answer)`` pairs. The
            reference is used as a fallback if the judge fails.
        score_reranker: A ``ReRankerEngine`` that scores candidate
            answers against the judge's answers.

    Returns:
        A callable ``evaluate(engine) -> float`` suitable for search.
    """
    def evaluate(engine: Any) -> float:
        scores: List[float] = []
        for query, fallback_ref in benchmark:
            try:
                judge_answer = judge_engine.get_response(query)
                reference = judge_answer if judge_answer else fallback_ref
            except Exception:
                reference = fallback_ref

            try:
                answer = engine.get_response(query)
                if not answer:
                    scores.append(0.0)
                    continue
                ranked = score_reranker.rerank(query, [answer, reference])
                if ranked:
                    scores.append(ranked[0][0])
                else:
                    scores.append(0.0)
            except Exception as e:
                LOG.warning(f"Evaluation failed for '{query}': {e}")
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    return evaluate


def make_latency_evaluator(
    queries: List[str],
) -> Callable[[Any], float]:
    """Create an evaluator that measures average response latency.

    Returns negative average time (since search maximizes by default).
    Use ``maximize=False`` or negate the score.

    ``make_latency_evaluator`` — ``ovos_MoS/search.py:730``

    Args:
        queries: List of test queries to benchmark.

    Returns:
        A callable ``evaluate(engine) -> float`` returning negative
        average latency in seconds.
    """
    import time

    def evaluate(engine: Any) -> float:
        times: List[float] = []
        for query in queries:
            t0 = time.monotonic()
            try:
                engine.get_response(query)
            except Exception:
                pass
            times.append(time.monotonic() - t0)
        return -(sum(times) / len(times)) if times else 0.0

    return evaluate
