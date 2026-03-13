
# Hyperparameter & Architecture Search

`ovos_MoS.search` optimises the **entire MoS configuration** — which strategy to use, which plugins to combine, and what numeric parameters to set — by evaluating candidate configurations against a benchmark dataset.

## The Core Loop

Every search follows the same pattern:

```
for each candidate config:
    1. Build an engine from the config          (engine_factory)
    2. Ask the engine every question in the      (benchmark)
       benchmark
    3. Score each answer                         (evaluate)
    4. Average the scores → fitness of this config
```

The search algorithms (grid, random, genetic) differ only in **how they pick the next candidate config**. Everything else — building, querying, scoring — is the same.

---

## Step 1: Build a Benchmark Dataset

The benchmark is the most important part. It is a list of `(query, reference_answer)` pairs that represents the kind of questions your MoS will face in production.

### What the benchmark looks like

```python
benchmark = [
    ("What is the speed of light?",
     "The speed of light in vacuum is approximately 299,792 km/s."),
    ("Who wrote Romeo and Juliet?",
     "William Shakespeare wrote Romeo and Juliet."),
    ("What is the capital of Portugal?",
     "Lisbon is the capital of Portugal."),
    # ... 20–200 more pairs
]
```

Each pair is `(query: str, reference_answer: str)`:
- **query** — the question you would ask your voice assistant
- **reference_answer** — a known-good answer to compare against

### Requirements for a good benchmark

| Requirement | Why | Guideline |
|-------------|-----|-----------|
| **Size** | Too few questions → noisy scores, too many → slow evaluation | 30–200 pairs is the sweet spot. Under 30 the genetic algorithm can't distinguish good configs from lucky ones. Over 200 and each evaluation takes too long. |
| **Diversity** | If all questions are about geography, you optimise for geography at the expense of everything else | Mix topic areas: factual recall, explanations, current events, math, opinions. Match the distribution your assistant actually sees. |
| **Unambiguous references** | The scorer compares candidate answers to references. Vague references produce noisy scores. | Write references that are short, factual, and contain the key fact. Don't write essay-length references. |
| **Difficulty spread** | If every question is trivial, every config scores the same and the search has no gradient | Include easy questions (most configs get right), medium (some get right), and hard (few get right). |
| **Stability** | Questions about today's weather or live sports change answers over time | Use time-stable facts. Avoid questions whose answers change daily. |

### Where to get benchmark data

**Option A: Write your own.** Best for domain-specific assistants. Write 50–100 questions that represent real user queries. Write short reference answers.

**Option B: Use existing QA datasets.** Several open datasets work well:

```python
# SQuAD-style: (context, question, answer) → just use (question, answer)
benchmark = [
    ("When was the Eiffel Tower built?", "The Eiffel Tower was built in 1889."),
    ("What river runs through London?", "The Thames runs through London."),
]

# TriviaQA, Natural Questions, or any QA corpus:
# Extract (question, short_answer) pairs
import json
with open("trivia_qa.jsonl") as f:
    benchmark = [
        (item["question"], item["answer"])
        for item in (json.loads(line) for line in f)
    ][:100]  # take first 100
```

**Option C: Generate from logs.** If you have conversation logs from your voice assistant, extract real user questions and manually write reference answers for a subset.

**Option D: No references at all.** If you truly have no reference answers, you can still evaluate — use a MoS-as-judge (a strong MoS generates the reference on the fly) or evaluate on proxy metrics like response length, latency, or whether the engine returned anything at all.

### Loading from CSV

A common pattern is to store the benchmark in a CSV file:

```csv
query,reference
What is the speed of light?,The speed of light is approximately 299792 km/s.
Who painted the Mona Lisa?,Leonardo da Vinci painted the Mona Lisa.
```

```python
import csv

def load_benchmark(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [(row["query"], row["reference"]) for row in reader]

benchmark = load_benchmark("benchmark.csv")
```

---

## Step 2: Choose How to Score Answers

Scoring is the second critical decision. The evaluator receives a built engine and must return a single float. Higher = better (by default).

### Option A: ReRanker scoring (recommended)

A `ReRankerEngine` (cross-encoder, BM25, etc.) scores the candidate answer against the reference. This is the most practical approach because OVOS already has reranker plugins installed.

How it works internally:

```
For each (query, reference) in benchmark:
    candidate_answer = engine.get_response(query)
    ranked = judge.rerank(query, [candidate_answer, reference])
    score = ranked[0][0]   # score of the top-ranked document
```

If the candidate answer is better than (or as good as) the reference, the reranker gives it a high score. If it's worse, low score. The average across all benchmark items is the fitness.

```python
from ovos_MoS.search import make_reranker_evaluator

# Any installed ReRankerEngine works
from ovos_plugin_manager.agents import load_reranker_plugin
judge_cls = load_reranker_plugin("ovos-reranker-bm25-plugin")
judge = judge_cls()

evaluator = make_reranker_evaluator(judge, benchmark)
# evaluator(engine) → float between 0.0 and 1.0
```

`make_reranker_evaluator` — `ovos_MoS/search.py:640`

**Which reranker to use as judge?**

| Reranker | Speed | Quality | When to use |
|----------|-------|---------|-------------|
| BM25 | Fast | Lexical overlap only | Quick iteration, large benchmarks |
| Cross-encoder | Slow | Semantic similarity | Final tuning, small benchmarks |
| Sentence-transformer | Medium | Embedding cosine | Good balance |

The judge reranker should **not** be one of the rerankers being searched over. Use a separate, trusted reranker as the judge. If the judge is also a search candidate, the search will trivially optimise for "pick the judge as king" — which teaches you nothing.

### Option B: MoS-as-judge (evolving MoS with MoS)

Instead of fixed reference answers, a **strong MoS configuration** generates reference answers on the fly. Then a reranker scores candidates against the judge's answers.

This is useful when:
- You don't have pre-written reference answers
- You want to evolve cheap/fast MoS configs that match a slow/expensive one
- You want to distill a large ensemble into a smaller one

```python
from ovos_MoS.search import make_mos_judge_evaluator
from ovos_MoS.agents import KingMoSEngine

# Build a strong (but slow/expensive) judge MoS
judge = KingMoSEngine(
    king=best_cross_encoder,
    workers=[llm_gpt4, llm_claude, llm_local],
    config={"worker_timeout": 60},
)

# The scoring reranker compares candidate vs judge answers
scoring_rr = load_reranker_plugin("ovos-reranker-cross-encoder-plugin")()

# benchmark still has (query, fallback_reference) pairs
# The fallback is used if the judge engine fails on a query
evaluator = make_mos_judge_evaluator(judge, benchmark, scoring_rr)
```

`make_mos_judge_evaluator` — `ovos_MoS/search.py:682`

**How the scoring works:**

```
For each (query, fallback_ref) in benchmark:
    judge_answer = judge_engine.get_response(query)  # strong MoS answers
    reference = judge_answer or fallback_ref          # fallback if judge fails
    candidate_answer = engine.get_response(query)     # candidate MoS answers
    score = scoring_rr.rerank(query, [candidate, reference])[0][0]
```

The search evolves candidate configs that produce answers as good as the judge's — but potentially with fewer workers, cheaper plugins, or faster strategies.

### Option C: Latency evaluation

Optimise for speed rather than quality:

```python
from ovos_MoS.search import make_latency_evaluator

evaluator = make_latency_evaluator(["What is AI?", "Explain DNS", ...])
# Returns negative avg latency (search maximizes, so faster = higher score)
```

`make_latency_evaluator` — `ovos_MoS/search.py:730`

### Option D: Custom evaluator

Any `Callable[[engine], float]` works. Common patterns:

```python
# Binary: did the engine return a non-empty answer?
def evaluate(engine):
    hits = sum(1 for q, _ in benchmark if engine.get_response(q))
    return hits / len(benchmark)

# Length-weighted: longer answers score higher (crude proxy for detail)
def evaluate(engine):
    return sum(len(engine.get_response(q) or "") for q, _ in benchmark)

# Multi-objective: quality minus latency penalty
def evaluate(engine):
    quality = quality_evaluator(engine)    # 0.0–1.0
    latency = -latency_evaluator(engine)   # positive seconds
    return quality - 0.05 * latency        # penalise slow configs
```

### Option E: No benchmark at all (reference-free)

If you have queries but no reference answers:

```python
# Just check that the engine produces *something* for each query
queries = ["What is Python?", "Who is Turing?", "Explain TCP"]

def evaluate(engine):
    results = [engine.get_response(q) for q in queries]
    non_empty = sum(1 for r in results if r and len(r) > 10)
    return non_empty / len(queries)
```

This is weak — it only measures "did the engine respond?" — but it's enough to eliminate broken configs and find which strategies/plugins are functional.

---

## Step 3: Define the Search Space

The search space tells the algorithm what to explore. Each key maps to a dimension type.

### Dimension types

| Type | What it represents | Example |
|------|-------------------|---------|
| `CategoricalDim(choices)` | Pick one from a list | Strategy name, plugin config dict, list of worker configs |
| `IntRangeDim(low, high)` | Integer in [low, high] | `max_rounds` (1–5), `discussion_rounds` (1–5) |
| `FloatRangeDim(low, high)` | Float in [low, high] | `threshold` (0.5–1.0), `worker_timeout` (10–120) |
| `FloatListDim(length, low, high)` | Fixed-length float vector | `juror_weights` for Jury strategy |
| `IntListDim(length, low, high)` | Fixed-length int vector | Worker subset indices |

`CategoricalDim` — `ovos_MoS/search.py:36`
`IntRangeDim` — `ovos_MoS/search.py:47`
`FloatRangeDim` — `ovos_MoS/search.py:55`
`FloatListDim` — `ovos_MoS/search.py:63`
`IntListDim` — `ovos_MoS/search.py:73`

### Manual search space

```python
from ovos_MoS.search import (
    CategoricalDim, IntRangeDim, FloatRangeDim, FloatListDim,
)

space = {
    # Which strategy architecture to use
    "strategy": CategoricalDim(["king", "democracy", "cascade", "jury"]),

    # Which worker plugin combinations to try
    "workers": CategoricalDim([
        [{"module": "ovos-solver-plugin-ddg"}],
        [{"module": "ovos-solver-plugin-ddg"},
         {"module": "ovos-solver-plugin-wikipedia"}],
        [{"module": "ovos-solver-plugin-ddg"},
         {"module": "ovos-solver-plugin-wikipedia"},
         {"module": "ovos-solver-plugin-wolfram"}],
    ]),

    # Which king/scorer/referee to try
    "king": CategoricalDim([
        {"module": "ovos-reranker-bm25-plugin"},
        {"module": "ovos-reranker-cross-encoder-plugin"},
    ]),

    # Numeric hyperparameters
    "threshold": FloatRangeDim(0.5, 1.0),
    "max_rounds": IntRangeDim(1, 5),
    "juror_weights": FloatListDim(length=3, low=0.5, high=5.0),
}
```

**Important:** The `"workers"` dimension is a `CategoricalDim` where each choice is a **list of plugin config dicts**. The search doesn't pick individual workers — it picks entire worker sets. This is because strategy behaviour changes with the number of workers (e.g., Tournament needs at least 2, Jury needs juror count to match `juror_weights` length).

### Auto-discovered search space

If you just want to try everything that's installed:

```python
from ovos_MoS.search import discover_search_space

space = discover_search_space()
```

`discover_search_space` — `ovos_MoS/search.py:268`

This uses `find_chat_plugins()` and `find_reranker_plugins()` from OPM to discover installed plugins, filters out `ovos-mos-*` self-references (to prevent infinite recursion), and generates worker combinations (1, 2, and 3 workers from the pool). You can narrow it:

```python
space = discover_search_space(
    include_strategies=["king", "cascade"],
    worker_pool=[{"module": "ovos-solver-plugin-ddg"}],
    reranker_pool=[{"module": "ovos-reranker-bm25-plugin"}],
)
```

### Which dimensions matter for which strategy?

Not all dimensions apply to all strategies. The `default_mos_factory` ignores irrelevant keys:

| Dimension | king | democracy | duopoly | tournament | cascade | jury | chain | committee |
|-----------|------|-----------|---------|------------|---------|------|-------|-----------|
| `workers` | yes | yes | yes | yes | yes | yes | yes | yes |
| `king` | **yes** | — | — | — | — | — | — | — |
| `voters` | — | **yes** | — | — | — | — | — | — |
| `jurors` | — | — | — | — | — | **yes** | — | — |
| `juror_weights` | — | — | — | — | — | **yes** | — | — |
| `referee` | — | — | — | **yes** | — | — | — | — |
| `scorer` | — | — | — | — | **yes** | — | — | — |
| `president` | — | optional | **yes** | — | — | optional | optional | optional |
| `founders` | — | — | **yes** | — | — | — | — | — |
| `threshold` | — | — | — | — | **yes** | — | — | — |
| `max_rounds` | — | — | — | — | — | — | — | **yes** |
| `discussion_rounds` | — | — | **yes** | — | — | — | — | — |

Including irrelevant dimensions wastes search budget but doesn't break anything — extra keys are simply ignored by the factory.

---

## Step 4: Build the Engine Factory

The factory turns a sampled params dict into a runnable engine.

### Default factory

`default_mos_factory` handles all 8 strategies:

```python
from ovos_MoS.search import default_mos_factory

engine = default_mos_factory({
    "strategy": "king",
    "workers": [{"module": "ovos-solver-plugin-ddg"}],
    "king": {"module": "ovos-reranker-bm25-plugin"},
    "threshold": 0.8,  # ignored for king, harmless
})
```

`default_mos_factory` — `ovos_MoS/search.py:140`

It maps `"strategy"` → engine class, loads sub-plugins via `_load_chat_worker`/`_load_reranker_worker` from `ovos_MoS/factory.py`, and passes numeric params as config.

### Custom factory

If you need non-standard plugin loading or want to inject pre-built workers:

```python
def my_factory(params):
    strategy = params["strategy"]
    if strategy == "king":
        return KingMoSEngine(
            king=my_preloaded_reranker,
            workers=my_preloaded_workers[:params.get("n_workers", 2)],
            config={"worker_timeout": params.get("worker_timeout", 30)},
        )
    elif strategy == "cascade":
        return CascadeMoSEngine(
            scorer=my_preloaded_reranker,
            workers=my_preloaded_workers,
            config={"threshold": params["threshold"]},
        )
    raise ValueError(f"Unknown strategy: {strategy}")
```

---

## Step 5: Run the Search

### RandomSearch (start here)

Fast exploration. Sample random configs and evaluate. Use this first to get a baseline and identify which strategies/plugins are promising.

```python
from ovos_MoS.search import RandomSearch

rs = RandomSearch(
    space=space,
    engine_factory=default_mos_factory,
    evaluate=evaluator,
    seed=42,
)
report = rs.run(n_trials=50)

print(f"Best: {report.best.params['strategy']} — {report.best.score:.3f}")
for r in report.top_k(5):
    print(f"  {r.score:.3f} — {r.params['strategy']}")
```

`RandomSearch.run` — `ovos_MoS/search.py:459`

### GeneticSearch (refine)

Once you know which strategies/plugins work, narrow the search space and use genetic search to fine-tune numeric parameters.

```python
from ovos_MoS.search import GeneticSearch

ga = GeneticSearch(
    space=space,
    engine_factory=default_mos_factory,
    evaluate=evaluator,
    seed=42,
)
report = ga.run(
    population_size=20,   # configs per generation
    generations=50,       # number of generations
    mutation_rate=0.1,    # probability of mutating a child
    crossover_rate=0.7,   # probability of crossover vs clone
    elite_count=2,        # top N survive unchanged
)
```

`GeneticSearch.run` — `ovos_MoS/search.py:474`

**How the genetic algorithm evolves MoS configs:**

1. **Generation 0:** 20 random configs are evaluated
2. **Selection:** Tournament selection (k=3) picks parents — better-scoring configs are more likely to be selected
3. **Crossover:** For each dimension, the child inherits from parent A or parent B with 50/50 chance. This mixes strategies, plugins, and parameters.
4. **Mutation:** Per-dimension perturbation:
   - `CategoricalDim` (strategy, plugins): random re-pick from choices
   - `IntRangeDim`: `current +/- randint(delta)`, clamped to bounds
   - `FloatRangeDim`: `current + gauss(0, range*0.1)`, clamped
   - `FloatListDim`/`IntListDim`: per-element perturbation
5. **Elites:** The top 2 configs survive unchanged to the next generation
6. **Repeat** for 50 generations

Over time, good combinations of strategy + plugins + parameters get reinforced while bad ones are eliminated.

### GridSearch (exhaustive)

Only practical for small spaces. Tries every combination.

```python
from ovos_MoS.search import GridSearch

gs = GridSearch(
    space=small_space,
    engine_factory=default_mos_factory,
    evaluate=evaluator,
    float_steps=5,          # discretise FloatRangeDim into 5 steps
    max_evaluations=1000,   # abort after 1000 evaluations
)
report = gs.run()
```

`GridSearch.run` — `ovos_MoS/search.py:430`

**Warning:** Grid search explodes combinatorially. 3 strategies * 4 worker combos * 5 float steps = 60 evaluations. Add 3 more dimensions and you're at 60 * 5^3 = 7,500. Use `max_evaluations` to cap it.

---

## Step 6: Interpret Results

```python
report.best              # SearchResult with highest score
report.best.params       # Dict — the winning config
report.best.score        # float — fitness score
report.results           # List[SearchResult] sorted best-first
report.top_k(5)          # top 5 results
report.total_evaluations # how many configs were tested
report.search_type       # "grid", "random", or "genetic"
```

`SearchResult` — `ovos_MoS/search.py:93`
`SearchReport` — `ovos_MoS/search.py:100`

### Using the winning config

```python
# Build the winning engine
best_engine = default_mos_factory(report.best.params)

# Use it
answer = best_engine.get_response("What is quantum entanglement?")

# Or export the config for use in production
import json
print(json.dumps(report.best.params, indent=2))
```

---

## Complete Walkthrough: From Zero to Evolved MoS

### Scenario
You have three solver plugins installed (`ovos-solver-plugin-ddg`, `ovos-solver-plugin-wikipedia`, `ovos-solver-plugin-wolfram`) and one reranker (`ovos-reranker-bm25-plugin`). You want to find the best MoS configuration.

### 1. Write a benchmark

```python
benchmark = [
    ("What is photosynthesis?",
     "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."),
    ("Who invented the telephone?",
     "Alexander Graham Bell is credited with inventing the telephone in 1876."),
    ("What is the largest ocean?",
     "The Pacific Ocean is the largest ocean on Earth."),
    ("How does a transistor work?",
     "A transistor is a semiconductor device that amplifies or switches electronic signals."),
    ("What causes tides?",
     "Tides are caused by the gravitational pull of the Moon and Sun on Earth's oceans."),
    # ... add 25–195 more pairs
]
```

### 2. Set up the judge

```python
from ovos_plugin_manager.agents import load_reranker_plugin

judge_cls = load_reranker_plugin("ovos-reranker-bm25-plugin")
judge = judge_cls()
```

### 3. Create the evaluator

```python
from ovos_MoS.search import make_reranker_evaluator

evaluator = make_reranker_evaluator(judge, benchmark)
```

### 4. Define the search space

```python
from ovos_MoS.search import (
    CategoricalDim, IntRangeDim, FloatRangeDim,
)

space = {
    "strategy": CategoricalDim(["king", "cascade", "chain", "committee"]),
    "workers": CategoricalDim([
        [{"module": "ovos-solver-plugin-ddg"}],
        [{"module": "ovos-solver-plugin-ddg"},
         {"module": "ovos-solver-plugin-wikipedia"}],
        [{"module": "ovos-solver-plugin-ddg"},
         {"module": "ovos-solver-plugin-wikipedia"},
         {"module": "ovos-solver-plugin-wolfram"}],
    ]),
    "king": CategoricalDim([
        {"module": "ovos-reranker-bm25-plugin"},
    ]),
    "scorer": CategoricalDim([
        {"module": "ovos-reranker-bm25-plugin"},
    ]),
    "threshold": FloatRangeDim(0.5, 0.95),
    "max_rounds": IntRangeDim(1, 4),
}
```

### 5. Run random search (explore)

```python
from ovos_MoS.search import RandomSearch, default_mos_factory

rs = RandomSearch(space, default_mos_factory, evaluator, seed=42)
report = rs.run(n_trials=30)

print(f"Random search best: {report.best.score:.3f}")
print(f"  Strategy: {report.best.params['strategy']}")
print(f"  Workers: {len(report.best.params['workers'])}")
```

### 6. Run genetic search (refine)

```python
from ovos_MoS.search import GeneticSearch

ga = GeneticSearch(space, default_mos_factory, evaluator, seed=42)
report = ga.run(population_size=15, generations=20)

print(f"\nGenetic search best: {report.best.score:.3f}")
print(f"  Strategy: {report.best.params['strategy']}")
for r in report.top_k(3):
    print(f"  {r.score:.3f} — {r.params['strategy']} "
          f"({len(r.params['workers'])} workers)")
```

### 7. Deploy the winner

```python
import json

best_config = report.best.params
print(json.dumps(best_config, indent=2))

# Build and use the winning engine
engine = default_mos_factory(best_config)
print(engine.get_response("What is dark matter?"))
```

---

## Cost and Time Considerations

Each evaluation calls `engine.get_response()` for every benchmark item. That means:

- **n_trials=50** with **benchmark size=100** = 5,000 `get_response()` calls
- **population=20, generations=50** = ~1,000 configs * 100 queries = 100,000 calls
- Each `get_response()` calls all workers in the engine (2–3 workers = 200,000–300,000 individual plugin calls)

**If your workers are API-based LLMs**, this can be expensive and slow. Strategies to manage cost:

1. **Start with a small benchmark** (30 queries) for exploration, then validate the top 5 configs on a larger benchmark
2. **Use cheap workers** (DDG, Wikipedia) during search, then swap in expensive ones for the final config
3. **Use RandomSearch first** (n_trials=20–50) to identify promising strategies before running GeneticSearch
4. **Cap evaluations** with `GridSearch(max_evaluations=100)` or few generations
5. **Use local models** (GGUF solvers, BM25 rerankers) — they're free and fast

**If your workers are local/free plugins** (DDG, Wikipedia, Wolfram, local GGUF), cost is not an issue — only time. A typical search with 50 random trials and a 50-query benchmark takes 5–15 minutes with local plugins.
