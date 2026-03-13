
# MoS Strategies

## The King

One king decides the final answer from worker responses.

**ReRanker mode**: The king is a `ReRankerEngine` that scores and selects the best worker answer.

`KingMoSEngine.continue_chat` — `ovos_MoS/agents.py:75`

```python
from ovos_MoS.agents import KingMoSEngine

engine = KingMoSEngine(king=my_reranker, workers=[chat1, chat2, chat3])
answer = engine.get_response("What is the speed of light?")
```

**Generative mode**: The king is a `ChatEngine` (LLM) that synthesizes a new answer from all worker responses.

```python
engine = KingMoSEngine(king=my_llm, workers=[ddg_solver, wikipedia_solver])
answer = engine.get_response("Explain quantum entanglement")
```

The mode is detected automatically via `isinstance(self.king, ReRankerEngine)` at `ovos_MoS/agents.py:85`.

## Democracy

Multiple voters vote on worker answers. Majority wins.

`DemocracyMoSEngine.gather_votes` — `ovos_MoS/agents.py:124`
`DemocracyMoSEngine.continue_chat` — `ovos_MoS/agents.py:136`

```python
from ovos_MoS.agents import DemocracyMoSEngine

engine = DemocracyMoSEngine(
    voters=[reranker1, reranker2, reranker3],
    workers=[chat1, chat2]
)
answer = engine.get_response("Who wrote Hamlet?")
```

**With president** (tie-breaking or refinement):

```python
# ReRanker president reranks the voted answers
engine = DemocracyMoSEngine(
    voters=[rr1, rr2], workers=[chat1, chat2],
    president=my_reranker
)

# Generative president synthesizes from voted answers
engine = DemocracyMoSEngine(
    voters=[rr1, rr2], workers=[chat1, chat2],
    president=my_llm
)
```

## Duopoly

Founders engage in multi-round discussion about worker answers, then a president decides.

`DuopolyMoSEngine.continue_chat` — `ovos_MoS/agents.py:205`

```python
from ovos_MoS.agents import DuopolyMoSEngine

engine = DuopolyMoSEngine(
    president=my_reranker,
    founders=[llm1, llm2],
    workers=[ddg, wikipedia],
    config={"discussion_rounds": 3}
)
answer = engine.get_response("What caused the 2008 financial crisis?")
```

When no `workers` are specified, founders also serve as workers:

```python
engine = DuopolyMoSEngine(
    president=my_llm,
    founders=[llm1, llm2],
    config={"discussion_rounds": 2}
)
```

### Discussion flow

1. Workers provide initial answers (concurrent)
2. For each round, each founder receives a prompt containing the query, initial answers, and prior discussion
3. After all rounds, the president decides:
   - **ReRanker president**: founders generate final candidate answers, president reranks
   - **Generative president**: president synthesizes from the full discussion

### Config options

| Key | Default | Description |
|-----|---------|-------------|
| `discussion_rounds` | 3 | Number of discussion rounds |
| `system_prompt` | (built-in) | System prompt for final answer generation |
| `discuss_prompt` | (built-in) | System prompt for discussion rounds |
| `prompt_template` | (built-in) | Template with `{system}`, `{query}`, `{ans}`, `{discussion}` placeholders |

## Tournament

Bracket-style elimination. Workers provide answers, then pairs compete head-to-head via a ReRanker. Winners advance to the next round until one champion remains.

`TournamentMoSEngine.continue_chat` — `ovos_MoS/agents.py:279`

```python
from ovos_MoS.agents import TournamentMoSEngine

engine = TournamentMoSEngine(referee=my_reranker, workers=[c1, c2, c3, c4])
answer = engine.get_response("What is the tallest mountain?")
```

**Properties**:
- Only O(log N) reranker calls instead of N (efficient for many workers)
- If the number of candidates is odd, the last one gets a bye to the next round
- Referee failure on a match defaults to the first candidate in the pair

## Cascade

Sequential querying with early stopping. Workers are queried one by one in order of priority/cost. After each response, the scorer evaluates all accumulated answers. If the top score exceeds a threshold, stop early.

`CascadeMoSEngine.continue_chat` — `ovos_MoS/agents.py:329`

```python
from ovos_MoS.agents import CascadeMoSEngine

# Order workers cheapest to most expensive
engine = CascadeMoSEngine(
    scorer=my_reranker,
    workers=[cheap_cache, web_search, expensive_llm],
    config={"threshold": 0.8}
)
answer = engine.get_response("What year was Python created?")
```

**Properties**:
- Saves cost when the first good answer is often sufficient
- Workers are queried sequentially (not concurrently) — order matters
- If no answer meets the threshold, returns the best available after exhausting all workers

| Config key | Default | Description |
|------------|---------|-------------|
| `threshold` | `0.8` | Score threshold for early stopping |

## Jury

Weighted voting. Like Democracy but each juror has a numeric weight representing model quality, cost, or historical accuracy.

`JuryMoSEngine.gather_weighted_votes` — `ovos_MoS/agents.py:404`
`JuryMoSEngine.continue_chat` — `ovos_MoS/agents.py:416`

```python
from ovos_MoS.agents import JuryMoSEngine

engine = JuryMoSEngine(
    jurors=[cheap_rr, mid_rr, expert_rr],
    juror_weights=[1.0, 2.0, 5.0],  # expert's vote counts 5x
    workers=[chat1, chat2, chat3]
)
answer = engine.get_response("What is dark matter?")
```

**Properties**:
- With equal weights, behaves identically to Democracy
- Optional president for tie-breaking/refinement (same as Democracy)
- Default weight is 1.0 for all jurors if not specified

## Chain

Sequential refinement pipeline. Each worker builds on the previous worker's answer, progressively improving it.

`ChainMoSEngine.continue_chat` — `ovos_MoS/agents.py:472`

```python
from ovos_MoS.agents import ChainMoSEngine

engine = ChainMoSEngine(
    workers=[fast_drafter, careful_refiner, style_polisher],
    president=quality_reranker  # optional: picks best version
)
answer = engine.get_response("Explain CRISPR gene editing")
```

**Properties**:
- Worker 1 receives the raw query
- Workers 2+ receive a refinement prompt with the query and previous answer
- Optional president selects from all versions (reranker) or synthesizes (generative)
- Worker failures are skipped — the chain continues with the last good answer

| Config key | Default | Description |
|------------|---------|-------------|
| `refinement_prompt` | (built-in) | Template with `{query}` and `{previous}` placeholders |

## Committee

Multi-round convergence. All workers answer independently, then see each other's answers and revise. Repeats until convergence or max_rounds.

`CommitteeMoSEngine.continue_chat` — `ovos_MoS/agents.py:558`

```python
from ovos_MoS.agents import CommitteeMoSEngine

engine = CommitteeMoSEngine(
    workers=[llm1, llm2, llm3],
    president=my_reranker,
    config={"max_rounds": 3}
)
answer = engine.get_response("Is nuclear energy safe?")
```

**Properties**:
- Round 1: all workers answer independently (concurrent)
- Rounds 2+: each worker sees all other answers and revises
- Stops early if all answers converge (become identical)
- Optional president for final selection when no convergence

| Config key | Default | Description |
|------------|---------|-------------|
| `max_rounds` | `3` | Maximum revision rounds |
| `revision_prompt` | (built-in) | Template with `{query}`, `{others}`, `{previous}` |

## Recursive Composition

Every MoS engine is itself a `ChatEngine`, enabling composition:

```python
# A Democracy of Kings
king1 = KingMoSEngine(king=reranker1, workers=[chat1, chat2])
king2 = KingMoSEngine(king=reranker2, workers=[chat3, chat4])
democracy = DemocracyMoSEngine(
    voters=[voter1, voter2],
    workers=[king1, king2]  # MoS engines used as workers
)
```
