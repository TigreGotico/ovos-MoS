
# Composition Helpers

## Overview

`ovos_MoS.compose` provides convenience functions for common recursive MoS compositions. Every MoS engine is a `ChatEngine`, so engines can be nested as workers, kings, voters, founders, or presidents.

## Functions

### democracy_of_kings

`democracy_of_kings` — `ovos_MoS/compose.py:19`

Builds a Democracy where each worker is a King MoS.

```python
from ovos_MoS.compose import democracy_of_kings

engine = democracy_of_kings(
    king_configs=[
        {"king": reranker1, "workers": [chat1, chat2]},
        {"king": reranker2, "workers": [chat3, chat4]},
    ],
    voters=[voter1, voter2],
    president=tie_breaker,  # optional
)
answer = engine.get_response("What is dark matter?")
```

### cascade_then_committee

`cascade_then_committee` — `ovos_MoS/compose.py:57`

Builds a Committee where one worker is a Cascade (cheap-first with early stopping).

```python
from ovos_MoS.compose import cascade_then_committee

engine = cascade_then_committee(
    cascade_workers=[cache, web_search, expensive_llm],
    cascade_scorer=quality_scorer,
    committee_workers=[llm1, llm2],
    committee_president=final_reranker,
)
```

### chain_with_jury

`chain_with_jury` — `ovos_MoS/compose.py:91`

Builds a Jury where one worker is a Chain (sequential refinement).

```python
from ovos_MoS.compose import chain_with_jury

engine = chain_with_jury(
    chain_workers=[drafter, refiner, polisher],
    jurors=[cheap_rr, expert_rr],
    juror_weights=[1.0, 5.0],
)
```

### tournament_of_committees

`tournament_of_committees` — `ovos_MoS/compose.py:118`

Builds a Tournament where each worker is a Committee.

```python
from ovos_MoS.compose import tournament_of_committees

engine = tournament_of_committees(
    committee_configs=[
        {"workers": [llm1, llm2], "president": rr1},
        {"workers": [llm3, llm4], "president": rr2},
    ],
    referee=bracket_referee,
)
```

### duopoly_with_cascade_workers

`duopoly_with_cascade_workers` — `ovos_MoS/compose.py:143`

Builds a Duopoly where the workers include a Cascade for cost-efficient initial answers.

```python
from ovos_MoS.compose import duopoly_with_cascade_workers

engine = duopoly_with_cascade_workers(
    president=final_reranker,
    founders=[llm1, llm2],
    cascade_workers=[cache, web_search],
    cascade_scorer=scorer,
)
```
