
# Migration Guide: Legacy to Modern API

## Why Migrate?

The legacy classes in `ovos_MoS.__init__` inherit from `QuestionSolver`/`MultipleChoiceSolver` which are deprecated in OPM and will be removed in OPM 3.0. The modern API uses `ChatEngine`/`ReRankerEngine` from `ovos_plugin_manager.templates.agents`.

Importing `ovos_MoS` now emits a `DeprecationWarning` at `ovos_MoS/__init__.py:10`.

## Class Mapping

| Legacy Class | Modern Class |
|-------------|-------------|
| `AbstractMoS` | `AbstractMoSEngine` — `ovos_MoS/agents.py:17` |
| `AbstractKingMoS` | `KingMoSEngine` — `ovos_MoS/agents.py:53` |
| `ReRankerKingMoS` | `KingMoSEngine` with `ReRankerEngine` king |
| `GenerativeKingMoS` | `KingMoSEngine` with `ChatEngine` king |
| `DemocracyMoS` | `DemocracyMoSEngine` — `ovos_MoS/agents.py:100` |
| `ReRankerDemocracyMoS` | `DemocracyMoSEngine` with `ReRankerEngine` president |
| `GenerativeDemocracyMoS` | `DemocracyMoSEngine` with `ChatEngine` president |
| `AbstractDuopolyMoS` | `DuopolyMoSEngine` — `ovos_MoS/agents.py:172` |
| `ReRankerDuopolyMoS` | `DuopolyMoSEngine` with `ReRankerEngine` president |
| `GenerativeDuopolyMoS` | `DuopolyMoSEngine` with `ChatEngine` president |

## API Changes

### Constructor

**Legacy** (many parameters for translation, detection, priority):
```python
from ovos_MoS import ReRankerKingMoS
mos = ReRankerKingMoS(king=reranker, workers=[w1, w2],
                       config={}, translator=None, detector=None,
                       priority=50, enable_tx=False)
```

**Modern** (config-only):
```python
from ovos_MoS.agents import KingMoSEngine
engine = KingMoSEngine(king=reranker, workers=[w1, w2],
                        config={"max_workers": 4})
```

### Getting answers

**Legacy**:
```python
answer = mos.get_spoken_answer("query", lang="en-us")
# or
answer = mos.spoken_answer("query")  # inherited from QuestionSolver
```

**Modern**:
```python
answer = engine.get_response("query", lang="en-us")
# or for multi-turn:
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
msg = AgentMessage(role=MessageRole.USER, content="query")
response = engine.continue_chat([msg], lang="en-us")
print(response.content)
```

### Workers

**Legacy**: Workers must be `QuestionSolver` instances, queried sequentially.

**Modern**: Workers must be `ChatEngine` instances, queried concurrently via `ThreadPoolExecutor`.

### ReRanker vs Generative

**Legacy**: Separate classes (`ReRankerKingMoS` vs `GenerativeKingMoS`).

**Modern**: Single class (`KingMoSEngine`) detects the mode at runtime via `isinstance(king, ReRankerEngine)` at `ovos_MoS/agents.py:85`.

## Bugs Fixed in Legacy Code

If you're staying on the legacy API temporarily, these bugs were fixed in v0.1.0:

1. `AbstractDuopolyMoS.__init__` at `ovos_MoS/__init__.py:166`: `self.founders` was referenced before assignment — fixed to use parameter `founders`
2. `ReRankerDuopolyMoS.discuss_answers` at `ovos_MoS/__init__.py:478`: `select_answer` was missing the `answers` argument
3. Both Duopoly `discuss_answers` at `ovos_MoS/__init__.py:457` and `ovos_MoS/__init__.py:588`: redundant `gather_responses()` overwrote the `answers` parameter
