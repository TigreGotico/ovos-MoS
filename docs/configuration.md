
# Configuration

## OPM Entry Points

ovos-MoS registers 5 `opm.agents.chat` entry points that can be loaded from config:

| Entry Point | Factory Class | Strategy |
|-------------|---------------|----------|
| `ovos-mos-king-reranker` | `ReRankerKingMoSPlugin` — `ovos_MoS/factory.py:61` | King with ReRanker |
| `ovos-mos-king-generative` | `GenerativeKingMoSPlugin` — `ovos_MoS/factory.py:79` | King with ChatEngine |
| `ovos-mos-democracy` | `DemocracyMoSPlugin` — `ovos_MoS/factory.py:97` | Democracy |
| `ovos-mos-duopoly-reranker` | `ReRankerDuopolyMoSPlugin` — `ovos_MoS/factory.py:125` | Duopoly with ReRanker |
| `ovos-mos-duopoly-generative` | `GenerativeDuopolyMoSPlugin` — `ovos_MoS/factory.py:148` | Duopoly with ChatEngine |
| `ovos-mos-tournament` | `TournamentMoSPlugin` — `ovos_MoS/factory.py:180` | Bracket elimination |
| `ovos-mos-cascade` | `CascadeMoSPlugin` — `ovos_MoS/factory.py:200` | Sequential early stopping |
| `ovos-mos-jury` | `JuryMoSPlugin` — `ovos_MoS/factory.py:220` | Weighted voting |
| `ovos-mos-chain` | `ChainMoSPlugin` — `ovos_MoS/factory.py:249` | Sequential refinement |
| `ovos-mos-committee` | `CommitteeMoSPlugin` — `ovos_MoS/factory.py:274` | Multi-round convergence |

## Config Structure

### King (ReRanker)

```json
{
    "module": "ovos-mos-king-reranker",
    "king": {
        "module": "ovos-reranker-bm25-plugin",
        "config_key": "value"
    },
    "workers": [
        {"module": "ovos-solver-plugin-ddg"},
        {"module": "ovos-solver-plugin-wikipedia"}
    ],
    "max_workers": 4,
    "worker_timeout": 30
}
```

### King (Generative)

```json
{
    "module": "ovos-mos-king-generative",
    "king": {
        "module": "ovos-chat-openai-plugin",
        "key": "sk-...",
        "model": "gpt-4"
    },
    "workers": [
        {"module": "ovos-solver-plugin-ddg"},
        {"module": "ovos-solver-plugin-wikipedia"}
    ]
}
```

### Democracy

```json
{
    "module": "ovos-mos-democracy",
    "voters": [
        {"module": "ovos-reranker-bm25-plugin"},
        {"module": "ovos-reranker-cross-encoder-plugin"}
    ],
    "workers": [
        {"module": "ovos-solver-plugin-ddg"},
        {"module": "ovos-solver-plugin-wikipedia"}
    ],
    "president": {
        "module": "ovos-reranker-bm25-plugin",
        "type": "reranker"
    }
}
```

The `president` field is optional. If present, set `"type": "reranker"` to use a `ReRankerEngine`, otherwise it defaults to `ChatEngine`.

### Duopoly

```json
{
    "module": "ovos-mos-duopoly-generative",
    "president": {
        "module": "ovos-chat-openai-plugin",
        "model": "gpt-4"
    },
    "founders": [
        {"module": "ovos-chat-openai-plugin", "model": "gpt-4"},
        {"module": "ovos-chat-anthropic-plugin", "model": "claude-opus-4-6"}
    ],
    "workers": [
        {"module": "ovos-solver-plugin-ddg"}
    ],
    "discussion_rounds": 3,
    "max_workers": 4,
    "worker_timeout": 30
}
```

When `workers` is omitted, founders are used as workers.

## Common Config Keys

| Key | Default | Description |
|-----|---------|-------------|
| `max_workers` | `len(workers)` | Max concurrent threads for worker querying |
| `worker_timeout` | `30.0` | Timeout in seconds for worker responses |
| `discussion_rounds` | `3` | Duopoly: number of founder discussion rounds |
| `system_prompt` | (built-in) | Custom system prompt for final answer |
| `discuss_prompt` | (built-in) | Custom prompt for discussion rounds |
| `prompt_template` | (built-in) | Custom template with `{system}`, `{query}`, `{ans}`, `{discussion}` |

### Tournament

```json
{
    "module": "ovos-mos-tournament",
    "referee": {"module": "ovos-reranker-bm25-plugin"},
    "workers": [
        {"module": "ovos-solver-plugin-ddg"},
        {"module": "ovos-solver-plugin-wikipedia"},
        {"module": "ovos-solver-plugin-wolfram"}
    ]
}
```

### Cascade

```json
{
    "module": "ovos-mos-cascade",
    "scorer": {"module": "ovos-reranker-cross-encoder-plugin"},
    "workers": [
        {"module": "ovos-solver-plugin-cache"},
        {"module": "ovos-solver-plugin-ddg"},
        {"module": "ovos-chat-openai-plugin", "model": "gpt-4"}
    ],
    "threshold": 0.8
}
```

Workers are queried in order. Order by cost (cheapest first) for maximum savings.

### Jury

```json
{
    "module": "ovos-mos-jury",
    "jurors": [
        {"module": "ovos-reranker-bm25-plugin"},
        {"module": "ovos-reranker-cross-encoder-plugin"}
    ],
    "juror_weights": [1.0, 3.0],
    "workers": [
        {"module": "ovos-solver-plugin-ddg"},
        {"module": "ovos-solver-plugin-wikipedia"}
    ]
}
```

### Chain

```json
{
    "module": "ovos-mos-chain",
    "workers": [
        {"module": "ovos-solver-plugin-ddg"},
        {"module": "ovos-chat-openai-plugin", "model": "gpt-4"}
    ],
    "president": {
        "module": "ovos-reranker-bm25-plugin",
        "type": "reranker"
    }
}
```

Workers are queried sequentially. Each worker after the first receives a refinement prompt.

### Committee

```json
{
    "module": "ovos-mos-committee",
    "workers": [
        {"module": "ovos-chat-openai-plugin", "model": "gpt-4"},
        {"module": "ovos-chat-anthropic-plugin", "model": "claude-opus-4-6"},
        {"module": "ovos-chat-gemini-plugin", "model": "gemini-pro"}
    ],
    "president": {
        "module": "ovos-reranker-cross-encoder-plugin",
        "type": "reranker"
    },
    "max_rounds": 3
}
```

## Self-Loading Guard

`_MOS_ENTRY_POINTS` — `ovos_MoS/factory.py:17`

The factory prevents loading any `ovos-mos-*` entry point as a sub-plugin. This blocks infinite recursion scenarios like configuring an `ovos-mos-king-reranker` as a worker inside another `ovos-mos-king-reranker`.

`_load_chat_worker` — `ovos_MoS/factory.py:26`
`_load_reranker_worker` — `ovos_MoS/factory.py:38`

Both functions check the module name against `_MOS_ENTRY_POINTS` and raise `ValueError` if it matches.
