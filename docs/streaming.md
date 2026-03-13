
# Streaming Support

## Overview

MoS engines inherit `stream_tokens()` and `stream_sentences()` from `ChatEngine`. The base implementation calls `continue_chat()` and splits the result. For strategies with generative kings/presidents, streaming mixins delegate directly to the inner generative engine for real-time streaming.

## Streaming Mixins

`StreamingKingMixin` — `ovos_MoS/_streaming.py:14`

Applied to `KingMoSEngine`. When the king is a generative `ChatEngine`, streaming is delegated to `king.stream_tokens()` / `king.stream_sentences()`. When the king is a `ReRankerEngine`, falls back to the base class default.

`StreamingChainMixin` — `ovos_MoS/_streaming.py:62`

Applied to `ChainMoSEngine`. When a generative president is set, the chain runs all workers sequentially, then streams the president's synthesis call.

`StreamingDuopolyMixin` — `ovos_MoS/_streaming.py:102`

Applied to `DuopolyMoSEngine`. When the president is a generative `ChatEngine`, discussion rounds execute normally, then the president's final answer is streamed.

`StreamingCommitteeMixin` — `ovos_MoS/_streaming.py:144`

Applied to `CommitteeMoSEngine`. Falls back to base class since the committee logic is complex (multi-round convergence).

## Usage

```python
from ovos_MoS.agents import KingMoSEngine

engine = KingMoSEngine(king=my_llm, workers=[chat1, chat2])

# Token streaming (partial words)
for token in engine.stream_tokens(messages):
    print(token, end="", flush=True)

# Sentence streaming (TTS-ready)
for sentence in engine.stream_sentences(messages):
    tts.speak(sentence)
```

## Strategy Support Matrix

| Strategy | Generative Streaming | ReRanker Streaming |
|----------|---------------------|--------------------|
| King | Delegates to king | Base class fallback |
| Democracy | Base class fallback | Base class fallback |
| Duopoly | Delegates to president | Base class fallback |
| Tournament | N/A (ReRanker only) | Base class fallback |
| Cascade | N/A (ReRanker scorer) | Base class fallback |
| Jury | Base class fallback | Base class fallback |
| Chain | Delegates to president | Base class fallback |
| Committee | Base class fallback | Base class fallback |
