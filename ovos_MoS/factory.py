"""Config-driven OPM plugin wrappers for MoS engines.

Each factory class reads sub-plugin configurations, instantiates them via OPM,
creates the inner MoS engine, and delegates continue_chat.
"""
from typing import Optional, List, Dict, Any

from ovos_utils.log import LOG

from ovos_plugin_manager.templates.agents import (
    ChatEngine, ReRankerEngine, AgentMessage
)
from ovos_plugin_manager.agents import load_chat_plugin, load_reranker_plugin
from ovos_MoS.agents import (
    KingMoSEngine, DemocracyMoSEngine, DuopolyMoSEngine,
    TournamentMoSEngine, CascadeMoSEngine, JuryMoSEngine,
    ChainMoSEngine, CommitteeMoSEngine,
)

# Entry point names for self-loading guard
_MOS_ENTRY_POINTS = frozenset({
    "ovos-mos-king-reranker",
    "ovos-mos-king-generative",
    "ovos-mos-democracy",
    "ovos-mos-duopoly-reranker",
    "ovos-mos-duopoly-generative",
    "ovos-mos-tournament",
    "ovos-mos-cascade",
    "ovos-mos-jury",
    "ovos-mos-chain",
    "ovos-mos-committee",
})


def _load_chat_worker(cfg: Dict[str, Any]) -> ChatEngine:
    """Instantiate a ChatEngine sub-plugin from config."""
    module = cfg["module"]
    if module in _MOS_ENTRY_POINTS:
        raise ValueError(
            f"Self-loading guard: cannot load MoS plugin '{module}' "
            f"as a sub-plugin (infinite recursion)."
        )
    plugin_cls = load_chat_plugin(module)
    return plugin_cls(config=cfg)


def _load_reranker_worker(cfg: Dict[str, Any]) -> ReRankerEngine:
    """Instantiate a ReRankerEngine sub-plugin from config."""
    module = cfg["module"]
    if module in _MOS_ENTRY_POINTS:
        raise ValueError(
            f"Self-loading guard: cannot load MoS plugin '{module}' "
            f"as a sub-plugin (infinite recursion)."
        )
    plugin_cls = load_reranker_plugin(module)
    return plugin_cls(config=cfg)


def _load_workers(worker_configs: List[Dict[str, Any]]) -> List[ChatEngine]:
    """Load a list of ChatEngine workers from config dicts."""
    workers = []
    for cfg in worker_configs:
        try:
            workers.append(_load_chat_worker(cfg))
        except Exception as e:
            LOG.error(f"Failed to load worker '{cfg.get('module', '?')}': {e}")
    return workers


class ReRankerKingMoSPlugin(ChatEngine):
    """OPM plugin: King MoS with a ReRankerEngine king."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        king_cfg = self.config.get("king", {})
        worker_cfgs = self.config.get("workers", [])
        king = _load_reranker_worker(king_cfg)
        workers = _load_workers(worker_cfgs)
        self._engine = KingMoSEngine(config=self.config, king=king, workers=workers)

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class GenerativeKingMoSPlugin(ChatEngine):
    """OPM plugin: King MoS with a ChatEngine king (generative)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        king_cfg = self.config.get("king", {})
        worker_cfgs = self.config.get("workers", [])
        king = _load_chat_worker(king_cfg)
        workers = _load_workers(worker_cfgs)
        self._engine = KingMoSEngine(config=self.config, king=king, workers=workers)

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class DemocracyMoSPlugin(ChatEngine):
    """OPM plugin: Democracy MoS with voter ReRankerEngines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        voter_cfgs = self.config.get("voters", [])
        worker_cfgs = self.config.get("workers", [])
        president_cfg = self.config.get("president")
        voters = [_load_reranker_worker(c) for c in voter_cfgs]
        workers = _load_workers(worker_cfgs)
        president = None
        if president_cfg:
            if president_cfg.get("type") == "reranker":
                president = _load_reranker_worker(president_cfg)
            else:
                president = _load_chat_worker(president_cfg)
        self._engine = DemocracyMoSEngine(
            config=self.config, voters=voters, workers=workers,
            president=president
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class ReRankerDuopolyMoSPlugin(ChatEngine):
    """OPM plugin: Duopoly MoS with ReRankerEngine president."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        president_cfg = self.config.get("president", {})
        founder_cfgs = self.config.get("founders", [])
        worker_cfgs = self.config.get("workers", founder_cfgs)
        president = _load_reranker_worker(president_cfg)
        founders = [_load_chat_worker(c) for c in founder_cfgs]
        workers = _load_workers(worker_cfgs) if worker_cfgs is not founder_cfgs else founders
        self._engine = DuopolyMoSEngine(
            config=self.config, president=president,
            founders=founders, workers=workers
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class GenerativeDuopolyMoSPlugin(ChatEngine):
    """OPM plugin: Duopoly MoS with ChatEngine president (generative)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        president_cfg = self.config.get("president", {})
        founder_cfgs = self.config.get("founders", [])
        worker_cfgs = self.config.get("workers", founder_cfgs)
        president = _load_chat_worker(president_cfg)
        founders = [_load_chat_worker(c) for c in founder_cfgs]
        workers = _load_workers(worker_cfgs) if worker_cfgs is not founder_cfgs else founders
        self._engine = DuopolyMoSEngine(
            config=self.config, president=president,
            founders=founders, workers=workers
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class TournamentMoSPlugin(ChatEngine):
    """OPM plugin: Tournament MoS with bracket-style elimination."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        referee_cfg = self.config.get("referee", {})
        worker_cfgs = self.config.get("workers", [])
        referee = _load_reranker_worker(referee_cfg)
        workers = _load_workers(worker_cfgs)
        self._engine = TournamentMoSEngine(
            config=self.config, referee=referee, workers=workers
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class CascadeMoSPlugin(ChatEngine):
    """OPM plugin: Cascade MoS with sequential early stopping."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        scorer_cfg = self.config.get("scorer", {})
        worker_cfgs = self.config.get("workers", [])
        scorer = _load_reranker_worker(scorer_cfg)
        workers = _load_workers(worker_cfgs)
        self._engine = CascadeMoSEngine(
            config=self.config, scorer=scorer, workers=workers
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class JuryMoSPlugin(ChatEngine):
    """OPM plugin: Jury MoS with weighted voting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        juror_cfgs = self.config.get("jurors", [])
        worker_cfgs = self.config.get("workers", [])
        president_cfg = self.config.get("president")
        jurors = [_load_reranker_worker(c) for c in juror_cfgs]
        juror_weights = self.config.get("juror_weights", [1.0] * len(jurors))
        workers = _load_workers(worker_cfgs)
        president = None
        if president_cfg:
            if president_cfg.get("type") == "reranker":
                president = _load_reranker_worker(president_cfg)
            else:
                president = _load_chat_worker(president_cfg)
        self._engine = JuryMoSEngine(
            config=self.config, jurors=jurors, juror_weights=juror_weights,
            workers=workers, president=president
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class ChainMoSPlugin(ChatEngine):
    """OPM plugin: Chain MoS with sequential refinement."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        worker_cfgs = self.config.get("workers", [])
        president_cfg = self.config.get("president")
        workers = _load_workers(worker_cfgs)
        president = None
        if president_cfg:
            if president_cfg.get("type") == "reranker":
                president = _load_reranker_worker(president_cfg)
            else:
                president = _load_chat_worker(president_cfg)
        self._engine = ChainMoSEngine(
            config=self.config, workers=workers, president=president
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)


class CommitteeMoSPlugin(ChatEngine):
    """OPM plugin: Committee MoS with multi-round convergence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        worker_cfgs = self.config.get("workers", [])
        president_cfg = self.config.get("president")
        workers = _load_workers(worker_cfgs)
        president = None
        if president_cfg:
            if president_cfg.get("type") == "reranker":
                president = _load_reranker_worker(president_cfg)
            else:
                president = _load_chat_worker(president_cfg)
        self._engine = CommitteeMoSEngine(
            config=self.config, workers=workers, president=president
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return self._engine.continue_chat(messages, session_id, lang, units)
