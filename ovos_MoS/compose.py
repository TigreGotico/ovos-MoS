"""Strategy composition helpers for MoS.

Convenience functions for building common recursive MoS compositions.
Every MoS engine is a ChatEngine, so any engine can serve as a worker,
king, voter, founder, or president in another engine.
"""
from typing import Optional, List, Dict, Any, Union

from ovos_plugin_manager.templates.agents import ChatEngine, ReRankerEngine

from ovos_MoS.agents import (
    KingMoSEngine, DemocracyMoSEngine, DuopolyMoSEngine,
    TournamentMoSEngine, CascadeMoSEngine, JuryMoSEngine,
    ChainMoSEngine, CommitteeMoSEngine,
)


def democracy_of_kings(
    king_configs: List[Dict[str, Any]],
    voters: List[ReRankerEngine],
    president: Optional[Union[ReRankerEngine, ChatEngine]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> DemocracyMoSEngine:
    """Build a Democracy where each worker is itself a King MoS.

    Args:
        king_configs: List of dicts, each with ``king`` and ``workers`` keys.
            Example: [{"king": reranker1, "workers": [chat1, chat2]}, ...]
        voters: ReRankerEngines that vote on the King outputs.
        president: Optional tie-breaking president.
        config: Config passed to the outer DemocracyMoSEngine.

    Returns:
        A DemocracyMoSEngine whose workers are KingMoSEngines.
    """
    kings = []
    for kc in king_configs:
        kings.append(KingMoSEngine(
            king=kc["king"],
            workers=kc["workers"],
            config=kc.get("config"),
        ))
    return DemocracyMoSEngine(
        config=config, voters=voters, workers=kings, president=president,
    )


def cascade_then_committee(
    cascade_workers: List[ChatEngine],
    cascade_scorer: ReRankerEngine,
    committee_workers: List[ChatEngine],
    committee_president: Optional[Union[ReRankerEngine, ChatEngine]] = None,
    cascade_config: Optional[Dict[str, Any]] = None,
    committee_config: Optional[Dict[str, Any]] = None,
) -> CommitteeMoSEngine:
    """Build a Committee where one worker is a Cascade (cheap-first).

    The Cascade tries cheap workers first and stops early on high confidence.
    The Committee then uses the Cascade alongside other workers for
    multi-round convergence.

    Args:
        cascade_workers: Workers for the inner Cascade (ordered cheap→expensive).
        cascade_scorer: ReRanker for the Cascade's early stopping.
        committee_workers: Other workers for the Committee.
        committee_president: Optional president for final selection.
        cascade_config: Config for the inner CascadeMoSEngine.
        committee_config: Config for the outer CommitteeMoSEngine.

    Returns:
        A CommitteeMoSEngine with a CascadeMoSEngine as one of its workers.
    """
    cascade = CascadeMoSEngine(
        config=cascade_config, scorer=cascade_scorer, workers=cascade_workers,
    )
    all_workers = [cascade] + list(committee_workers)
    return CommitteeMoSEngine(
        config=committee_config, workers=all_workers,
        president=committee_president,
    )


def chain_with_jury(
    chain_workers: List[ChatEngine],
    jurors: List[ReRankerEngine],
    juror_weights: Optional[List[float]] = None,
    chain_config: Optional[Dict[str, Any]] = None,
    jury_config: Optional[Dict[str, Any]] = None,
) -> JuryMoSEngine:
    """Build a Jury where one worker is a Chain (sequential refinement).

    The Chain progressively refines an answer through multiple workers.
    The Jury then compares the Chain's output against other workers using
    weighted voting.

    Args:
        chain_workers: Workers for the inner Chain (drafter→refiner→polisher).
        jurors: ReRankerEngines for weighted voting.
        juror_weights: Weight per juror (default 1.0 each).
        chain_config: Config for the inner ChainMoSEngine.
        jury_config: Config for the outer JuryMoSEngine.

    Returns:
        A JuryMoSEngine with a ChainMoSEngine as one of its workers.
    """
    chain = ChainMoSEngine(config=chain_config, workers=chain_workers)
    return JuryMoSEngine(
        config=jury_config, jurors=jurors, juror_weights=juror_weights,
        workers=[chain],
    )


def tournament_of_committees(
    committee_configs: List[Dict[str, Any]],
    referee: ReRankerEngine,
    config: Optional[Dict[str, Any]] = None,
) -> TournamentMoSEngine:
    """Build a Tournament where each worker is itself a Committee.

    Args:
        committee_configs: List of dicts, each with ``workers`` and optionally
            ``president`` and ``config`` keys.
        referee: ReRankerEngine for bracket elimination.
        config: Config for the outer TournamentMoSEngine.

    Returns:
        A TournamentMoSEngine whose workers are CommitteeMoSEngines.
    """
    committees = []
    for cc in committee_configs:
        committees.append(CommitteeMoSEngine(
            workers=cc["workers"],
            president=cc.get("president"),
            config=cc.get("config"),
        ))
    return TournamentMoSEngine(
        config=config, referee=referee, workers=committees,
    )


def duopoly_with_cascade_workers(
    president: Union[ReRankerEngine, ChatEngine],
    founders: List[ChatEngine],
    cascade_workers: List[ChatEngine],
    cascade_scorer: ReRankerEngine,
    cascade_config: Optional[Dict[str, Any]] = None,
    duopoly_config: Optional[Dict[str, Any]] = None,
) -> DuopolyMoSEngine:
    """Build a Duopoly where the workers include a Cascade for cost-efficient
    initial answers.

    Args:
        president: Decision maker for the Duopoly.
        founders: Founders for multi-round discussion.
        cascade_workers: Workers for the inner Cascade.
        cascade_scorer: Scorer for the Cascade's early stopping.
        cascade_config: Config for the inner CascadeMoSEngine.
        duopoly_config: Config for the outer DuopolyMoSEngine.

    Returns:
        A DuopolyMoSEngine with a CascadeMoSEngine as one of its workers.
    """
    cascade = CascadeMoSEngine(
        config=cascade_config, scorer=cascade_scorer, workers=cascade_workers,
    )
    return DuopolyMoSEngine(
        config=duopoly_config, president=president,
        founders=founders, workers=[cascade],
    )
