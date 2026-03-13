"""Modern ChatEngine-based Mixture of Solvers classes.

These classes use the OPM agent API (ChatEngine / ReRankerEngine) instead of
the deprecated QuestionSolver / MultipleChoiceSolver hierarchy.
"""
import abc
import math
from typing import Optional, List, Dict, Any, Union

from ovos_utils.log import LOG

from ovos_plugin_manager.templates.agents import (
    ChatEngine, ReRankerEngine, AgentMessage, MessageRole
)
from ovos_MoS._concurrent import gather_concurrent
from ovos_MoS._streaming import (
    StreamingKingMixin, StreamingChainMixin,
    StreamingDuopolyMixin, StreamingCommitteeMixin,
)


class AbstractMoSEngine(ChatEngine):
    """Base class for all modern MoS engines.

    Gathers responses from worker ChatEngines concurrently and delegates
    final answer selection to subclasses.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 workers: Optional[List[ChatEngine]] = None):
        super().__init__(config)
        self.workers = workers or []
        self._max_workers = self.config.get("max_workers", len(self.workers))
        self._worker_timeout = self.config.get("worker_timeout", 30.0)

    def gather_responses(self, query: str,
                         lang: Optional[str] = None,
                         units: Optional[str] = None) -> List[str]:
        """Query all workers concurrently and return their responses."""
        return gather_concurrent(
            workers=self.workers,
            fn=lambda w: w.get_response(query, lang=lang, units=units),
            query=query,
            lang=lang,
            units=units,
            max_workers=self._max_workers,
            timeout=self._worker_timeout,
        )

    @abc.abstractmethod
    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        raise NotImplementedError


class KingMoSEngine(StreamingKingMixin, AbstractMoSEngine):
    """King MoS: workers provide answers, a king selects or generates the final one.

    If the king is a ReRankerEngine, the best answer is selected via reranking.
    If the king is a ChatEngine, a generative prompt synthesizes the final answer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 king: Optional[Union[ReRankerEngine, ChatEngine]] = None,
                 workers: Optional[List[ChatEngine]] = None):
        super().__init__(config, workers)
        self.king = king
        self._system = self.config.get(
            "system_prompt",
            "given a natural language query and search results, your task is "
            "to write a short and factual conversational response to the query"
        )
        self._prompt_template = self.config.get(
            "prompt_template",
            "{system}\nquery: {query}\n\nsearch results:{ans}"
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        query = messages[-1].content if messages else ""
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        if isinstance(self.king, ReRankerEngine):
            ranked = self.king.rerank(query, answers, lang=lang)
            best = ranked[0][1] if ranked else answers[0]
            return AgentMessage(role=MessageRole.ASSISTANT, content=best)

        # Generative mode — king is a ChatEngine
        prompt = self._prompt_template.format(
            system=self._system, query=query,
            ans="\n-".join(answers)
        )
        gen_messages = [AgentMessage(role=MessageRole.USER, content=prompt)]
        return self.king.continue_chat(gen_messages, session_id=session_id,
                                       lang=lang, units=units)


class DemocracyMoSEngine(AbstractMoSEngine):
    """Democracy MoS: workers provide answers, voters vote, majority wins.

    An optional president (ReRankerEngine or ChatEngine) breaks ties or
    refines the voted answers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 voters: Optional[List[ReRankerEngine]] = None,
                 workers: Optional[List[ChatEngine]] = None,
                 president: Optional[Union[ReRankerEngine, ChatEngine]] = None):
        super().__init__(config, workers)
        self.voters = voters or []
        self.president = president
        self._system = self.config.get(
            "system_prompt",
            "given a natural language query and search results, your task is "
            "to write a short and factual conversational response to the query"
        )
        self._prompt_template = self.config.get(
            "prompt_template",
            "{system}\nquery: {query}\n\nsearch results:{ans}"
        )

    def gather_votes(self, query: str, answers: List[str],
                     lang: Optional[str] = None) -> Dict[str, int]:
        """Each voter selects the best answer; tally the votes."""
        count: Dict[str, int] = {}
        for voter in self.voters:
            try:
                ans = voter.select_answer(query, answers, lang=lang)
                count[ans] = count.get(ans, 0) + 1
            except Exception as e:
                LOG.error(f"Voter {voter} failed: {e}")
        return count

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        query = messages[-1].content if messages else ""
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        votes = self.gather_votes(query, answers, lang=lang)
        if not votes:
            winner = answers[0]
        else:
            winner = max(votes, key=lambda k: votes[k])

        if self.president is None:
            return AgentMessage(role=MessageRole.ASSISTANT, content=winner)

        # President refines or reranks the voted answers
        voted_answers = list(votes.keys()) if votes else answers
        if isinstance(self.president, ReRankerEngine):
            ranked = self.president.rerank(query, voted_answers, lang=lang)
            best = ranked[0][1] if ranked else winner
            return AgentMessage(role=MessageRole.ASSISTANT, content=best)

        # Generative president
        prompt = self._prompt_template.format(
            system=self._system, query=query,
            ans="\n-".join(voted_answers)
        )
        gen_messages = [AgentMessage(role=MessageRole.USER, content=prompt)]
        return self.president.continue_chat(gen_messages, session_id=session_id,
                                            lang=lang, units=units)


class DuopolyMoSEngine(StreamingDuopolyMixin, AbstractMoSEngine):
    """Duopoly MoS: founders discuss worker answers, president decides.

    Workers provide initial answers. Founders engage in multi-round discussion
    to refine them. The president then selects (rerank) or generates the final
    answer based on the discussion.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 president: Optional[Union[ReRankerEngine, ChatEngine]] = None,
                 founders: Optional[List[ChatEngine]] = None,
                 workers: Optional[List[ChatEngine]] = None):
        workers = workers or founders or []
        super().__init__(config, workers)
        self.founders = founders or []
        self.president = president
        self._discussion_rounds = self.config.get("discussion_rounds", 3)
        self._discuss_prompt = self.config.get(
            "discuss_prompt",
            "given a natural language query and potential answers, your task is "
            "to discuss the responses, improving them and correcting any flaws"
        )
        self._system = self.config.get(
            "system_prompt",
            "given a natural language query and a discussion about it, your "
            "task is to generate a final answer, it needs to be short, "
            "factual and conversational"
        )
        self._prompt_template = self.config.get(
            "prompt_template",
            "{system}\nquery: {query}\n\nresponses:{ans}\n\ndiscussion:{discussion}"
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        query = messages[-1].content if messages else ""
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        # Multi-round founder discussion
        discussion: List[str] = []
        for _round in range(self._discussion_rounds):
            for founder in self.founders:
                prompt = self._prompt_template.format(
                    system=self._discuss_prompt, query=query,
                    ans="\n-".join(answers),
                    discussion="\n-".join(discussion)
                )
                try:
                    resp = founder.get_response(prompt, lang=lang, units=units)
                    if resp:
                        discussion.append(resp)
                        LOG.debug(f"Founder {founder} says: {resp}")
                except Exception as e:
                    LOG.error(f"Founder {founder} discussion failed: {e}")

        if not discussion:
            return AgentMessage(role=MessageRole.ASSISTANT, content=answers[0])

        # President decides based on discussion
        if isinstance(self.president, ReRankerEngine):
            # Founders generate final candidates
            final_prompt = (f"{self._system}\n\nDiscussion:\n"
                            + "\n".join(discussion))
            candidates = []
            for founder in self.founders:
                try:
                    resp = founder.get_response(final_prompt, lang=lang, units=units)
                    if resp:
                        candidates.append(resp)
                except Exception as e:
                    LOG.error(f"Founder {founder} final answer failed: {e}")

            if not candidates:
                candidates = discussion[-len(self.founders):]

            ranked = self.president.rerank(query, candidates, lang=lang)
            best = ranked[0][1] if ranked else candidates[0]
            return AgentMessage(role=MessageRole.ASSISTANT, content=best)

        # Generative president
        final_prompt = (f"{self._system}\n\nDiscussion:\n"
                        + "\n".join(discussion))
        gen_messages = [AgentMessage(role=MessageRole.USER, content=final_prompt)]
        return self.president.continue_chat(gen_messages, session_id=session_id,
                                            lang=lang, units=units)


class TournamentMoSEngine(AbstractMoSEngine):
    """Tournament MoS: bracket-style elimination via ReRanker.

    Workers provide answers, then pairs compete head-to-head via a ReRanker.
    Winners advance to the next round until one remains.
    If the number of answers is odd, the last one gets a bye.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 referee: Optional[ReRankerEngine] = None,
                 workers: Optional[List[ChatEngine]] = None):
        super().__init__(config, workers)
        self.referee = referee

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        query = messages[-1].content if messages else ""
        candidates = self.gather_responses(query, lang=lang, units=units)
        if not candidates:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        # Bracket elimination
        round_num = 0
        while len(candidates) > 1:
            round_num += 1
            next_round = []
            for i in range(0, len(candidates) - 1, 2):
                pair = [candidates[i], candidates[i + 1]]
                try:
                    ranked = self.referee.rerank(query, pair, lang=lang)
                    winner = ranked[0][1] if ranked else pair[0]
                except Exception as e:
                    LOG.error(f"Tournament round {round_num} match failed: {e}")
                    winner = pair[0]
                next_round.append(winner)
            # Bye for odd one out
            if len(candidates) % 2 == 1:
                next_round.append(candidates[-1])
            candidates = next_round
            LOG.debug(f"Tournament round {round_num}: {len(candidates)} remaining")

        return AgentMessage(role=MessageRole.ASSISTANT, content=candidates[0])


class CascadeMoSEngine(ChatEngine):
    """Cascade MoS: sequential querying with early stopping.

    Workers are queried one by one in order. After each worker responds,
    the scorer (ReRankerEngine) evaluates all accumulated answers. If the
    top score exceeds the threshold, stop early and return the best answer.
    Saves cost when the first good answer is often sufficient.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 scorer: Optional[ReRankerEngine] = None,
                 workers: Optional[List[ChatEngine]] = None):
        super().__init__(config)
        self.workers = workers or []
        self.scorer = scorer
        self._threshold = self.config.get("threshold", 0.8)

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        query = messages[-1].content if messages else ""
        accumulated: List[str] = []

        for worker in self.workers:
            try:
                resp = worker.get_response(query, lang=lang, units=units)
                if resp:
                    accumulated.append(resp)
            except Exception as e:
                LOG.error(f"Cascade worker {worker} failed: {e}")
                continue

            if not accumulated:
                continue

            # Score accumulated answers
            try:
                ranked = self.scorer.rerank(query, accumulated, lang=lang)
                if ranked:
                    best_score, best_answer = ranked[0]
                    LOG.debug(f"Cascade: best score {best_score:.3f} "
                              f"after {len(accumulated)} workers")
                    if best_score >= self._threshold:
                        return AgentMessage(role=MessageRole.ASSISTANT,
                                            content=best_answer)
            except Exception as e:
                LOG.error(f"Cascade scorer failed: {e}")

        # Exhausted all workers — return best available or first
        if accumulated:
            try:
                ranked = self.scorer.rerank(query, accumulated, lang=lang)
                if ranked:
                    return AgentMessage(role=MessageRole.ASSISTANT,
                                        content=ranked[0][1])
            except Exception:
                pass
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content=accumulated[0])

        return AgentMessage(role=MessageRole.ASSISTANT,
                            content="No answers could be gathered from workers.")


class JuryMoSEngine(AbstractMoSEngine):
    """Jury MoS: weighted voting.

    Like Democracy but each voter has a weight. The answer with the highest
    weighted vote total wins. Weights can represent model quality, historical
    accuracy, or confidence.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 jurors: Optional[List[ReRankerEngine]] = None,
                 juror_weights: Optional[List[float]] = None,
                 workers: Optional[List[ChatEngine]] = None,
                 president: Optional[Union[ReRankerEngine, ChatEngine]] = None):
        super().__init__(config, workers)
        self.jurors = jurors or []
        self.juror_weights = juror_weights or [1.0] * len(self.jurors)
        self.president = president
        self._system = self.config.get(
            "system_prompt",
            "given a natural language query and search results, your task is "
            "to write a short and factual conversational response to the query"
        )
        self._prompt_template = self.config.get(
            "prompt_template",
            "{system}\nquery: {query}\n\nsearch results:{ans}"
        )

    def gather_weighted_votes(self, query: str, answers: List[str],
                              lang: Optional[str] = None) -> Dict[str, float]:
        """Each juror votes; votes are weighted by juror_weights."""
        scores: Dict[str, float] = {}
        for juror, weight in zip(self.jurors, self.juror_weights):
            try:
                ans = juror.select_answer(query, answers, lang=lang)
                scores[ans] = scores.get(ans, 0.0) + weight
            except Exception as e:
                LOG.error(f"Juror {juror} failed: {e}")
        return scores

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        query = messages[-1].content if messages else ""
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        scores = self.gather_weighted_votes(query, answers, lang=lang)
        if not scores:
            winner = answers[0]
        else:
            winner = max(scores, key=lambda k: scores[k])

        if self.president is None:
            return AgentMessage(role=MessageRole.ASSISTANT, content=winner)

        voted_answers = list(scores.keys()) if scores else answers
        if isinstance(self.president, ReRankerEngine):
            ranked = self.president.rerank(query, voted_answers, lang=lang)
            best = ranked[0][1] if ranked else winner
            return AgentMessage(role=MessageRole.ASSISTANT, content=best)

        prompt = self._prompt_template.format(
            system=self._system, query=query,
            ans="\n-".join(voted_answers)
        )
        gen_messages = [AgentMessage(role=MessageRole.USER, content=prompt)]
        return self.president.continue_chat(gen_messages, session_id=session_id,
                                            lang=lang, units=units)


class ChainMoSEngine(StreamingChainMixin, ChatEngine):
    """Chain MoS: sequential refinement pipeline.

    Each worker refines the previous worker's answer. Worker 1 answers the
    query directly. Worker 2 receives the query + Worker 1's answer and
    improves it. Worker 3 improves Worker 2's output, and so on.
    An optional president selects or synthesizes the final answer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 workers: Optional[List[ChatEngine]] = None,
                 president: Optional[Union[ReRankerEngine, ChatEngine]] = None):
        super().__init__(config)
        self.workers = workers or []
        self.president = president
        self._refinement_prompt = self.config.get(
            "refinement_prompt",
            "Query: {query}\n\nPrevious answer: {previous}\n\n"
            "Please improve and refine the above answer. Fix any errors, "
            "add missing information, and make it more concise and accurate."
        )

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        query = messages[-1].content if messages else ""
        if not self.workers:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        # First worker answers directly
        current_answer = None
        all_versions: List[str] = []
        for i, worker in enumerate(self.workers):
            try:
                if i == 0:
                    resp = worker.get_response(query, lang=lang, units=units)
                else:
                    prompt = self._refinement_prompt.format(
                        query=query, previous=current_answer
                    )
                    resp = worker.get_response(prompt, lang=lang, units=units)
                if resp:
                    current_answer = resp
                    all_versions.append(resp)
                    LOG.debug(f"Chain step {i}: {resp[:80]}...")
            except Exception as e:
                LOG.error(f"Chain worker {i} ({worker}) failed: {e}")

        if not current_answer:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        if self.president is None:
            return AgentMessage(role=MessageRole.ASSISTANT, content=current_answer)

        # President selects from all versions or generates final
        if isinstance(self.president, ReRankerEngine):
            ranked = self.president.rerank(query, all_versions, lang=lang)
            best = ranked[0][1] if ranked else current_answer
            return AgentMessage(role=MessageRole.ASSISTANT, content=best)

        # Generative president sees the full chain
        chain_text = "\n\n".join(
            f"Version {i+1}: {v}" for i, v in enumerate(all_versions)
        )
        prompt = (f"Query: {query}\n\nRefinement chain:\n{chain_text}\n\n"
                  f"Based on this progressive refinement, provide the best "
                  f"final answer.")
        gen_messages = [AgentMessage(role=MessageRole.USER, content=prompt)]
        return self.president.continue_chat(gen_messages, session_id=session_id,
                                            lang=lang, units=units)


class CommitteeMoSEngine(StreamingCommitteeMixin, AbstractMoSEngine):
    """Committee MoS: multi-round convergence.

    All workers answer independently in round 1. In subsequent rounds, each
    worker sees all answers from the previous round and revises. Repeats
    until answers converge or max_rounds is reached. A president then selects
    or generates the final answer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 workers: Optional[List[ChatEngine]] = None,
                 president: Optional[Union[ReRankerEngine, ChatEngine]] = None):
        super().__init__(config, workers)
        self.president = president
        self._max_rounds = self.config.get("max_rounds", 3)
        self._revision_prompt = self.config.get(
            "revision_prompt",
            "Query: {query}\n\nOther committee members answered:\n{others}\n\n"
            "Your previous answer: {previous}\n\n"
            "Revise your answer considering the other perspectives. "
            "Correct errors and incorporate valid points."
        )
        self._system = self.config.get(
            "system_prompt",
            "given a natural language query and committee responses, your task "
            "is to write a short and factual conversational final answer"
        )

    @staticmethod
    def _answers_converged(answers: List[str]) -> bool:
        """Check if all answers are identical (full convergence)."""
        return len(set(answers)) == 1

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        query = messages[-1].content if messages else ""
        if not self.workers:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        # Round 1: independent answers (concurrent)
        current_answers = self.gather_responses(query, lang=lang, units=units)
        if not current_answers:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="No answers could be gathered from workers.")

        # Pad if fewer answers than workers (some failed)
        while len(current_answers) < len(self.workers):
            current_answers.append(current_answers[-1])

        # Subsequent rounds: revision
        for round_num in range(1, self._max_rounds):
            if self._answers_converged(current_answers):
                LOG.debug(f"Committee converged at round {round_num}")
                break

            next_answers = []
            for i, worker in enumerate(self.workers):
                previous = current_answers[i] if i < len(current_answers) else ""
                others = "\n- ".join(
                    a for j, a in enumerate(current_answers) if j != i
                )
                prompt = self._revision_prompt.format(
                    query=query, others=others, previous=previous
                )
                try:
                    resp = worker.get_response(prompt, lang=lang, units=units)
                    next_answers.append(resp if resp else previous)
                except Exception as e:
                    LOG.error(f"Committee worker {i} revision failed: {e}")
                    next_answers.append(previous)

            current_answers = next_answers
            LOG.debug(f"Committee round {round_num + 1}: "
                      f"{len(set(current_answers))} distinct answers")

        # Final selection
        unique_answers = list(set(current_answers))

        if len(unique_answers) == 1:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content=unique_answers[0])

        if self.president is None:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content=current_answers[0])

        if isinstance(self.president, ReRankerEngine):
            ranked = self.president.rerank(query, unique_answers, lang=lang)
            best = ranked[0][1] if ranked else unique_answers[0]
            return AgentMessage(role=MessageRole.ASSISTANT, content=best)

        prompt = (f"{self._system}\nquery: {query}\n\n"
                  f"committee answers:\n- " + "\n- ".join(unique_answers))
        gen_messages = [AgentMessage(role=MessageRole.USER, content=prompt)]
        return self.president.continue_chat(gen_messages, session_id=session_id,
                                            lang=lang, units=units)
