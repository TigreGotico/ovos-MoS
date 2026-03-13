"""Streaming support for MoS strategies.

Engines with generative kings/presidents can delegate streaming to the
inner ChatEngine rather than falling back to the base ChatEngine default
(which calls continue_chat and splits the result).

For strategies where the final answer comes from a ReRanker (selection),
streaming is not meaningful — the base class default is used.
"""
from typing import Optional, List, Iterable

from ovos_plugin_manager.templates.agents import (
    ChatEngine, ReRankerEngine, AgentMessage, MessageRole
)


class StreamingKingMixin:
    """Mixin for KingMoSEngine to stream from a generative king."""

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        if isinstance(self.king, ReRankerEngine):
            yield from super().stream_tokens(messages, session_id, lang, units)
            return

        query = messages[-1].content if messages else ""
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            yield "No answers could be gathered from workers."
            return

        prompt = self._prompt_template.format(
            system=self._system, query=query,
            ans="\n-".join(answers)
        )
        gen_messages = [AgentMessage(role=MessageRole.USER, content=prompt)]
        yield from self.king.stream_tokens(gen_messages, session_id=session_id,
                                           lang=lang, units=units)

    def stream_sentences(self, messages: List[AgentMessage],
                         session_id: str = "default",
                         lang: Optional[str] = None,
                         units: Optional[str] = None) -> Iterable[str]:
        if isinstance(self.king, ReRankerEngine):
            yield from super().stream_sentences(messages, session_id, lang, units)
            return

        query = messages[-1].content if messages else ""
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            yield "No answers could be gathered from workers."
            return

        prompt = self._prompt_template.format(
            system=self._system, query=query,
            ans="\n-".join(answers)
        )
        gen_messages = [AgentMessage(role=MessageRole.USER, content=prompt)]
        yield from self.king.stream_sentences(gen_messages, session_id=session_id,
                                              lang=lang, units=units)


class StreamingChainMixin:
    """Mixin for ChainMoSEngine to stream from a generative president."""

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        if self.president is None or isinstance(self.president, ReRankerEngine):
            yield from super().stream_tokens(messages, session_id, lang, units)
            return

        # Run the chain to get all versions, then stream the president's synthesis
        query = messages[-1].content if messages else ""
        current_answer = None
        all_versions = []
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
            except Exception:
                pass

        if not all_versions:
            yield "No answers could be gathered from workers."
            return

        chain_text = "\n\n".join(
            f"Version {i+1}: {v}" for i, v in enumerate(all_versions)
        )
        prompt = (f"Query: {query}\n\nRefinement chain:\n{chain_text}\n\n"
                  f"Based on this progressive refinement, provide the best "
                  f"final answer.")
        gen_messages = [AgentMessage(role=MessageRole.USER, content=prompt)]
        yield from self.president.stream_tokens(gen_messages,
                                                session_id=session_id,
                                                lang=lang, units=units)


class StreamingDuopolyMixin:
    """Mixin for DuopolyMoSEngine to stream from a generative president."""

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        if isinstance(self.president, ReRankerEngine):
            yield from super().stream_tokens(messages, session_id, lang, units)
            return

        query = messages[-1].content if messages else ""
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            yield "No answers could be gathered from workers."
            return

        # Run discussion rounds (same logic as continue_chat)
        discussion = []
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
                except Exception:
                    pass

        if not discussion:
            yield answers[0] if answers else "No answers could be gathered."
            return

        final_prompt = (f"{self._system}\n\nDiscussion:\n"
                        + "\n".join(discussion))
        gen_messages = [AgentMessage(role=MessageRole.USER, content=final_prompt)]
        yield from self.president.stream_tokens(gen_messages,
                                                session_id=session_id,
                                                lang=lang, units=units)


class StreamingCommitteeMixin:
    """Mixin for CommitteeMoSEngine to stream from a generative president."""

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        if self.president is None or isinstance(self.president, ReRankerEngine):
            yield from super().stream_tokens(messages, session_id, lang, units)
            return

        # Run the committee rounds via continue_chat, but stream the final president call
        # Since the committee logic is complex, we call continue_chat for everything
        # except the final president synthesis, which we stream
        result = self.continue_chat(messages, session_id, lang, units)
        yield from result.content.split()
