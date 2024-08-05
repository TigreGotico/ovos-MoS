import abc
from typing import Optional, List, Dict, Any

from ovos_gguf_solver import GGUFSolver
from ovos_utils.log import LOG

from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_plugin_manager.templates.solvers import AbstractSolver, MultipleChoiceSolver, QuestionSolver


class AbstractMoS(QuestionSolver):
    def __init__(self, workers: List[QuestionSolver],
                 config: Optional[Dict[str, Any]] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs) -> None:
        """
        Initialize the Mixture Of Solvers.

        Args:
            workers (List[QuestionSolver]): List of solvers providing intermediate answers.
            config (Optional[Dict[str, Any]]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        super().__init__(config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self.workers = workers

    @abc.abstractmethod
    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> str:
        """
        Obtain the spoken answer for a given query.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The spoken answer as a text response.
        """
        raise NotImplementedError

    def gather_responses(self, query: str,
                         lang: Optional[str] = None,
                         units: Optional[str] = None) -> List[str]:
        """
        Consult the QuestionSolver workers and gather their responses.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            List[str]: A list of responses from the workers.
        """
        answers = []
        for solver in self.workers:
            try:
                answer = solver.get_spoken_answer(query, lang=lang, units=units)
                if answer:
                    answers.append(answer)
            except Exception as e:
                LOG.error(f"Error from solver {solver}: {e}")
        if not answers:
            LOG.warning("No answers gathered from workers.")
        return answers


class AbstractKingMoS(AbstractMoS):
    def __init__(self, king: AbstractSolver,
                 workers: List[QuestionSolver],
                 config: Optional[Dict[str, Any]] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs) -> None:
        """
        Initialize the Mixture Of Solvers.

        Args:
            king (AbstractSolver): Solver that selects the final answer.
            workers (List[QuestionSolver]): List of solvers providing intermediate answers.
            config (Optional[Dict[str, Any]]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        super().__init__(workers, config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self.king = king

    @abc.abstractmethod
    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> str:
        """
        Obtain the spoken answer for a given query.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The spoken answer as a text response.
        """
        raise NotImplementedError


class AbstractDuopolyMoS(AbstractMoS):
    def __init__(self, president: AbstractSolver,
                 founders: List[QuestionSolver],
                 workers: Optional[List[QuestionSolver]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs) -> None:
        """
        Initialize the Duopoly Mixture Of Solvers.

        Args:
            president (MultipleChoiceSolver): Solver that chooses the final answer.
            founders (List[AbstractSolver]): Solvers that provide intermediate discussions.
            workers (List[QuestionSolver]): Solvers that provide initial answers for consideration.
            config (Optional[Dict[str, Any]]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        workers = workers or self.founders
        super().__init__(workers, config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self.founders = founders
        self.president = president

    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> str:
        """
        Obtain the spoken answer for a given query.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The spoken answer as a text response.
        """
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            return "No answer could be gathered from workers."

        final_answer = self.discuss_answers(query, answers, lang=lang, units=units)
        return final_answer

    @abc.abstractmethod
    def discuss_answers(self, query: str, answers: List[str],
                        lang: Optional[str] = None,
                        units: Optional[str] = None) -> str:
        """
        The founders discuss the gathered answers and refine them.

        Args:
            query (str): The query text.
            answers (List[str]): The list of answers to discuss.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The refined answer after discussion.
        """
        raise NotImplementedError


#########################
## Vote based MoS
class DemocracyMoS(AbstractMoS):
    def __init__(self, voters: List[MultipleChoiceSolver],
                 workers: List[QuestionSolver],
                 config: Optional[Dict[str, Any]] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs) -> None:
        """
        Initialize the Democracy Mixture Of Solvers.

        Args:
            voters (List[MultipleChoiceSolver]): Solvers that vote on the best answer.
            workers (List[QuestionSolver]): Solvers that provide initial answers for consideration.
            config (Optional[Dict[str, Any]]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        super().__init__(workers, config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self.voters = voters

    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> str:
        """
        Obtain the spoken answer for a given query.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The spoken answer as a text response.
        """
        answers = self.gather_responses(query, lang=lang, units=units)
        if not answers:
            return "No answer could be gathered from workers."
        final_answer = self.vote_on_answers(query, answers, lang=lang)
        return final_answer

    def gather_votes(self, query: str, answers: List[str],
                     lang: Optional[str] = None) -> Dict[str, int]:
        count = {}
        for voter in self.voters:
            ans = voter.select_answer(query, answers, lang=lang)
            if ans not in count:
                count[ans] = 1
            else:
                count[ans] += 1
        return count

    def vote_on_answers(self, query: str, answers: List[str],
                        lang: Optional[str] = None) -> str:
        """
        The voters vote on the gathered answers to select best

        Args:
            query (str): The query text.
            answers (List[str]): The list of answers to vote on
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The refined answer after discussion.
        """
        count = self.gather_votes(query, answers, lang=lang)
        return max(count, key=lambda k: count[k])


##########################
## ReRanker based MoS

class ReRankerKingMoS(AbstractKingMoS):
    def __init__(self, king: MultipleChoiceSolver,
                 workers: List[QuestionSolver],
                 config: Optional[Dict[str, Any]] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs) -> None:
        """
        Initialize the ReRanker King MoS.

        Args:
            king (MultipleChoiceSolver): A ReRanker plugin to be used for final answer selection.
            workers (List[QuestionSolver]): List of solvers providing intermediate answers.
            config (Optional[Dict[str, Any]]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        super().__init__(king, workers, config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)

    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> str:
        """
        Consult the QuestionSolver workers and use a ReRanker to select the final answer.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The spoken answer as a text response.
        """
        answers = self.gather_responses(query, lang=lang, units=units)
        best = None
        assert isinstance(self.king, MultipleChoiceSolver)
        for score, ans in self.king.rerank(query, answers, lang=lang):
            LOG.debug(f"ReRanker score: {score} - {ans}")
            if not best:
                best = ans
        return best


class ReRankerDemocracyMoS(DemocracyMoS):
    def __init__(self, president: MultipleChoiceSolver,
                 voters: List[MultipleChoiceSolver],
                 workers: List[QuestionSolver],
                 config: Optional[Dict] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs):
        """
        Initialize the Democracy Mixture Of Solvers.

        Args:
            president (MultipleChoiceSolver)
            voters (List[MultipleChoiceSolver]): vote on best answer
            workers (List[QuestionSolver]): provide initial answers for consideration
            config (Optional[Dict]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        self.president = president
        super().__init__(voters, workers, config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)

    def vote_on_answers(self, query: str, answers: List[str],
                        lang: Optional[str] = None) -> str:
        """
        Votes are gathered to filter possible answers,
        then a reranker is used to select final answer

        Args:
            query (str): The query text.
            answers (List[str]): The list of answers to vote on
            lang (Optional[str]): Optional language code. Defaults to None.
        Returns:
            str: The refined answer after discussion.
        """
        ans = list(set(self.gather_votes(query, answers, lang=lang)))
        best = None
        for score, ans in self.president.rerank(query, list(set(ans)), lang=lang):
            LOG.debug(f"ReRanker score: {score} - {ans}")
            if not best:
                best = ans
        return best


class ReRankerDuopolyMoS(AbstractDuopolyMoS):
    def __init__(self, president: MultipleChoiceSolver,
                 founders: List[QuestionSolver],
                 workers: Optional[List[QuestionSolver]] = None,
                 config: Optional[Dict] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs):
        """
        Initialize the Duopoly Mixture Of Solvers.

        Args:
            president: MultipleChoiceSolver: choose final answer
            founders (List[GenerativeSolver]): provide intermediate discussions
            workers (List[QuestionSolver]): provide initial answers for consideration
            config (Optional[Dict]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        workers = workers or founders
        super().__init__(president, founders, workers, config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self.discuss_prompt = self.config.get("discuss_prompt",
                                              "given a natural language query and potential answers, your task is to discuss the responses, improving them and correcting any flaws")
        self.system = self.config.get("system_prompt",
                                      "given a natural language query and a discussion about it, your task is to generate a final answer, it needs to be short, factual and conversational")
        self.prompt = self.config.get("prompt_template",
                                      "{system}\nquery: {query}\n\nresponses:{ans}\n\ndiscussion:{discussion}")

    def discuss_answers(self, query: str, answers: List[str],
                        lang: Optional[str] = None,
                        units: Optional[str] = None) -> str:
        """
        The founders discuss the gathered answers and refine them

        Args:
            query (str): The query text.
            answers (List[str]): The list of answers to discuss.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The refined answer after discussion.
        """
        answers = self.gather_responses(query, lang=lang, units=units)
        # discuss
        discussion = []
        for i in range(self.config.get("discussion_rounds", 3)):
            for founder in self.founders:
                assert isinstance(founder, QuestionSolver)
                prompt = self.prompt.format(system=self.discuss_prompt, query=query,
                                            ans='\n-'.join(answers),
                                            discussion='\n-'.join(discussion))
                ans = founder.get_spoken_answer(prompt, lang=lang, units=units)
                LOG.debug(f"founder {founder} says: {ans}")
                discussion.append(ans)

        # select final answer
        prompt = f"{self.system}\n\nDiscussion:\n" + "\n".join(discussion)
        # generate final answer
        answers = []
        for founder in self.founders:
            assert isinstance(founder, QuestionSolver)
            ans = founder.get_spoken_answer(prompt, lang=lang, units=units)
            answers.append(ans)
            LOG.debug(f"founder {founder} says: {ans}")
        return self.president.select_answer(query, lang=lang)


##########################
## LLM based MoS

class GenerativeKingMoS(AbstractKingMoS):
    def __init__(self, king: QuestionSolver,
                 workers: List[QuestionSolver],
                 config: Optional[Dict[str, Any]] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs) -> None:
        """
        Initialize the GenerativeKing MoS.

        Args:
            king (QuestionSolver): The main solver that generates the final answer.
            workers (List[QuestionSolver]): List of solvers providing intermediate answers.
            config (Optional[Dict[str, Any]]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        super().__init__(king, workers, config, translator, detector, priority, enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self.system = self.config.get("system_prompt",
                                      "given a natural language query and search results, your task is to write a short and factual conversational response to the query")
        self.prompt = self.config.get("prompt_template", "{system}\nquery: {query}\n\nsearch results:{ans}")

    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> Optional[str]:
        """
        Consult the QuestionSolver workers and use a ReRanker to select the final answer.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            Optional[str]: The spoken answer as a text response.
        """
        answers = self.gather_responses(query, lang=lang, units=units)
        assert isinstance(self.king, QuestionSolver)
        prompt = self.prompt.format(system=self.system, query=query, ans='\n-'.join(answers))
        return self.king.get_spoken_answer(prompt, lang=lang, units=units)


class GenerativeDuopolyMoS(AbstractDuopolyMoS):
    def __init__(self, president: QuestionSolver,
                 founders: List[QuestionSolver],
                 workers: Optional[List[QuestionSolver]] = None,
                 config: Optional[Dict] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs):
        """
        Initialize the Duopoly Mixture Of Solvers.

        Args:
            president: QuestionSolver: choose final answer
            founders (List[QuestionSolver]): provide intermediate discussions
            workers (List[QuestionSolver]): provide initial answers for consideration
            config (Optional[Dict]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        workers = workers or founders
        super().__init__(president, founders, workers, config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self.discuss_prompt = self.config.get("discuss_prompt",
                                              "given a natural language query and potential answers, your task is to discuss the responses, improving them and correcting any flaws")
        self.system = self.config.get("system_prompt",
                                      "given a natural language query and a discussion about it, your task is to generate a final answer, it needs to be short, factual and conversational")
        self.prompt = self.config.get("prompt_template",
                                      "{system}\nquery: {query}\n\nresponses:{ans}\n\ndiscussion:{discussion}")

    def discuss_answers(self, query: str, answers: List[str],
                        lang: Optional[str] = None,
                        units: Optional[str] = None) -> str:
        """
        The founders discuss the gathered answers and refine them

        Args:
            query (str): The query text.
            answers (List[str]): The list of answers to discuss.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The refined answer after discussion.
        """
        answers = self.gather_responses(query, lang=lang, units=units)
        # discuss
        discussion = []
        for i in range(self.config.get("discussion_rounds", 3)):
            for founder in self.founders:
                assert isinstance(founder, QuestionSolver)
                prompt = self.prompt.format(system=self.discuss_prompt, query=query,
                                            ans='\n-'.join(answers),
                                            discussion='\n-'.join(discussion))
                ans = founder.get_spoken_answer(prompt, lang=lang, units=units)
                LOG.debug(f"founder {founder} says: {ans}")
                discussion.append(ans)

        # select final answer
        prompt = f"{self.system}\n\nDiscussion:\n" + "\n".join(discussion)
        return self.president.get_spoken_answer(prompt, lang=lang)


class GenerativeDemocracyMoS(DemocracyMoS):
    def __init__(self, president: QuestionSolver,
                 voters: List[MultipleChoiceSolver],
                 workers: List[QuestionSolver],
                 config: Optional[Dict] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs):
        """
        Initialize the Democracy Mixture Of Solvers.

        Args:
            voters (List[QuestionSolver]): vote on best answer
            workers (List[QuestionSolver]): provide initial answers for consideration
            config (Optional[Dict]): Optional configuration dictionary.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority of the solver.
            enable_tx (bool): Flag to enable translation.
            enable_cache (bool): Flag to enable caching.
            internal_lang (Optional[str]): Internal language code. Defaults to None.
        """
        self.president = president
        super().__init__(voters, workers, config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self.system = self.config.get("system_prompt",
                                      "given a natural language query and search results, your task is to write a short and factual conversational response to the query")
        self.prompt = self.config.get("prompt_template", "{system}\nquery: {query}\n\nsearch results:{ans}")

    def vote_on_answers(self, query: str, answers: List[str],
                        lang: Optional[str] = None) -> str:
        """
        Votes are gathered to filter possible answers,
        then a reranker is used to select final answer

        Args:
            query (str): The query text.
            answers (List[str]): The list of answers to vote on
            lang (Optional[str]): Optional language code. Defaults to None.
        Returns:
            str: The refined answer after discussion.
        """
        ans = list(set(self.gather_votes(query, answers, lang=lang)))
        prompt = self.prompt.format(system=self.system, query=query, ans='\n-'.join(ans))
        return self.president.get_spoken_answer(prompt, lang=lang)
