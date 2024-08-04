# MoS - Mixture Of Solvers

Using [OpenVoiceOS solver plugins](https://openvoiceos.github.io/ovos-technical-manual/solvers), we implement three
strategies to combine solvers before deciding the final answer.

A solver may be [an LLM](https://github.com/OpenVoiceOS/ovos-solver-plugin-openai-persona),
a [HiveMind connection](https://github.com/JarbasHiveMind/ovos-solver-hivemind-plugin/),
or [any other chatbot](https://openvoiceos.github.io/ovos-technical-manual/persona_server).

Each MoS strategy is also implemented as an individual solver plugin and can be used with any application built around
OVOS solvers.

![img.png](img.png)

> NOTE: MoS can be used recursively. You can use a full MoS in place of any individual solver from this scheme, such as
> a Democracy of Kings or a Duopoly of Democracies.

## MoS Strategies

### The King

![img_6.png](img_6.png)

##### ReRankerKingMoS

`ReRankerKingMoS` uses a re-ranker to select the best answer from the intermediate responses provided by the worker
solvers.

```python
class ReRankerKingMoS(AbstractKingMos):
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
```


##### GGUFKingMoS

`GGUFKingMoS` uses a GGUFSolver as the king solver to generate the final answer based on the intermediate responses.

```python
class GGUFKingMoS(AbstractKingMos):
    def __init__(self, king: GGUFSolver,
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
        Initialize the GGUFKing MoS.

        Args:
            king (GGUFSolver): The main solver that generates the final answer.
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
        assert isinstance(self.king, GGUFSolver)
        prompt = self.prompt.format(system=self.system, query=query, ans='\n-'.join(answers))
        return self.king.get_spoken_answer(prompt, lang=lang, units=units)
```


### Duopoly

![img_3.png](img_3.png)

`DuopolyMos` introduces a pair of "founder" solvers that discuss and refine the intermediate answers provided by the
worker solvers.
A "president" solver (reranker) then selects the final answer based on this discussion.

##### DuopolyGGUFMos

`DuopolyGGUFMos` uses GGUFSolvers as the founders to discuss and refine the intermediate answers before the president
solver selects the final answer.

```python
class DuopolyGGUFMos(AbstractDuopolyMos):
    def __init__(self, president: MultipleChoiceSolver,
                 founders: List[GGUFSolver],
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
            founders (List[GGUFSolver]): provide intermediate discussions
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
                assert isinstance(founder, GGUFSolver)
                prompt = self.prompt.format(system=self.discuss_prompt, query=query,
                                            ans='\n-'.join(answers),
                                            discussion='\n-'.join(discussion))
                ans = founder.get_spoken_answer(prompt, lang=lang, units=units)
                LOG.debug(f"founder {founder} says: {ans}")
                discussion.append(ans)

        # generate final answer
        answers = []
        prompt = f"{self.system}\n\nDiscussion:\n" + "\n".join(discussion)
        for founder in self.founders:
            ans = founder.get_spoken_answer(prompt, lang=lang, units=units)
            answers.append(ans)
            LOG.debug(f"founder {founder} says: {ans}")

        # select final answer
        return self.president.select_answer(query, lang=lang)
```

### Democracy

![img_4.png](img_4.png)

##### DemocracyMos

`DemocracyMos` introduces a set of "voter" solvers (rerankers) that vote on the intermediate answers provided by the
worker solvers.
The answer with the most votes is selected as the final answer.

```python
class DemocracyMos(AbstractMos):
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
        count = {}
        for voter in self.voters:
            ans = voter.select_answer(query, answers, lang=lang)
            if ans not in count:
                count[ans] = 1
            else:
                count[ans] += 1
        return max(count, key=lambda k: count[k])
```


##### ReRankerDemocracyMos

`ReRankerDemocracyMos` replaces vote counting with a re-ranker to select the best final answer.

```python
class ReRankerDemocracyMos(DemocracyMos):
    def __init__(self, reranker: MultipleChoiceSolver,
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
        self.reranker = reranker
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
        ans = []
        for voter in self.voters:
            ans.append(voter.select_answer(query, answers, lang=lang))

        best = None
        for score, ans in self.reranker.rerank(query, list(set(ans)), lang=lang):
            LOG.debug(f"ReRanker score: {score} - {ans}")
            if not best:
                best = ans
        return best
```
