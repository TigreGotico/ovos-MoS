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

For the choice of final answer, typically a ReRanker is used, but a QuestionSolver can also be used for generative responses

`ReRankerKingMoS` uses a re-ranker to select the best answer from the intermediate responses provided by the worker
solvers.

```python
# Initialize solvers
workers = [QuestionSolver1(), QuestionSolver2(), QuestionSolver3()]
king = ReRankerSolver()

# Create a King MoS instance
mos = ReRankerKingMoS(king, workers)

# Get the answer to a query
query = "What is the speed of light?"
answer = mos.spoken_answer(query)
print(answer)
```

`GenerativeKingMoS` uses a LLM Solver as the king to generate the final answer based on the intermediate responses.

```python
# Initialize solvers
workers = [QuestionSolver1(), QuestionSolver2(), QuestionSolver3()]
cfg = {
    "model": "RichardErkhov/GritLM_-_GritLM-7B-gguf",
    "remote_filename": "*Q4_K_M.gguf",
    "n_gpu_layers": -1
}
king = GGUFSolver(cfg)

# Create a King MoS instance
mos = GenerativeKingMoS(king, workers)

# Get the answer to a query
query = "Explain quantum mechanics in simple terms"
answer = mos.spoken_answer(query)
print(answer)
```


### Democracy

![img_4.png](img_4.png)


`DemocracyMoS` introduces a set of "voter" solvers (rerankers) that vote on the intermediate answers provided by the
worker solvers.
The answer with the most votes is selected as the final answer.

```python
# Initialize solvers
workers = [QuestionSolver1(), QuestionSolver2(), QuestionSolver3()]
voters = [ReRankerSolver1(), ReRankerSolver2(), ReRankerSolver3()]

# Create a Democrcy MoS instance
mos = DemocracyMoS(voters, workers)

# Get the answer to a query
query = "Explain quantum mechanics in simple terms"
answer = mos.spoken_answer(query)
print(answer)
```

`ReRankerDemocracyMoS` voters are used to filter answers with 0 votes, a re-ranker is then used to select the best final answer.

```python
# Initialize solvers
workers = [QuestionSolver1(), QuestionSolver2(), QuestionSolver3()]
voters = [ReRankerSolver1(), ReRankerSolver2(), ReRankerSolver3()]
president = ReRankerSolver()

# Create a Democracy MoS instance
mos = ReRankerDemocracyMoS(president, voters, workers)

# Get the answer to a query
query = "Explain quantum mechanics in simple terms"
answer = mos.spoken_answer(query)
print(answer)
```

`GenerativeDemocracyMoS` replaces re-ranking with a LLM that generates the final answer.

```python
# Initialize solvers
workers = [QuestionSolver1(), QuestionSolver2(), QuestionSolver3()]
voters = [ReRankerSolver1(), ReRankerSolver2(), ReRankerSolver3()]
president = GGUFSolver({
    "model": "RichardErkhov/GritLM_-_GritLM-7B-gguf",
    "remote_filename": "*Q4_K_M.gguf",
    "n_gpu_layers": -1
})

# Create a Democracy MoS instance
mos = GenerativeDemocracyMoS(president, voters, workers)

# Get the answer to a query
query = "Explain quantum mechanics in simple terms"
answer = mos.spoken_answer(query)
print(answer)
```


### Duopoly

![img_3.png](img_3.png)

`ReRankerDuopolyMoS` introduces a pair of "founder" solvers that discuss and refine the intermediate answers provided by the
worker solvers.
A "president" solver (reranker) then selects the final answer based on this discussion.

```python
# Initialize solvers
founders = [QuestionSolver1(), QuestionSolver2()]
president = ReRankerSolver()

# Create a Duopoly MoS instance
mos = ReRankerDuopolyMoS(president, founders)

# Get the answer to a query
query = "Explain quantum mechanics in simple terms"
answer = mos.spoken_answer(query)
print(answer)
```


`GenerativeDuopolyMoS` uses LLMs as the founders to discuss and refine the intermediate answers before the president
generates the final answer.

```python
# Initialize solvers
founder = GGUFSolver({
    "model": "RichardErkhov/GritLM_-_GritLM-7B-gguf",
    "remote_filename": "*Q4_K_M.gguf",
    "n_gpu_layers": -1
})
cofounder = GGUFSolver({
    "model": "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
})

founders = [founder, cofounder]
president = founder

# Create a Duopoly MoS instance
mos = GenerativeDuopolyMoS(president, founders)

# Get the answer to a query
query = "Explain quantum mechanics in simple terms"
answer = mos.spoken_answer(query)
print(answer)
```