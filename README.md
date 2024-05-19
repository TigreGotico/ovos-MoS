# MoS - Mixture Of Solvers

using [OpenVoiceOS solver plugins](https://openvoiceos.github.io/ovos-technical-manual/solvers) we implement 3 strategies to combine solvers before deciding the final answer

A solver may be [a LLM](https://github.com/OpenVoiceOS/ovos-solver-plugin-openai-persona), a [HiveMind connection](https://github.com/JarbasHiveMind/ovos-solver-hivemind-plugin/), or [any other chatbot](https://openvoiceos.github.io/ovos-technical-manual/persona_server)

Each MoS strategy is also implemented as a individual solver plugin, and can be used with any application built around OVOS solvers

![img.png](img.png)

> NOTE: MoS can be used recursively, you can use a full MoS in place of any individual solver from this scheme. Such as a Democracy of Kings, or a Duopoly of Democracies

![img_6.png](img_6.png)

![img_3.png](img_3.png)

![img_4.png](img_4.png)

