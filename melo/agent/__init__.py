"""Agent: the public agent loop entry point.

``Chat`` is intentionally not exported — it is an internal LLM wrapper
used by ``Agent`` and bypasses checker/memory/executor policy when used
directly. Callers that need direct LLM access should construct their
own provider via ``localmelo.melo.contracts.providers``.
"""

from localmelo.melo.agent.agent import Agent

__all__ = ["Agent"]
