"""A run's own limit on model requests (``AgentOptions.max_calls``).

Internal to :mod:`localmelo.melo.agent`: :class:`~localmelo.melo.agent._chat.Chat`
raises :class:`CallBudgetError` before a request that would pass the limit, and
:meth:`Agent.run_task <localmelo.melo.agent.agent.Agent.run_task>` ends the run
with it. :func:`is_budget_refusal` tells a refusal for a budget or rate limit,
which ``answer_on_stop`` answers, from any other failed request.
"""

from __future__ import annotations

import re

from localmelo.melo.contracts.providers import ProviderError


class CallBudgetError(Exception):
    """The next request would pass the run's ``max_calls``; nothing was sent."""


# The words of a refusal for a used budget ("budget_exhausted: the run has
# used its 30 steps", "budget exceeded") or a rate limit, in the error's text.
_REFUSAL_TEXT = re.compile(
    r"budget[\s_-]*(?:is\s+|was\s+|has\s+been\s+)?(?:exhaust|used|exceed|reach)"
    r"|(?:used|exhausted|exceeded|reached)\s+(?:its|the|your|all)?\s*(?:\w+\s+)?budget"
    r"|rate[\s_-]*limit|too many requests",
    re.IGNORECASE,
)


def is_budget_refusal(exc: ProviderError) -> bool:
    """Whether the server refused a request because a budget or rate limit is used.

    True for HTTP 429, and for an error whose text (it carries the start of the
    response body) says that a budget is used up or a rate limit reached.
    """
    return exc.status_code == 429 or bool(_REFUSAL_TEXT.search(str(exc)))
