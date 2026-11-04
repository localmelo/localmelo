"""A turn that ended on a blocked step: what it says is no memory (#428).

When Auto stops a step or the approval policy denies one, the model's reply says
what melo's rules were *then* ("this was blocked, ask the user to type a
permission"). Recalled later, once the rules have changed, that reply is a false
precedent that the model may follow against what the person now says. So a turn
that ended on a block is indexed with the person's words only; what they said in
that turn stays part of their history (:mod:`localmelo.support.memory.service`).

A turn *ended on a block* when the last result of its tool calls is one:

- the executor's denial of a call (error category ``approval_denied``: the approval
  policy, the person's Don't allow, a card nobody answered, Auto's stop), or
- a failed call (status ``error`` or category ``tool_error``, as a tool that asks its
  own question returns when the question is refused) whose text starts with one of
  :data:`BLOCK_PREFIXES`, behind the lead such a tool puts first (``melo did not
  read example.com. ``).

A block followed by a call that went on to succeed is the last result no longer: the
turn did not end on it, and its answer is kept. So is a turn without tool calls and
a turn whose last call failed for another reason.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final

from localmelo.melo.contracts.approvals import BLOCK_NOTE, BLOCK_STOP, STOPPED_NOTE
from localmelo.melo.executor.models import ErrorCategory, ExecutionStatus

# What a result is, as :meth:`SessionStore.turn_tool_results` gives it: the
# call's ``status``, its ``error_category``, and the head of its text (the error,
# or the output when there is none).
ToolResult = tuple[str, str, str]


def derive_prefixes(templates: Iterable[str]) -> tuple[str, ...]:
    """The stable text each of *templates* starts with: what comes before its first
    ``{`` and on its first line, stripped. A template that starts with a field has
    none; a prefix is given once."""
    found: list[str] = []
    for template in templates:
        head = template.split("{", 1)[0].split("\n", 1)[0].strip()
        if head and head not in found:
            found.append(head)
    return tuple(found)


# The text of the templates now in ``melo/contracts/approvals.py``, read at import
# so it cannot drift: a note Auto blocked a step with, the note of a second block
# (which stops the turn), and the note of a call after that stop.
CURRENT_PREFIXES: Final = derive_prefixes((BLOCK_NOTE, BLOCK_STOP, STOPPED_NOTE))

# The same three notes' first sentences as every version of them has had them
# since Auto's block was first written, kept as literals so a later rewording of a
# template does not stop an old turn from being recognised. Only the bodies changed
# in between (how the person can allow the step: "say so in the next message", then
# "switch the chat to Ask", then "melo asks on a card; never ask them to type a
# permission"), which is why a reply written about an old note reports a rule that
# no longer holds, and why only the text before the body is matched.
HISTORICAL_PREFIXES: Final = (
    # the block note: "Auto mode blocked this step, so it did not run. {it}..."
    "Auto mode blocked this step, so it did not run.",
    # the note of a second block, which stops the turn
    "Auto mode blocked this step too:",
    # the note of a call after that stop
    "Auto mode stopped this turn after blocking a step, so this step did not run.",
)

BLOCK_PREFIXES: Final = tuple(dict.fromkeys((*CURRENT_PREFIXES, *HISTORICAL_PREFIXES)))

# A tool that asks its own question (a site it opens, a click, typed text) puts its
# own sentence before the refusal's text: "melo did not read example.com. ".
_TOOL_LEAD: Final = re.compile(r"melo did not [^\n]{1,160}?\. ")

# Enough of a result's text for the lead and the longest prefix.
HEAD_CHARS: Final = 400


def starts_with_block(text: str, prefixes: Sequence[str] = BLOCK_PREFIXES) -> bool:
    """Whether *text* is a block's note: one of *prefixes* first, or after the lead of
    a tool whose own question was refused."""
    text = text.lstrip()
    if text.startswith(tuple(prefixes)):
        return True
    lead = _TOOL_LEAD.match(text)
    return lead is not None and text[lead.end() :].startswith(tuple(prefixes))


def is_block(result: ToolResult, prefixes: Sequence[str] = BLOCK_PREFIXES) -> bool:
    """Whether *result* (``(status, error_category, text head)``) is a blocked step."""
    status, category, head = result
    if category == ErrorCategory.APPROVAL_DENIED.value:
        return True
    if status == ExecutionStatus.ERROR.value or category == (
        ErrorCategory.TOOL_ERROR.value
    ):
        return starts_with_block(head, prefixes)
    return False


def ended_on_block(
    results: Sequence[ToolResult], prefixes: Sequence[str] = BLOCK_PREFIXES
) -> bool:
    """Whether a turn whose tool calls ended in *results* (in the order they
    finished) ended on a block: the last one is a block, so no later call succeeded.

    A block in the middle of the turn, followed by a call that ran, is not the end:
    the turn went on and its answer holds what came of it. No tool call at all
    (*results* empty) is no block.
    """
    return bool(results) and is_block(results[-1], prefixes)
