"""localmelo package.

Top-level re-exports for convenience. The canonical implementations live in:
- ``localmelo.melo`` — core agent runtime
- ``localmelo.support`` — infrastructure (providers, gateway, serving, config)
"""

from localmelo.melo import Agent, Message, ToolCall, ToolDef, ToolResult
from localmelo.melo.agent import AgentOptions
from localmelo.melo.schema import RunResult

__all__ = [
    "Agent",
    "AgentOptions",
    "Message",
    "RunResult",
    "ToolCall",
    "ToolDef",
    "ToolResult",
]
