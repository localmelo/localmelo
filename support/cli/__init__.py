"""The interactive terminal UI behind bare ``melo``.

:func:`run_interactive` opens it; ``localmelo.__main__`` imports this
package only for bare ``melo``, so direct mode never loads its libraries.
"""

from localmelo.support.cli.launch import run_interactive

__all__ = ["run_interactive"]
