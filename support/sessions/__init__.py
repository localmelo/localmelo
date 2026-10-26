"""Sessions: conversations kept in SQLite, with turns that outlive a request.

Nothing outside the surfaces that hold sessions (a terminal ``melo``, the
gateway's console) imports this package: a direct run and every request of
the gateway's ``/v1`` API run as before.

- :mod:`~localmelo.support.sessions.store`: :class:`SessionStore`, the
  SQLite store of sessions, turns, submissions and events;
- :mod:`~localmelo.support.sessions.submissions`: client request ids and
  their rules;
- :mod:`~localmelo.support.sessions.window`: the window checkpoint codec.
"""
