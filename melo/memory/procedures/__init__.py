"""Procedural memory: step templates learned from the runs of recurring tasks.

A routine is a recurring task (:mod:`.normalize`). Each run's tool calls
get a class, a label and a phase (:mod:`.classify`) and form a trace
(:mod:`.trace`). At a run's end the store (:mod:`.store`) keeps an episode
and, for a completed run, a new version of the routine's template: step
statistics and named steps (:mod:`.template`). The tracker
(:mod:`.tracker`) places a running task on its template, so that a
display can show what is known: facts, past runs and, when the template
is trusted, a bar.

The agent keeps each run's trace, records the run at its end, and, when
its caller asks for progress, reports the tracker's state as
``run.progress`` events. Level 2, the bar, is off for a running task
(:data:`.tracker.LEVEL2_ENABLED`); the gate is still evaluated and its
level recorded.

Nothing here calls a model or changes a request, and no template enters a
prompt. Experimental: the interface may change.
"""
