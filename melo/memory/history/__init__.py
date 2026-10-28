from __future__ import annotations

from localmelo.melo.schema import StepRecord, TaskRecord


class History:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    async def save_task(self, task: TaskRecord) -> None:
        self._tasks[task.task_id] = task

    async def get_task(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    async def add_step(self, task_id: str, step: StepRecord) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.steps.append(step)

    async def get_steps(self, task_id: str) -> list[StepRecord]:
        task = self._tasks.get(task_id)
        return task.steps if task else []

    async def forget(self, task_id: str) -> None:
        """Forget one task and its steps; an unknown *task_id* is no error.

        This removes the task's step records with it. Long-term memory is a
        separate store: ``Hippo.forget`` forgets both.
        """
        self._tasks.pop(task_id, None)

    async def forget_all(self) -> None:
        """Forget every task and step.

        Long-term memory is a separate store: ``Hippo.forget_all`` forgets
        both.
        """
        self._tasks.clear()
