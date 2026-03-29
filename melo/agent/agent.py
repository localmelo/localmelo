from __future__ import annotations

import os
from typing import TYPE_CHECKING

from localmelo.melo.agent.chat import Chat
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import (
    ExecutorResultPayload,
    MemoryWritePayload,
    ToolResolutionResult,
)
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.executor import Executor, register_builtins
from localmelo.melo.executor.models import ExecutionRequest
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import (
    MAX_AGENT_STEPS,
    MIN_AGENT_STEPS,
    Message,
    StepRecord,
    TaskRecord,
    ToolDef,
    ToolResult,
)

if TYPE_CHECKING:
    from localmelo.support.config import Config


def _providers_from_config(
    cfg: Config,
) -> tuple[BaseLLMProvider, BaseEmbeddingProvider | None]:
    """Build LLM and embedding providers from a Config object."""
    from localmelo.support.providers.embedding.openai_compat import (
        OpenAICompatEmbedding,
    )
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    llm: BaseLLMProvider
    embedding: BaseEmbeddingProvider | None

    if cfg.backend == "mlc-llm":
        base_url = f"http://127.0.0.1:{cfg.mlc.chat_port}/v1"
        llm = OpenAICompatLLM(base_url=base_url, model=cfg.mlc.chat_model)
        embedding = OpenAICompatEmbedding(
            base_url=base_url, model=cfg.mlc.embedding_model
        )

    elif cfg.backend == "ollama":
        chat_url = cfg.ollama.chat_url.rstrip("/") + "/v1"
        llm = OpenAICompatLLM(base_url=chat_url, model=cfg.ollama.chat_model)

        if cfg.ollama.embedding_model:
            emb_base = cfg.ollama.embedding_url or cfg.ollama.chat_url
            emb_url = emb_base.rstrip("/") + "/v1"
            embedding = OpenAICompatEmbedding(
                base_url=emb_url, model=cfg.ollama.embedding_model
            )
        else:
            emb_url = f"http://127.0.0.1:{cfg.mlc.chat_port}/v1"
            embedding = OpenAICompatEmbedding(
                base_url=emb_url, model="qwen3-embedding-0.6b"
            )

    elif cfg.backend == "online":
        api_base = {
            "openai": "https://api.openai.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
            "anthropic": "https://api.anthropic.com/v1",
        }
        base_url = api_base.get(cfg.online.provider, "")
        api_key = os.environ.get(cfg.online.api_key_env, "")
        llm = OpenAICompatLLM(
            base_url=base_url, model=cfg.online.chat_model, api_key=api_key
        )

        if cfg.online.local_embedding:
            emb_url = f"http://127.0.0.1:{cfg.mlc.chat_port}/v1"
            embedding = OpenAICompatEmbedding(
                base_url=emb_url, model="qwen3-embedding-0.6b"
            )
        else:
            embedding = None

    else:
        raise ValueError(f"Unknown backend: {cfg.backend!r}")

    return llm, embedding


class Agent:
    """Task-solving agent with tool use and memory.

    Two construction modes:

    1. **Config-based** (gateway / full app)::

        from localmelo.support.config import load
        agent = Agent(config=load())

    2. **Direct provider injection** (testing / custom setups)::

        agent = Agent(llm=my_llm_provider, embedding=my_emb_provider)
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        llm: BaseLLMProvider | None = None,
        embedding: BaseEmbeddingProvider | None = None,
    ) -> None:
        if llm is not None:
            self._llm = llm
            self._embedding = embedding
        elif config is not None:
            self._llm, self._embedding = _providers_from_config(config)
        else:
            raise TypeError("Agent requires either config= or llm= to be provided")

        # Optional persistent memory backends (env-based opt-in).
        # LOCALMELO_PERSIST_MEMORY=1 enables SQLite-backed history and long-term.
        # LOCALMELO_MEMORY_DIR overrides the default storage directory.
        history_backend = None
        long_backend = None
        if os.environ.get("LOCALMELO_PERSIST_MEMORY"):
            from localmelo.melo.memory.history.sqlite import SqliteHistory
            from localmelo.melo.memory.long.sqlite import SqliteLongTerm

            mem_dir = os.environ.get(
                "LOCALMELO_MEMORY_DIR",
                os.path.expanduser("~/.cache/localmelo/memory"),
            )
            os.makedirs(mem_dir, exist_ok=True)
            history_backend = SqliteHistory(os.path.join(mem_dir, "history.db"))
            if self._embedding:
                long_backend = SqliteLongTerm(os.path.join(mem_dir, "long_term.db"))

        self.hippo = Hippo(
            embedding=self._embedding,
            history=history_backend,
            long=long_backend,
        )
        self.checker = Checker()
        self.executor = Executor(self.hippo, self.checker)
        self.chat = Chat(self._llm)

        register_builtins(self.executor, self.hippo)

    # ── Stage helpers ──
    # Each helper owns one phase of the agent loop.  They return a
    # short status string when the loop should break, or None to
    # continue to the next stage.

    async def _do_retrieval(
        self, query: str
    ) -> tuple[list[Message], list[Message], list[ToolDef], str | None]:
        """Stage 1+2: retrieve context, resolve tools, run boundary checks.

        Returns (long_context, short_window, tools, fail_reason).
        *fail_reason* is non-None when a check fails and the loop must break.
        """
        long_context = await self.hippo.retrieve_context(query)
        short_window = self.hippo.short.get_window()

        tool_hints = self.hippo.extract_tool_hints(long_context + short_window)
        tools = self.hippo.resolve_tools(query, hints=tool_hints)

        resolution_check = self.checker.check_tool_resolution(
            ToolResolutionResult(
                query=query,
                hints=tool_hints,
                resolved_tool_names=[t.name for t in tools],
            )
        )
        if not resolution_check.allowed:
            return (
                long_context,
                short_window,
                tools,
                (f"Tool resolution failed: {resolution_check.reason}"),
            )

        all_msgs = long_context + short_window
        check = await self.checker.pre_plan(all_msgs)
        if not check.allowed:
            return (
                long_context,
                short_window,
                tools,
                (f"Plan check failed: {check.reason}"),
            )

        return long_context, short_window, tools, None

    async def _do_plan(
        self,
        long_context: list[Message],
        short_window: list[Message],
        tools: list[ToolDef],
        query: str,
    ) -> tuple[Message, str | None]:
        """Stage 3: LLM planning step with post-plan check.

        Returns (response, fail_reason).
        """
        response = await self.chat.plan_step(
            context=long_context,
            short=short_window,
            tools=tools,
            query=query,
        )

        check = await self.checker.post_plan(response)
        if not check.allowed:
            return response, f"Response check failed: {check.reason}"

        return response, None

    async def _do_execute(self, response: Message) -> ToolResult:
        """Stage 4: execute tool call and validate the result.

        Uses the sanitized payload from the executor-result checker
        when truncation or modification was applied; falls back to
        the raw outcome otherwise.
        """
        assert response.tool_call is not None
        exec_request = ExecutionRequest(
            tool_name=response.tool_call.tool_name,
            arguments=dict(response.tool_call.arguments),
        )
        outcome = await self.executor.execute_structured(exec_request)

        exec_result_check = self.checker.check_executor_result(
            ExecutorResultPayload(
                tool_name=outcome.tool_name,
                output=outcome.output,
                error=outcome.error,
                duration_ms=outcome.duration_ms,
            )
        )
        if exec_result_check.sanitized_payload is not None:
            sp = exec_result_check.sanitized_payload
            return ToolResult(
                tool_name=sp.tool_name,
                output=sp.output,
                error=sp.error,
                duration_ms=sp.duration_ms,
            )
        return outcome.to_tool_result()

    async def _do_memorize(
        self, task_id: str, response: Message, result: ToolResult
    ) -> None:
        """Stage 5: record step and write checked results to memory."""
        step = StepRecord(
            thought=response.content,
            tool_call=response.tool_call,
            tool_result=result,
        )
        summary = await self.hippo.store_step(task_id, step)

        # Checked write: step summary -> short + long memory
        mem_check = self.checker.check_memory_write(
            MemoryWritePayload(
                text=summary,
                role="assistant",
                metadata={"step_id": step.step_id},
            )
        )
        if mem_check.allowed:
            await self.hippo.memorize(summary, metadata={"step_id": step.step_id})

        # Checked write: tool result -> short memory
        output = result.error if result.error else result.output
        tool_msg = f"[{result.tool_name}] {output}"
        tool_mem_check = self.checker.check_memory_write(
            MemoryWritePayload(text=tool_msg, role="tool")
        )
        if tool_mem_check.allowed:
            self.hippo.short.append(Message(role="tool", content=tool_msg))

    # ── Step estimation ──

    async def _estimate_max_steps(self, query: str) -> int:
        """Ask the LLM for a conservative step-count estimate.

        The result is clamped to [MIN_AGENT_STEPS, MAX_AGENT_STEPS].
        On parse failure the hard ceiling is used as a safe fallback.
        """
        estimate = await self.chat.estimate_steps(query)
        if estimate < 0:
            return MAX_AGENT_STEPS
        return max(MIN_AGENT_STEPS, min(estimate, MAX_AGENT_STEPS))

    # ── Main loop ──

    async def run(self, query: str) -> str:
        task = TaskRecord(query=query)
        await self.hippo.save_task(task)

        self.hippo.short.append(Message(role="user", content=query))

        max_steps = await self._estimate_max_steps(query)

        for _ in range(max_steps):
            # Retrieval + tool resolution + boundary checks
            long_context, short_window, tools, fail = await self._do_retrieval(query)
            if fail is not None:
                task.status = "failed"
                task.result = fail
                break

            # LLM planning step
            response, fail = await self._do_plan(
                long_context, short_window, tools, query
            )
            if fail is not None:
                task.status = "failed"
                task.result = fail
                break

            # No tool call -> direct answer; loop terminates
            if response.tool_call is None:
                task.status = "completed"
                task.result = response.content
                break

            # Execute tool call and validate result
            result = await self._do_execute(response)

            # Record step and write checked results to memory
            await self._do_memorize(task.task_id, response, result)
        else:
            # for-else: loop exhausted without break
            task.status = "failed"
            task.result = "Max steps reached"

        await self.hippo.save_task(task)
        return task.result

    async def close(self) -> None:
        self.hippo.close()
        await self._llm.close()
        if self._embedding:
            await self._embedding.close()
