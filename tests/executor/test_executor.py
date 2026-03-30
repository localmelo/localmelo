"""Executor tests.

Covers: builtin execution, unknown tool, blocked command, timeout,
file path policy, structured result/artifacts.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unittest

from localmelo.melo.checker import Checker
from localmelo.melo.executor import Executor, register_builtins
from localmelo.melo.executor.models import (
    ArtifactMeta,
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.executor.policy import WorkspacePolicy
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import ToolDef, ToolResult

# ── helpers ──


def _make_executor(
    *, timeout_ms: float = 60_000, workspace_root: str | None = None
) -> tuple[Executor, Hippo, Checker]:
    hippo = Hippo()
    checker = Checker()
    executor = Executor(
        hippo, checker, timeout_ms=timeout_ms, workspace_root=workspace_root
    )
    register_builtins(executor, hippo)
    return executor, hippo, checker


# ── 1. Successful builtin execution ──


class TestBuiltinExecution(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.executor, self.hippo, _ = _make_executor()

    async def test_python_exec(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="python_exec", arguments={"code": "print(1+1)"})
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertEqual(outcome.output, "2")
        self.assertEqual(outcome.error, "")
        self.assertGreater(outcome.duration_ms, 0)

    async def test_shell_exec(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="shell_exec", arguments={"command": "echo hello"}
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertEqual(outcome.output, "hello")

    async def test_file_read_write_roundtrip(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            # write
            wr = await self.executor.execute_structured(
                ExecutionRequest(
                    tool_name="file_write",
                    arguments={"path": path, "content": "round trip"},
                )
            )
            self.assertEqual(wr.status, ExecutionStatus.SUCCESS)
            self.assertIn("Written", wr.output)

            # read back
            rd = await self.executor.execute_structured(
                ExecutionRequest(tool_name="file_read", arguments={"path": path})
            )
            self.assertEqual(rd.output, "round trip")
        finally:
            os.unlink(path)


# ── 2. Unknown tool ──


class TestUnknownTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.executor, self.hippo, _ = _make_executor()

    async def test_structured_error_category(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="no_such_tool")
        )
        self.assertEqual(outcome.status, ExecutionStatus.ERROR)
        self.assertEqual(outcome.error_category, ErrorCategory.TOOL_NOT_FOUND)

    async def test_callable_without_registry_entry(self) -> None:
        """A callable registered but no ToolDef -> still not found."""

        async def _ghost() -> str:
            return "boo"

        self.executor.register("ghost", _ghost)

        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="ghost")
        )
        self.assertEqual(outcome.status, ExecutionStatus.ERROR)
        self.assertEqual(outcome.error_category, ErrorCategory.TOOL_NOT_FOUND)


# ── 3. Blocked command ──


class TestBlockedCommand(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.executor, self.hippo, _ = _make_executor()

    async def test_blocked_structured(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="shell_exec", arguments={"command": "rm -rf /"})
        )
        self.assertEqual(outcome.status, ExecutionStatus.BLOCKED)
        self.assertEqual(outcome.error_category, ErrorCategory.BLOCKED_BY_CHECKER)

    async def test_mkfs_blocked(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="shell_exec", arguments={"command": "mkfs /dev/sda1"}
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.BLOCKED)


# ── 4. Timeout path ──


class TestTimeout(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.executor, self.hippo, _ = _make_executor(timeout_ms=100)

        # register a slow tool
        slow_def = ToolDef(
            name="slow_tool",
            description="Sleeps forever",
            parameters={"type": "object", "properties": {}},
        )
        self.hippo.register_tool(slow_def)

        async def _slow() -> str:
            await asyncio.sleep(60)
            return "done"

        self.executor.register("slow_tool", _slow)

    async def test_timeout_structured(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="slow_tool")
        )
        self.assertEqual(outcome.status, ExecutionStatus.TIMEOUT)
        self.assertEqual(outcome.error_category, ErrorCategory.TIMEOUT)
        self.assertGreater(outcome.duration_ms, 0)
        self.assertIn("timed out", outcome.error.lower())

    async def test_per_request_timeout_override(self) -> None:
        """A generous per-request timeout lets a fast tool succeed."""
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="python_exec",
                arguments={"code": "print('fast')"},
                timeout_ms=30_000,
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertEqual(outcome.output, "fast")


# ── 5. File path policy ──


class TestFilePathPolicy(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.executor, self.hippo, _ = _make_executor(workspace_root=self.tmpdir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_allowed_path_write(self) -> None:
        path = os.path.join(self.tmpdir, "ok.txt")
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="file_write",
                arguments={"path": path, "content": "ok"},
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertIn("Written", outcome.output)

    async def test_allowed_path_read(self) -> None:
        path = os.path.join(self.tmpdir, "readable.txt")
        with open(path, "w") as f:
            f.write("data")
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="file_read", arguments={"path": path})
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertEqual(outcome.output, "data")

    async def test_blocked_path_escape(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="file_read", arguments={"path": "/etc/hosts"})
        )
        self.assertEqual(outcome.status, ExecutionStatus.ERROR)
        self.assertEqual(outcome.error_category, ErrorCategory.PATH_POLICY_VIOLATION)

    async def test_blocked_path_structured(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="file_read", arguments={"path": "/etc/hosts"})
        )
        self.assertEqual(outcome.status, ExecutionStatus.ERROR)
        self.assertEqual(outcome.error_category, ErrorCategory.PATH_POLICY_VIOLATION)

    async def test_dotdot_escape_blocked(self) -> None:
        path = os.path.join(self.tmpdir, "..", "etc", "passwd")
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="file_read", arguments={"path": path})
        )
        self.assertEqual(outcome.error_category, ErrorCategory.PATH_POLICY_VIOLATION)

    async def test_per_request_workspace_override(self) -> None:
        """Per-request workspace_root narrows the allowed root."""
        subdir = os.path.join(self.tmpdir, "sub")
        os.makedirs(subdir, exist_ok=True)

        # file in tmpdir but outside subdir
        path = os.path.join(self.tmpdir, "outside.txt")
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="file_write",
                arguments={"path": path, "content": "x"},
                workspace_root=subdir,
            )
        )
        self.assertEqual(outcome.error_category, ErrorCategory.PATH_POLICY_VIOLATION)

    async def test_no_policy_allows_any_path(self) -> None:
        """Without workspace_root, any valid path is allowed."""
        hippo = Hippo()
        checker = Checker()
        executor = Executor(hippo, checker)
        register_builtins(executor, hippo)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("free")
            path = f.name
        try:
            outcome = await executor.execute_structured(
                ExecutionRequest(tool_name="file_read", arguments={"path": path})
            )
            self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
            self.assertEqual(outcome.output, "free")
        finally:
            os.unlink(path)


# ── 6. Structured result / artifact metadata ──


class TestStructuredResult(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.executor, self.hippo, _ = _make_executor()

    async def test_outcome_fields_on_success(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="python_exec", arguments={"code": "print('hi')"})
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertEqual(outcome.output, "hi")
        self.assertEqual(outcome.error, "")
        self.assertEqual(outcome.error_category, ErrorCategory.NONE)
        self.assertGreater(outcome.duration_ms, 0)
        self.assertIsInstance(outcome.artifacts, list)
        self.assertIsInstance(outcome.logs, list)

    async def test_file_write_artifact(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            outcome = await self.executor.execute_structured(
                ExecutionRequest(
                    tool_name="file_write",
                    arguments={"path": path, "content": "artifact test"},
                )
            )
            self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
            self.assertEqual(len(outcome.artifacts), 1)
            art = outcome.artifacts[0]
            self.assertEqual(art.kind, "file")
            self.assertEqual(art.path, path)
            self.assertEqual(art.size_bytes, len(b"artifact test"))
            self.assertEqual(art.description, "Written file")
        finally:
            os.unlink(path)

    async def test_file_read_artifact(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("read me")
            path = f.name
        try:
            outcome = await self.executor.execute_structured(
                ExecutionRequest(tool_name="file_read", arguments={"path": path})
            )
            self.assertEqual(len(outcome.artifacts), 1)
            self.assertEqual(outcome.artifacts[0].kind, "file")
            self.assertEqual(outcome.artifacts[0].size_bytes, len(b"read me"))
        finally:
            os.unlink(path)

    async def test_non_file_tool_has_no_artifacts(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="python_exec", arguments={"code": "print(1)"})
        )
        self.assertEqual(outcome.artifacts, [])

    async def test_to_tool_result_conversion(self) -> None:
        outcome = ExecutionOutcome(
            tool_name="test",
            status=ExecutionStatus.SUCCESS,
            output="hello",
            error="",
            duration_ms=42.0,
            artifacts=[ArtifactMeta(kind="file", path="/tmp/x")],
        )
        result = outcome.to_tool_result()
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_name, "test")
        self.assertEqual(result.output, "hello")
        self.assertEqual(result.duration_ms, 42.0)

    async def test_error_outcome_preserves_duration(self) -> None:
        err_def = ToolDef(
            name="err_tool",
            description="raises",
            parameters={"type": "object", "properties": {}},
        )
        hippo = Hippo()
        checker = Checker()
        executor = Executor(hippo, checker)
        register_builtins(executor, hippo)
        hippo.register_tool(err_def)

        async def _err() -> str:
            raise ValueError("boom")

        executor.register("err_tool", _err)

        outcome = await executor.execute_structured(
            ExecutionRequest(tool_name="err_tool")
        )
        self.assertEqual(outcome.status, ExecutionStatus.ERROR)
        self.assertEqual(outcome.error, "boom")
        self.assertGreater(outcome.duration_ms, 0)


# ── WorkspacePolicy unit tests ──


class TestWorkspacePolicy(unittest.TestCase):
    def test_no_root_allows_everything(self) -> None:
        policy = WorkspacePolicy()
        self.assertTrue(policy.check_path("/any/path"))
        self.assertIsNone(policy.root)

    def test_root_allows_children(self) -> None:
        policy = WorkspacePolicy("/tmp/workspace")
        self.assertTrue(policy.check_path("/tmp/workspace/file.txt"))
        self.assertTrue(policy.check_path("/tmp/workspace/sub/deep"))

    def test_root_blocks_siblings(self) -> None:
        policy = WorkspacePolicy("/tmp/workspace")
        self.assertFalse(policy.check_path("/tmp/other"))
        self.assertFalse(policy.check_path("/etc/passwd"))

    def test_root_blocks_dotdot(self) -> None:
        policy = WorkspacePolicy("/tmp/workspace")
        self.assertFalse(policy.check_path("/tmp/workspace/../other"))

    def test_root_itself_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            policy = WorkspacePolicy(td)
            self.assertTrue(policy.check_path(td))


# ── 7. All builtins via execute_structured() ──


class TestAllBuiltinsStructured(unittest.IsolatedAsyncioTestCase):
    """Every builtin must work through execute_structured()."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.executor, self.hippo, _ = _make_executor()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_shell_exec_structured(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="shell_exec",
                arguments={"command": "echo structured"},
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertEqual(outcome.output, "structured")
        self.assertGreater(outcome.duration_ms, 0)

    async def test_file_write_structured(self) -> None:
        path = os.path.join(self.tmpdir, "structured.txt")
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="file_write",
                arguments={"path": path, "content": "hello structured"},
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertIn("Written", outcome.output)
        self.assertEqual(len(outcome.artifacts), 1)
        self.assertEqual(outcome.artifacts[0].kind, "file")

    async def test_file_read_structured(self) -> None:
        path = os.path.join(self.tmpdir, "read_me.txt")
        with open(path, "w") as f:
            f.write("structured content")
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="file_read",
                arguments={"path": path},
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertEqual(outcome.output, "structured content")
        self.assertEqual(len(outcome.artifacts), 1)

    async def test_python_exec_structured(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="python_exec",
                arguments={"code": "print(3*7)"},
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCESS)
        self.assertEqual(outcome.output, "21")


# ── 8. All blocked commands ──


class TestAllBlockedCommands(unittest.IsolatedAsyncioTestCase):
    """Every pattern in BLOCKED_COMMANDS must be caught."""

    def setUp(self) -> None:
        self.executor, self.hippo, _ = _make_executor()

    DANGEROUS_COMMANDS = [
        ("rm -rf /home", "rm -rf"),
        ("mkfs /dev/sda1", "mkfs"),
        ("dd if=/dev/zero of=/dev/sda", "dd"),
        (":(){:|:&};:", "fork bomb compact"),
        (":() { :|:& };:", "fork bomb spaced"),
        (":() { :|:& }; :", "fork bomb spaced trailing"),
        ("shutdown -h now", "shutdown"),
        ("reboot", "reboot"),
    ]

    async def test_blocked_via_execute_structured(self) -> None:
        for cmd, label in self.DANGEROUS_COMMANDS:
            with self.subTest(label=label):
                outcome = await self.executor.execute_structured(
                    ExecutionRequest(
                        tool_name="shell_exec",
                        arguments={"command": cmd},
                    )
                )
                self.assertEqual(
                    outcome.status,
                    ExecutionStatus.BLOCKED,
                    f"{label!r} should be BLOCKED via execute_structured()",
                )
                self.assertEqual(
                    outcome.error_category,
                    ErrorCategory.BLOCKED_BY_CHECKER,
                )


# ── 9. Unknown tool error categories and messages ──


class TestUnknownToolErrorMessages(unittest.IsolatedAsyncioTestCase):
    """Distinct error messages for 'not in registry' vs 'callable missing'."""

    def setUp(self) -> None:
        self.executor, self.hippo, _ = _make_executor()

    async def test_not_in_registry_error_message(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="totally_unknown")
        )
        self.assertEqual(outcome.status, ExecutionStatus.ERROR)
        self.assertEqual(outcome.error_category, ErrorCategory.TOOL_NOT_FOUND)
        self.assertIn("not found in registry", outcome.error.lower())

    async def test_callable_not_registered_error_message(self) -> None:
        """ToolDef exists but no callable registered."""
        ghost_def = ToolDef(
            name="ghost_tool",
            description="Has a def but no callable",
            parameters={"type": "object", "properties": {}},
        )
        self.hippo.register_tool(ghost_def)
        # Deliberately do NOT register a callable

        outcome = await self.executor.execute_structured(
            ExecutionRequest(tool_name="ghost_tool")
        )
        self.assertEqual(outcome.status, ExecutionStatus.ERROR)
        self.assertEqual(outcome.error_category, ErrorCategory.TOOL_NOT_FOUND)
        self.assertIn("no callable registered", outcome.error.lower())

    async def test_distinct_messages(self) -> None:
        """The two error paths produce distinguishable messages."""
        # Not in registry
        o1 = await self.executor.execute_structured(
            ExecutionRequest(tool_name="no_registry")
        )
        # Def exists, no callable
        self.hippo.register_tool(
            ToolDef(
                name="no_callable",
                description="",
                parameters={"type": "object", "properties": {}},
            )
        )
        o2 = await self.executor.execute_structured(
            ExecutionRequest(tool_name="no_callable")
        )
        self.assertNotEqual(o1.error, o2.error)


# ── 10. Per-request timeout override (shorter) ──


class TestPerRequestTimeoutShort(unittest.IsolatedAsyncioTestCase):
    """A short per-request timeout overrides the generous default."""

    def setUp(self) -> None:
        # Default timeout is generous (60s)
        self.executor, self.hippo, _ = _make_executor(timeout_ms=60_000)

        slow_def = ToolDef(
            name="slow_tool",
            description="Sleeps",
            parameters={"type": "object", "properties": {}},
        )
        self.hippo.register_tool(slow_def)

        async def _slow() -> str:
            await asyncio.sleep(60)
            return "done"

        self.executor.register("slow_tool", _slow)

    async def test_short_per_request_timeout_triggers(self) -> None:
        outcome = await self.executor.execute_structured(
            ExecutionRequest(
                tool_name="slow_tool",
                timeout_ms=100,  # 100ms override
            )
        )
        self.assertEqual(outcome.status, ExecutionStatus.TIMEOUT)
        self.assertEqual(outcome.error_category, ErrorCategory.TIMEOUT)


if __name__ == "__main__":
    unittest.main()
