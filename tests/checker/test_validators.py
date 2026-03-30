"""Tests for Checker boundary validators."""

import pytest

from localmelo.melo.checker import (
    Checker,
    ExecutorRequest,
    ExecutorResultPayload,
    GatewayIngressPayload,
    MemoryWritePayload,
    SessionTransition,
    ToolResolutionResult,
    ValidationResult,
)
from localmelo.melo.checker.validators import (
    BLOCKED_COMMANDS,
    MAX_MEMORY_TEXT_LEN,
    MAX_OUTPUT_LEN,
    MAX_QUERY_LEN,
    validate_executor_request,
    validate_executor_result,
    validate_gateway_ingress,
    validate_memory_write,
    validate_session_transition,
    validate_tool_resolution,
)
from localmelo.melo.schema import Message, ToolCall, ToolDef


@pytest.fixture
def checker() -> Checker:
    return Checker()


# ── Gateway ingress ──


class TestGatewayIngress:
    def test_valid_query(self) -> None:
        payload = GatewayIngressPayload(query="What is 2+2?")
        result = validate_gateway_ingress(payload)
        assert result.allowed
        assert result.sanitized_payload is not None
        assert result.sanitized_payload.query == "What is 2+2?"

    def test_valid_query_with_session_id(self) -> None:
        payload = GatewayIngressPayload(query="hello", session_id="abc123")
        result = validate_gateway_ingress(payload)
        assert result.allowed

    def test_empty_query_rejected(self) -> None:
        payload = GatewayIngressPayload(query="")
        result = validate_gateway_ingress(payload)
        assert not result.allowed
        assert "non-empty" in result.reason

    def test_whitespace_only_query_rejected(self) -> None:
        payload = GatewayIngressPayload(query="   \n\t  ")
        result = validate_gateway_ingress(payload)
        assert not result.allowed

    def test_oversized_query_rejected(self) -> None:
        payload = GatewayIngressPayload(query="x" * (MAX_QUERY_LEN + 1))
        result = validate_gateway_ingress(payload)
        assert not result.allowed
        assert "too large" in result.reason.lower()

    def test_max_length_query_allowed(self) -> None:
        payload = GatewayIngressPayload(query="x" * MAX_QUERY_LEN)
        result = validate_gateway_ingress(payload)
        assert result.allowed

    def test_invalid_session_id_format(self) -> None:
        payload = GatewayIngressPayload(query="hi", session_id="bad session id!")
        result = validate_gateway_ingress(payload)
        assert not result.allowed
        assert "session_id" in result.reason

    def test_session_id_too_long(self) -> None:
        payload = GatewayIngressPayload(query="hi", session_id="a" * 65)
        result = validate_gateway_ingress(payload)
        assert not result.allowed

    def test_none_session_id_allowed(self) -> None:
        payload = GatewayIngressPayload(query="hi", session_id=None)
        result = validate_gateway_ingress(payload)
        assert result.allowed

    def test_query_whitespace_stripped(self) -> None:
        payload = GatewayIngressPayload(query="  hello world  ")
        result = validate_gateway_ingress(payload)
        assert result.allowed
        assert result.sanitized_payload.query == "hello world"

    def test_via_checker_instance(self, checker: Checker) -> None:
        payload = GatewayIngressPayload(query="test")
        result = checker.check_gateway_ingress(payload)
        assert result.allowed


# ── Session state transitions ──


class TestSessionTransition:
    @pytest.mark.parametrize(
        "from_s,to_s",
        [
            ("idle", "running"),
            ("running", "idle"),
            ("running", "closed"),
            ("idle", "closed"),
        ],
    )
    def test_legal_transitions(self, from_s: str, to_s: str) -> None:
        t = SessionTransition(from_status=from_s, to_status=to_s)
        result = validate_session_transition(t)
        assert result.allowed

    @pytest.mark.parametrize(
        "from_s,to_s",
        [
            ("closed", "idle"),
            ("closed", "running"),
            ("closed", "closed"),
            ("idle", "idle"),
            ("running", "running"),
        ],
    )
    def test_illegal_transitions(self, from_s: str, to_s: str) -> None:
        t = SessionTransition(from_status=from_s, to_status=to_s)
        result = validate_session_transition(t)
        assert not result.allowed
        assert "Illegal transition" in result.reason or "Unknown" in result.reason

    def test_unknown_source_state(self) -> None:
        t = SessionTransition(from_status="bogus", to_status="idle")
        result = validate_session_transition(t)
        assert not result.allowed
        assert "Unknown source" in result.reason

    def test_unknown_target_state(self) -> None:
        t = SessionTransition(from_status="idle", to_status="bogus")
        result = validate_session_transition(t)
        assert not result.allowed
        assert "Unknown target" in result.reason

    def test_via_checker_instance(self, checker: Checker) -> None:
        t = SessionTransition(from_status="idle", to_status="running")
        result = checker.check_session_transition(t)
        assert result.allowed


# ── Tool resolution ──


class TestToolResolution:
    def test_valid_resolution(self) -> None:
        r = ToolResolutionResult(
            query="list files",
            hints=["file"],
            resolved_tool_names=["shell_exec", "file_read"],
        )
        result = validate_tool_resolution(r)
        assert result.allowed

    def test_empty_query_rejected(self) -> None:
        r = ToolResolutionResult(query="", resolved_tool_names=["shell_exec"])
        result = validate_tool_resolution(r)
        assert not result.allowed
        assert "empty" in result.reason.lower()

    def test_empty_tool_name_rejected(self) -> None:
        r = ToolResolutionResult(query="test", resolved_tool_names=["shell_exec", ""])
        result = validate_tool_resolution(r)
        assert not result.allowed
        assert "Invalid" in result.reason

    def test_duplicate_tool_names_rejected(self) -> None:
        r = ToolResolutionResult(
            query="test",
            resolved_tool_names=["shell_exec", "file_read", "shell_exec"],
        )
        result = validate_tool_resolution(r)
        assert not result.allowed
        assert "Duplicate" in result.reason

    def test_empty_resolved_list_allowed(self) -> None:
        r = ToolResolutionResult(query="obscure task", resolved_tool_names=[])
        result = validate_tool_resolution(r)
        assert result.allowed

    def test_via_checker_instance(self, checker: Checker) -> None:
        r = ToolResolutionResult(query="test", resolved_tool_names=["shell_exec"])
        result = checker.check_tool_resolution(r)
        assert result.allowed


# ── Executor request ──


class TestExecutorRequest:
    def test_valid_request(self) -> None:
        req = ExecutorRequest(
            tool_name="shell_exec",
            arguments={"command": "ls"},
            tool_def_name="shell_exec",
        )
        result = validate_executor_request(req)
        assert result.allowed

    def test_empty_tool_name_rejected(self) -> None:
        req = ExecutorRequest(tool_name="", tool_def_name="shell_exec")
        result = validate_executor_request(req)
        assert not result.allowed

    def test_unresolved_tool_rejected(self) -> None:
        req = ExecutorRequest(tool_name="mystery", tool_def_name=None)
        result = validate_executor_request(req)
        assert not result.allowed
        assert "Unknown tool" in result.reason

    def test_tool_name_mismatch_rejected(self) -> None:
        req = ExecutorRequest(tool_name="shell_exec", tool_def_name="file_read")
        result = validate_executor_request(req)
        assert not result.allowed
        assert "mismatch" in result.reason

    def test_blocked_shell_command(self) -> None:
        req = ExecutorRequest(
            tool_name="shell_exec",
            arguments={"command": "rm -rf /"},
            tool_def_name="shell_exec",
        )
        result = validate_executor_request(req)
        assert not result.allowed
        assert "Blocked" in result.reason

    def test_safe_shell_command_allowed(self) -> None:
        req = ExecutorRequest(
            tool_name="shell_exec",
            arguments={"command": "echo hello"},
            tool_def_name="shell_exec",
        )
        result = validate_executor_request(req)
        assert result.allowed

    def test_blocked_mkfs(self) -> None:
        req = ExecutorRequest(
            tool_name="shell_exec",
            arguments={"command": "mkfs /dev/sda1"},
            tool_def_name="shell_exec",
        )
        result = validate_executor_request(req)
        assert not result.allowed

    def test_blocked_shutdown(self) -> None:
        req = ExecutorRequest(
            tool_name="shell_exec",
            arguments={"command": "shutdown -h now"},
            tool_def_name="shell_exec",
        )
        result = validate_executor_request(req)
        assert not result.allowed

    def test_non_shell_tool_not_command_checked(self) -> None:
        req = ExecutorRequest(
            tool_name="file_read",
            arguments={"path": "/etc/passwd"},
            tool_def_name="file_read",
        )
        result = validate_executor_request(req)
        assert result.allowed

    def test_via_checker_instance(self, checker: Checker) -> None:
        req = ExecutorRequest(
            tool_name="shell_exec",
            arguments={"command": "ls"},
            tool_def_name="shell_exec",
        )
        result = checker.check_executor_request(req)
        assert result.allowed


# ── Executor result ──


class TestExecutorResult:
    def test_valid_result(self) -> None:
        r = ExecutorResultPayload(
            tool_name="shell_exec", output="hello", duration_ms=10.0
        )
        result = validate_executor_result(r)
        assert result.allowed
        assert result.sanitized_payload is None

    def test_oversized_output_truncated(self) -> None:
        r = ExecutorResultPayload(
            tool_name="shell_exec",
            output="x" * (MAX_OUTPUT_LEN + 1000),
            duration_ms=5.0,
        )
        result = validate_executor_result(r)
        assert result.allowed
        assert result.reason == "Output truncated"
        assert result.sanitized_payload is not None
        assert len(result.sanitized_payload.output) < len(r.output)
        assert result.sanitized_payload.output.endswith("[truncated]")

    def test_max_length_output_not_truncated(self) -> None:
        r = ExecutorResultPayload(
            tool_name="shell_exec",
            output="x" * MAX_OUTPUT_LEN,
            duration_ms=1.0,
        )
        result = validate_executor_result(r)
        assert result.allowed
        assert result.sanitized_payload is None

    def test_negative_duration_rejected(self) -> None:
        r = ExecutorResultPayload(tool_name="shell_exec", output="ok", duration_ms=-1.0)
        result = validate_executor_result(r)
        assert not result.allowed
        assert "Negative duration" in result.reason

    def test_empty_tool_name_rejected(self) -> None:
        r = ExecutorResultPayload(tool_name="", output="ok", duration_ms=1.0)
        result = validate_executor_result(r)
        assert not result.allowed

    def test_error_field_preserved_in_truncation(self) -> None:
        r = ExecutorResultPayload(
            tool_name="shell_exec",
            output="x" * (MAX_OUTPUT_LEN + 100),
            error="some warning",
            duration_ms=2.0,
        )
        result = validate_executor_result(r)
        assert result.sanitized_payload.error == "some warning"
        assert result.sanitized_payload.duration_ms == 2.0

    def test_via_checker_instance(self, checker: Checker) -> None:
        r = ExecutorResultPayload(tool_name="shell_exec", output="hi", duration_ms=1.0)
        result = checker.check_executor_result(r)
        assert result.allowed


# ── Memory write ──


class TestMemoryWrite:
    def test_valid_write(self) -> None:
        p = MemoryWritePayload(text="remember this", role="user")
        result = validate_memory_write(p)
        assert result.allowed

    def test_empty_text_rejected(self) -> None:
        p = MemoryWritePayload(text="", role="user")
        result = validate_memory_write(p)
        assert not result.allowed
        assert "empty" in result.reason.lower()

    def test_whitespace_only_text_rejected(self) -> None:
        p = MemoryWritePayload(text="   \n  ", role="user")
        result = validate_memory_write(p)
        assert not result.allowed

    def test_oversized_text_rejected(self) -> None:
        p = MemoryWritePayload(text="x" * (MAX_MEMORY_TEXT_LEN + 1), role="assistant")
        result = validate_memory_write(p)
        assert not result.allowed
        assert "too large" in result.reason.lower()

    def test_invalid_role_rejected(self) -> None:
        p = MemoryWritePayload(text="ok", role="hacker")
        result = validate_memory_write(p)
        assert not result.allowed
        assert "Invalid memory role" in result.reason

    def test_valid_roles(self) -> None:
        for role in ("user", "assistant", "system", "tool"):
            p = MemoryWritePayload(text="ok", role=role)
            result = validate_memory_write(p)
            assert result.allowed, f"Role {role!r} should be valid"

    def test_empty_role_allowed(self) -> None:
        p = MemoryWritePayload(text="ok", role="")
        result = validate_memory_write(p)
        assert result.allowed

    def test_metadata_passed_through(self) -> None:
        p = MemoryWritePayload(text="note", role="user", metadata={"source": "test"})
        result = validate_memory_write(p)
        assert result.allowed

    def test_via_checker_instance(self, checker: Checker) -> None:
        p = MemoryWritePayload(text="test", role="user")
        result = checker.check_memory_write(p)
        assert result.allowed


# ── Async boundary checks ──


class TestAsyncBoundaryChecks:
    """Checker async boundary methods (pre_plan, post_plan, pre_execute)."""

    @pytest.mark.asyncio
    async def test_pre_plan_ok(self, checker: Checker) -> None:
        msgs = [Message(role="user", content="hello")]
        result = await checker.pre_plan(msgs)
        assert result.allowed

    @pytest.mark.asyncio
    async def test_pre_plan_too_large(self, checker: Checker) -> None:
        msgs = [Message(role="user", content="x" * 200_000)]
        result = await checker.pre_plan(msgs)
        assert not result.allowed

    @pytest.mark.asyncio
    async def test_post_plan_ok(self, checker: Checker) -> None:
        msg = Message(role="assistant", content="Sure thing")
        result = await checker.post_plan(msg)
        assert result.allowed

    @pytest.mark.asyncio
    async def test_post_plan_empty_tool_name(self, checker: Checker) -> None:
        msg = Message(
            role="assistant",
            content="",
            tool_call=ToolCall(tool_name="", arguments={}),
        )
        result = await checker.post_plan(msg)
        assert not result.allowed

    @pytest.mark.asyncio
    async def test_pre_execute_blocks_dangerous(self, checker: Checker) -> None:
        td = ToolDef(name="shell_exec", description="", parameters={})
        tc = ToolCall(tool_name="shell_exec", arguments={"command": "rm -rf /"})
        result = await checker.pre_execute(tc, td)
        assert not result.allowed

    @pytest.mark.asyncio
    async def test_pre_execute_allows_safe(self, checker: Checker) -> None:
        td = ToolDef(name="shell_exec", description="", parameters={})
        tc = ToolCall(tool_name="shell_exec", arguments={"command": "ls -la"})
        result = await checker.pre_execute(tc, td)
        assert result.allowed

    @pytest.mark.asyncio
    async def test_pre_execute_unknown_tool(self, checker: Checker) -> None:
        tc = ToolCall(tool_name="fake_tool", arguments={})
        result = await checker.pre_execute(tc, None)
        assert not result.allowed

    def test_check_result_dataclass(self) -> None:
        from localmelo.melo.schema import CheckResult

        cr = CheckResult(allowed=True, reason="ok")
        assert cr.allowed
        assert cr.reason == "ok"
        assert cr.modified_payload is None

    def test_validation_result_dataclass(self) -> None:
        vr = ValidationResult(allowed=False, reason="denied")
        assert not vr.allowed
        assert vr.reason == "denied"
        assert vr.sanitized_payload is None


# ── Single source of truth for constants ──


class TestConstantDeduplication:
    """BLOCKED_COMMANDS and MAX_OUTPUT_LEN must be the same object
    whether imported from checker.py or validators.py."""

    def test_blocked_commands_single_source(self) -> None:
        from localmelo.melo.checker.checker import (
            BLOCKED_COMMANDS as CHECKER_BLOCKED,
        )
        from localmelo.melo.checker.validators import (
            BLOCKED_COMMANDS as VALIDATORS_BLOCKED,
        )

        assert CHECKER_BLOCKED is VALIDATORS_BLOCKED

    def test_max_output_len_single_source(self) -> None:
        from localmelo.melo.checker.checker import (
            MAX_OUTPUT_LEN as CHECKER_MAX,
        )
        from localmelo.melo.checker.validators import (
            MAX_OUTPUT_LEN as VALIDATORS_MAX,
        )

        assert CHECKER_MAX is VALIDATORS_MAX

    def test_blocked_re_single_source(self) -> None:
        from localmelo.melo.checker.checker import (
            _BLOCKED_RE as CHECKER_RE,
        )
        from localmelo.melo.checker.validators import (
            _BLOCKED_RE as VALIDATORS_RE,
        )

        assert CHECKER_RE is VALIDATORS_RE

    def test_blocked_commands_has_six_patterns(self) -> None:
        assert len(BLOCKED_COMMANDS) == 6


# ── All blocked commands via validate_executor_request ──


class TestAllBlockedCommandsStructured:
    """Every blocked pattern must be caught by validate_executor_request."""

    DANGEROUS = [
        ("rm -rf /home", "rm -rf"),
        ("mkfs /dev/sda1", "mkfs"),
        ("dd if=/dev/zero of=/dev/sda", "dd"),
        (":(){:|:&};:", "fork bomb compact"),
        (":() { :|:& };:", "fork bomb spaced"),
        (":() { :|:& }; :", "fork bomb spaced trailing"),
        ("shutdown -h now", "shutdown"),
        ("reboot", "reboot"),
    ]

    @pytest.mark.parametrize(
        "cmd,label",
        DANGEROUS,
        ids=[d[1] for d in DANGEROUS],
    )
    def test_structured_blocks(self, cmd: str, label: str) -> None:
        req = ExecutorRequest(
            tool_name="shell_exec",
            arguments={"command": cmd},
            tool_def_name="shell_exec",
        )
        result = validate_executor_request(req)
        assert not result.allowed, f"{label!r} should be blocked"
        assert "Blocked" in result.reason


# ── All blocked commands via pre_execute ──


class TestAllBlockedCommandsPreExecute:
    """Every blocked pattern must be caught by checker.pre_execute."""

    DANGEROUS = [
        ("rm -rf /home", "rm -rf"),
        ("mkfs /dev/sda1", "mkfs"),
        ("dd if=/dev/zero of=/dev/sda", "dd"),
        (":(){:|:&};:", "fork bomb compact"),
        (":() { :|:& };:", "fork bomb spaced"),
        (":() { :|:& }; :", "fork bomb spaced trailing"),
        ("shutdown -h now", "shutdown"),
        ("reboot", "reboot"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cmd,label",
        DANGEROUS,
        ids=[d[1] for d in DANGEROUS],
    )
    async def test_pre_execute_blocks(
        self, checker: Checker, cmd: str, label: str
    ) -> None:
        td = ToolDef(name="shell_exec", description="", parameters={})
        tc = ToolCall(tool_name="shell_exec", arguments={"command": cmd})
        result = await checker.pre_execute(tc, td)
        assert not result.allowed, f"{label!r} should be blocked"
        assert "Blocked" in result.reason
