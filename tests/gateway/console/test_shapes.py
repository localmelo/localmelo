"""The pure JSON shapes: a pending approval's ``target``."""

from __future__ import annotations

from pathlib import Path

from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.support.gateway.console.shapes import approval_json


def request(tool_name: str, path: str, root: str | None) -> ApprovalRequest:
    return ApprovalRequest(
        request_id="req-1",
        run_id="",
        call_id="call_1",
        tool_name=tool_name,
        arguments={"path": path},
        preview=path,
        workspace_root=root,
    )


def test_approval_reports_when_arguments_were_truncated() -> None:
    pending = approval_json(
        ApprovalRequest(
            request_id="req",
            run_id="",
            call_id="call",
            tool_name="shell_exec",
            arguments={"command": "a" * 100_001},
            preview="command",
            workspace_root=None,
        )
    )
    assert pending["arguments_complete"] is False
    assert pending["arguments"]["command"].endswith("… [+1 chars]")
    assert (
        approval_json(request("shell_exec", "ls", None))["arguments_complete"] is True
    )


def test_approval_reports_converted_and_deep_arguments() -> None:
    for arguments in (
        {"x": (1, 2)},
        {"x": float("nan")},
        {"x": {1: "value"}},
        {"x": {"y": "z" * 100_001}},
    ):
        pending = approval_json(
            ApprovalRequest(
                request_id="req",
                run_id="",
                call_id="call",
                tool_name="custom",
                arguments=arguments,
                preview="",
                workspace_root=None,
            )
        )
        assert pending["arguments_complete"] is False


def test_a_missing_file_target(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    pending = approval_json(
        request("file_write", str(workspace / "todo.md"), str(workspace))
    )
    target = pending["target"]
    assert target == {"path_label": "todo.md", "exists": False, "size_bytes": 0}


def test_an_existing_file_target(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    path = workspace / "todo.md"
    path.write_text("abc")
    target = approval_json(request("file_write", str(path), str(workspace)))["target"]
    assert target == {"path_label": "todo.md", "exists": True, "size_bytes": 3}


def test_another_tools_target_is_null(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target = approval_json(request("shell_exec", "ls", str(workspace)))["target"]
    assert target is None
