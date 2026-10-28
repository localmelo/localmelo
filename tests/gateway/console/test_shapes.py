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
