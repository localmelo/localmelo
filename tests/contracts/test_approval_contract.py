"""The approval contract: requests, decisions, the built-in policies and texts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from localmelo.melo import contracts
from localmelo.melo.contracts.approvals import (
    APPROVAL_FAILED,
    DENIAL_ENDING,
    AllowAllPolicy,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    DenyAllPolicy,
    denial_text,
)
from localmelo.melo.contracts.review import ActionReview

ENDING = (
    "Do not run it again or reach the same result another way unless the user asks."
)
TIMEOUT_TEXT = (
    "The call was not approved in time, so it did not run. Silence is not "
    "consent: do not retry it without asking the user."
)


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        request_id="0" * 32,
        run_id="run",
        call_id="call_1",
        tool_name="shell_exec",
        arguments={"command": "ls"},
        preview="ls",
        workspace_root=None,
    )


class TestApprovalDecision:
    def test_defaults(self) -> None:
        decision = ApprovalDecision(True)
        assert (decision.reason, decision.remember, decision.origin) == (
            "",
            False,
            "user",
        )

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"allow": 1}, "ApprovalDecision.allow must be bool, got 1"),
            (
                {"allow": False, "reason": None},
                "ApprovalDecision.reason must be str, got None",
            ),
            (
                {"allow": True, "remember": "yes"},
                "ApprovalDecision.remember must be bool, got 'yes'",
            ),
            (
                {"allow": True, "origin": 3},
                "ApprovalDecision.origin must be str, got 3",
            ),
        ],
    )
    def test_a_field_of_the_wrong_type_is_refused(
        self, fields: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(TypeError) as caught:
            ApprovalDecision(**fields)
        assert str(caught.value) == message

    @pytest.mark.parametrize(
        "fields",
        [
            {"allow": True, "origin": "grant"},
            {"allow": False, "remember": True},
            {"allow": False, "origin": "session"},
            {"allow": False, "origin": "earlier"},
            {"allow": True, "origin": "timeout"},
        ],
    )
    def test_refused_values_and_pairs(self, fields: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ApprovalDecision(**fields)

    @pytest.mark.parametrize(
        "fields",
        [
            {"allow": True, "remember": True},
            {"allow": True, "origin": "session"},
            {"allow": True, "remember": True, "origin": "session"},
            {"allow": True, "origin": "earlier"},
            {"allow": False, "origin": "timeout"},
            {"allow": False, "reason": "no", "origin": "policy"},
        ],
    )
    def test_allowed_pairs(self, fields: dict[str, Any]) -> None:
        ApprovalDecision(**fields)

    def test_it_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            ApprovalDecision(True).allow = False  # type: ignore[misc]


class TestDenialText:
    @pytest.mark.parametrize(
        ("origin", "who"),
        [("user", "The user"), ("policy", "The approval policy")],
    )
    def test_without_a_reason(self, origin: Any, who: str) -> None:
        for reason in ("", "   ", " . "):
            decision = ApprovalDecision(False, reason, origin=origin)
            assert denial_text(decision) == f"{who} denied this call. {ENDING}"

    @pytest.mark.parametrize(
        ("origin", "who"),
        [("user", "The user"), ("policy", "The approval policy")],
    )
    def test_with_a_reason(self, origin: Any, who: str) -> None:
        for reason in ("not now", "  not now.  ", "not now."):
            decision = ApprovalDecision(False, reason, origin=origin)
            assert denial_text(decision) == (
                f"{who} denied this call: not now. {ENDING}"
            )

    def test_only_one_final_period_goes(self) -> None:
        decision = ApprovalDecision(False, "wait...")
        assert denial_text(decision) == f"The user denied this call: wait... {ENDING}"

    def test_a_long_reason_keeps_500_characters(self) -> None:
        decision = ApprovalDecision(False, "r" * 600)
        assert denial_text(decision) == (
            f"The user denied this call: {'r' * 500}. {ENDING}"
        )

    def test_a_timeout_ignores_the_reason(self) -> None:
        for reason in ("", "No answer within 60 s"):
            decision = ApprovalDecision(False, reason, origin="timeout")
            assert denial_text(decision) == TIMEOUT_TEXT

    def test_an_allowed_call_has_no_denial_text(self) -> None:
        with pytest.raises(ValueError):
            denial_text(ApprovalDecision(True))

    def test_a_high_impact_review_says_why_and_what_to_do(self) -> None:
        review = ActionReview(
            "high", ("install", "env"), ("Installs x with pip.", "Creates a venv.")
        )
        because = "installs x with pip; creates a venv"
        assert denial_text(ApprovalDecision(False), review) == (
            f"The user denied this call. melo asked because it {because}. {ENDING} "
            "If the task needs it, say in your answer what for and ask the user."
        )
        policy = ApprovalDecision(False, "no one can approve", origin="policy")
        assert denial_text(policy, review) == (
            f"The approval policy denied this call: no one can approve. It {because}. "
            f"{ENDING} Finish what you can without it, and say in your answer what "
            "you would need."
        )
        timeout = ApprovalDecision(False, origin="timeout")
        assert denial_text(timeout, review) == TIMEOUT_TEXT

    def test_a_routine_or_undecided_review_adds_nothing(self) -> None:
        for review in (ActionReview("routine"), ActionReview("undecided", (), ("x",))):
            decision = ApprovalDecision(False)
            assert denial_text(decision, review) == denial_text(decision)

    def test_the_constants(self) -> None:
        assert DENIAL_ENDING == ENDING
        assert APPROVAL_FAILED.format(error="RuntimeError: boom") == (
            "Approval failed, so the call did not run: RuntimeError: boom"
        )


class TestApprovalRequest:
    def test_the_tier_defaults_to_none(self) -> None:
        # Only a caller that builds the request itself leaves it out; an
        # agent always fills it.
        assert _request().tier is None
        assert (
            _request().request_id,
            _request().tool_name,
            _request().arguments,
        ) == ("0" * 32, "shell_exec", {"command": "ls"})

    @pytest.mark.parametrize("tier", ["read", "write", "exec", "network"])
    def test_the_four_tiers_are_accepted(self, tier: Any) -> None:
        request = ApprovalRequest(
            request_id="0" * 32,
            run_id="run",
            call_id="call_1",
            tool_name="shell_exec",
            arguments={"command": "ls"},
            preview="ls",
            workspace_root=None,
            tier=tier,
        )
        assert request.tier == tier


class TestPolicies:
    @pytest.mark.asyncio
    async def test_allow_all(self) -> None:
        policy = AllowAllPolicy()
        assert isinstance(policy, ApprovalPolicy)
        assert await policy.decide(_request()) == ApprovalDecision(
            True, origin="policy"
        )

    @pytest.mark.asyncio
    async def test_deny_all(self) -> None:
        assert await DenyAllPolicy().decide(_request()) == ApprovalDecision(
            False, origin="policy"
        )
        policy = DenyAllPolicy("approvals are set to deny")
        assert isinstance(policy, ApprovalPolicy)
        decision = await policy.decide(_request())
        assert decision == ApprovalDecision(
            False, "approvals are set to deny", origin="policy"
        )
        assert denial_text(decision) == (
            "The approval policy denied this call: approvals are set to deny. "
            f"{ENDING}"
        )

    def test_an_object_without_decide_is_no_policy(self) -> None:
        assert not isinstance(object(), ApprovalPolicy)

    def test_the_names_are_exported(self) -> None:
        for name in (
            "APPROVAL_FAILED",
            "DENIAL_ENDING",
            "AllowAllPolicy",
            "ApprovalDecision",
            "ApprovalOrigin",
            "ApprovalPolicy",
            "ApprovalRequest",
            "DenyAllPolicy",
            "denial_text",
        ):
            assert name in contracts.__all__
            assert hasattr(contracts, name)
