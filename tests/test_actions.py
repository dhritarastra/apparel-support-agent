"""actions.py: idempotency, policy gating and state writes (no model involved)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import actions
import store
from actions import ActionContext, Status
from models import ActionType, ReasonCode, ReturnReason as RR, idempotency_key


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "state.json"


@pytest.fixture
def make_ctx(state_path, catalog, customers, today):
    """A fresh context loaded from disk, as a new process would see it."""

    def _make(who: str = "C001") -> ActionContext:
        return ActionContext(
            customer=customers[who],
            catalog=catalog,
            state=store.load_state(state_path),
            state_path=state_path,
            today=today,
            now=lambda: datetime(2026, 11, 9, 12, 0, tzinfo=UTC),
        )

    return _make


def records(state_path):
    return list(store.load_state(state_path).executed_actions.values())


def test_idempotency_key_format(customers):
    assert idempotency_key("C001", "ORD-1001", ActionType.REFUND) == "C001|ORD-1001|refund"
    assert customers["gina.guest@example.com"].key == "gina.guest@example.com"
    assert idempotency_key("C001", None, ActionType.ESCALATION) == "C001|-|escalation"


def test_refund_executes_and_persists(make_ctx, orders, state_path):
    out = actions.refund(make_ctx(), orders["ORD-1001"], "TEE-BLK-M", RR.CHANGED_MIND)
    assert out.status is Status.EXECUTED and out.is_new
    [rec] = records(state_path)
    assert rec.idempotency_key == "C001|ORD-1001|refund|TEE-BLK-M"
    assert rec.detail["amount"] == "25.00" and rec.sku == "TEE-BLK-M"


def test_repeated_refund_is_recognised_across_processes(make_ctx, orders, state_path):
    first = actions.refund(make_ctx(), orders["ORD-1001"], "TEE-BLK-M", RR.CHANGED_MIND)
    again = actions.refund(make_ctx(), orders["ORD-1001"], "TEE-BLK-M", RR.CHANGED_MIND)
    assert again.status is Status.ALREADY_DONE and not again.is_new
    assert again.record == first.record
    assert len(records(state_path)) == 1  # never double-executed


def test_denied_request_writes_nothing(make_ctx, orders, state_path):
    out = actions.refund(make_ctx("C002"), orders["ORD-1003"], "JKT-OLV-M", RR.CHANGED_MIND)
    assert out.status is Status.DENIED
    assert out.decision.code is ReasonCode.SALE_ITEM_EXCHANGE_ONLY
    assert not state_path.exists()


def test_other_customers_order_is_denied(make_ctx, orders, state_path):
    out = actions.refund(make_ctx("C002"), orders["ORD-1001"], "TEE-BLK-M", RR.CHANGED_MIND)
    assert out.status is Status.DENIED and out.decision.code is ReasonCode.ORDER_NOT_OWNED
    assert not state_path.exists()


def test_defective_sale_item_needs_description_first(make_ctx, orders, state_path):
    out = actions.refund(make_ctx("C002"), orders["ORD-1003"], "JKT-OLV-M", RR.DEFECTIVE)
    assert out.status is Status.NEEDS_INFO and out.record is None
    assert not state_path.exists()


def test_defective_sale_item_escalates_and_never_refunds(make_ctx, orders, state_path):
    out = actions.refund(
        make_ctx("C002"), orders["ORD-1003"], "JKT-OLV-M", RR.DEFECTIVE, "zip broke on day 2"
    )
    assert out.status is Status.NEEDS_HUMAN and out.is_new
    assert out.decision.code is ReasonCode.DEFECT_UNVERIFIED
    [rec] = records(state_path)
    assert rec.action is ActionType.ESCALATION and rec.reason == "defect_unverified"
    assert rec.detail["defect_description"] == "zip broke on day 2"
    assert rec.detail["requested_action"] == "refund"

    again = actions.refund(
        make_ctx("C002"), orders["ORD-1003"], "JKT-OLV-M", RR.DEFECTIVE, "zip broke on day 2"
    )
    assert again.status is Status.NEEDS_HUMAN and not again.is_new
    [rec] = records(state_path)  # still one escalation, still no refund
    assert "follow_ups" not in rec.detail


def test_non_sale_defective_item_refunds_normally(make_ctx, orders, state_path):
    out = actions.refund(make_ctx(), orders["ORD-1001"], "TEE-BLK-M", RR.DEFECTIVE, "seam split")
    assert out.status is Status.EXECUTED
    assert [r.action for r in records(state_path)] == [ActionType.REFUND]


def test_refund_after_exchange_is_denied(make_ctx, orders, state_path):
    ex = actions.exchange(make_ctx(), orders["ORD-1001"], "TEE-BLK-M", "TEE-BLK-L", RR.WRONG_SIZE)
    assert ex.status is Status.EXECUTED
    rf = actions.refund(make_ctx(), orders["ORD-1001"], "TEE-BLK-M", RR.CHANGED_MIND)
    assert rf.status is Status.DENIED and rf.decision.code is ReasonCode.ALREADY_EXCHANGED
    assert len(records(state_path)) == 1


def test_cancel_then_repeat_then_refund(make_ctx, orders, state_path):
    order = orders["ORD-1005"]
    assert actions.cancel(make_ctx("C003"), order).status is Status.EXECUTED
    assert actions.cancel(make_ctx("C003"), order).status is Status.ALREADY_DONE
    rf = actions.refund(make_ctx("C003"), order, "SNK-WHT-9", RR.CHANGED_MIND)
    assert rf.decision.code is ReasonCode.ALREADY_CANCELLED
    assert len(records(state_path)) == 1


def test_escalation_dedupes_but_keeps_new_information(make_ctx, state_path):
    first = actions.escalate(make_ctx(), "wants an exception", order_id="ORD-1002")
    assert first.status is Status.EXECUTED
    same = actions.escalate(make_ctx(), "wants an exception", order_id="ORD-1002")
    assert same.status is Status.ALREADY_DONE
    assert "follow_ups" not in records(state_path)[0].detail
    newer = actions.escalate(make_ctx(), "now also asks about shipping", order_id="ORD-1002")
    assert newer.status is Status.ALREADY_DONE and not newer.is_new
    [rec] = records(state_path)
    assert rec.detail["follow_ups"] == [{"summary": "now also asks about shipping"}]


def test_failed_save_rolls_back_memory(make_ctx, orders, monkeypatch):
    ctx = make_ctx()

    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_state", boom)
    with pytest.raises(OSError):
        actions.refund(ctx, orders["ORD-1001"], "TEE-BLK-M", RR.CHANGED_MIND)
    assert ctx.state.executed_actions == {}


def test_corrupt_state_file_is_an_error_not_a_reset(state_path):
    state_path.write_text("{ not json")
    with pytest.raises(ValueError):
        store.load_state(state_path)


def test_failure_switch_blocks_order_reads(monkeypatch, customers):
    monkeypatch.setenv("FAIL_ORDER_LOOKUP", "1")
    with pytest.raises(store.OrderLookupError):
        store.fetch_order("ORD-1001")
    with pytest.raises(store.OrderLookupError):
        store.fetch_orders_for(customers["C001"])
