"""Executes requests and records them in state. The only module that writes actions.

Every request does the same three things, in this order:
  1. idempotency: if this customer already has this action on this order, report
     "already done" and change nothing;
  2. policy: ask policy.py (pure) whether the request is allowed;
  3. only if allowed: execute and write the record to state (atomically).

No rule logic lives here; this module never decides eligibility itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import policy
import store
from models import (
    ActionRecord,
    ActionType,
    Customer,
    Decision,
    Order,
    Product,
    ReturnReason,
    ReasonCode,
    State,
    idempotency_key,
)


class Status(StrEnum):
    EXECUTED = "executed"  # a new record was written
    ALREADY_DONE = "already_done"  # identical request seen before; nothing changed
    DENIED = "denied"  # policy said no
    NEEDS_HUMAN = "needs_human"  # policy cannot decide; escalated to a person
    NEEDS_INFO = "needs_info"  # a required detail is missing; nothing recorded


@dataclass(frozen=True)
class ActionOutcome:
    requested: ActionType
    status: Status
    record: ActionRecord | None = None  # the record written, or the earlier one on a repeat
    decision: Decision | None = None
    is_new: bool = False  # True only if this call wrote a new record


@dataclass
class ActionContext:
    customer: Customer  # set by the caller from --customer, never by the model
    catalog: Mapping[str, Product]
    state: State
    state_path: Path
    today: date
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def prior_actions(self) -> tuple[ActionRecord, ...]:
        return tuple(self.state.executed_actions.values())


# --------------------------------------------------------------------------- internals


def _find(ctx: ActionContext, order_id: str | None, action: ActionType, sku: str | None = None) -> ActionRecord | None:
    # Scan record facts as well as supporting keys written before line-level idempotency.
    return next((r for r in ctx.prior_actions()
                 if r.customer_key == ctx.customer.key and r.order_id == order_id
                 and r.action is action and (sku is None or r.sku == sku)), None)


def _write(
    ctx: ActionContext,
    action: ActionType,
    *,
    order_id: str | None,
    sku: str | None = None,
    reason: str | None = None,
    detail: dict[str, Any] | None = None,
) -> ActionRecord:
    key = idempotency_key(ctx.customer.key, order_id, action, sku)
    if key in ctx.state.executed_actions:  # callers check first; this is the backstop
        raise RuntimeError(f"refusing to overwrite executed action {key}")
    record = ActionRecord(
        idempotency_key=key,
        action=action,
        customer_key=ctx.customer.key,
        order_id=order_id,
        sku=sku,
        reason=reason,
        detail=detail or {},
        as_of=ctx.today,
        executed_at=ctx.now(),
    )
    ctx.state.executed_actions[key] = record
    try:
        store.save_state(ctx.state_path, ctx.state)
    except Exception:
        del ctx.state.executed_actions[key]  # keep memory and disk in agreement
        raise
    return record


def _already(action: ActionType, record: ActionRecord) -> ActionOutcome:
    return ActionOutcome(action, Status.ALREADY_DONE, record=record)


def _denied(action: ActionType, decision: Decision) -> ActionOutcome:
    return ActionOutcome(action, Status.DENIED, decision=decision)


# ---------------------------------------------------------------------------- requests


def refund(
    ctx: ActionContext,
    order: Order,
    sku: str,
    reason: ReturnReason,
    defect_description: str | None = None,
) -> ActionOutcome:
    if not policy.owns_order(ctx.customer, order):
        return _denied(ActionType.REFUND, Decision.deny(ReasonCode.ORDER_NOT_OWNED, "Order not found on this account."))
    if (existing := _find(ctx, order.order_id, ActionType.REFUND, sku)) is not None:
        return _already(ActionType.REFUND, existing)

    decision = policy.check_refund(
        ctx.customer,
        order,
        sku,
        reason=reason,
        today=ctx.today,
        catalog=ctx.catalog,
        prior_actions=ctx.prior_actions(),
    )

    if decision.needs_human:
        if not defect_description:
            return ActionOutcome(ActionType.REFUND, Status.NEEDS_INFO, decision=decision)
        escalation = escalate(
            ctx,
            f"Verify reported defect before any refund: {sku} on {order.order_id}. "
            f"Customer says: {defect_description}",
            order_id=order.order_id,
            sku=sku,
            trigger=decision.code.value,
            detail={"requested_action": "refund", "defect_description": defect_description},
        )
        return ActionOutcome(
            ActionType.REFUND,
            Status.NEEDS_HUMAN,
            record=escalation.record,
            decision=decision,
            is_new=escalation.is_new,
        )

    if not decision.allowed:
        return _denied(ActionType.REFUND, decision)

    item = order.item(sku)
    assert item is not None  # policy has confirmed the item is in the order
    amount = item.unit_price * item.quantity
    record = _write(
        ctx,
        ActionType.REFUND,
        order_id=order.order_id,
        sku=sku,
        reason=reason.value,
        detail={"amount": f"{amount:.2f}", "refund_to": "original payment method"},
    )
    return ActionOutcome(ActionType.REFUND, Status.EXECUTED, record=record, is_new=True)


def exchange(
    ctx: ActionContext, order: Order, sku: str, new_sku: str, reason: ReturnReason
) -> ActionOutcome:
    if not policy.owns_order(ctx.customer, order):
        return _denied(ActionType.EXCHANGE, Decision.deny(ReasonCode.ORDER_NOT_OWNED, "Order not found on this account."))
    if (existing := _find(ctx, order.order_id, ActionType.EXCHANGE, sku)) is not None:
        return _already(ActionType.EXCHANGE, existing)

    decision = policy.check_exchange(
        ctx.customer,
        order,
        sku,
        new_sku,
        reason=reason,
        today=ctx.today,
        catalog=ctx.catalog,
        prior_actions=ctx.prior_actions(),
    )
    if not decision.allowed:
        return _denied(ActionType.EXCHANGE, decision)

    record = _write(
        ctx,
        ActionType.EXCHANGE,
        order_id=order.order_id,
        sku=sku,
        reason=reason.value,
        detail={"new_sku": new_sku},
    )
    return ActionOutcome(ActionType.EXCHANGE, Status.EXECUTED, record=record, is_new=True)


def cancel(ctx: ActionContext, order: Order) -> ActionOutcome:
    if not policy.owns_order(ctx.customer, order):
        return _denied(ActionType.CANCELLATION, Decision.deny(ReasonCode.ORDER_NOT_OWNED, "Order not found on this account."))
    if (existing := _find(ctx, order.order_id, ActionType.CANCELLATION)) is not None:
        return _already(ActionType.CANCELLATION, existing)

    decision = policy.check_cancellation(ctx.customer, order, prior_actions=ctx.prior_actions())
    if not decision.allowed:
        return _denied(ActionType.CANCELLATION, decision)

    record = _write(ctx, ActionType.CANCELLATION, order_id=order.order_id, detail={})
    return ActionOutcome(ActionType.CANCELLATION, Status.EXECUTED, record=record, is_new=True)


def escalate(
    ctx: ActionContext,
    summary: str,
    *,
    order_id: str | None = None,
    sku: str | None = None,
    trigger: str = "agent_request",
    detail: dict[str, Any] | None = None,
) -> ActionOutcome:
    """Hand the case to a human. One escalation per customer+order: a repeat does not
    open a second ticket, but a *new* summary is appended to the existing one so the
    human sees it."""
    existing = _find(ctx, order_id, ActionType.ESCALATION)
    if existing is not None:
        known = [existing.detail.get("summary")] + [
            f.get("summary") for f in existing.detail.get("follow_ups", [])
        ]
        if summary not in known:
            follow_ups = [*existing.detail.get("follow_ups", []), {"summary": summary, **(detail or {})}]
            updated = existing.model_copy(
                update={"detail": {**existing.detail, "follow_ups": follow_ups}}
            )
            ctx.state.executed_actions[existing.idempotency_key] = updated
            try:
                store.save_state(ctx.state_path, ctx.state)
            except Exception:
                ctx.state.executed_actions[existing.idempotency_key] = existing
                raise
            existing = updated
        return _already(ActionType.ESCALATION, existing)

    record = _write(
        ctx,
        ActionType.ESCALATION,
        order_id=order_id,
        sku=sku,
        reason=trigger,
        detail={"summary": summary, **(detail or {})},
    )
    return ActionOutcome(ActionType.ESCALATION, Status.EXECUTED, record=record, is_new=True)


# ------------------------------------------------------------------------- presentation


def describe_record(record: ActionRecord) -> str:
    """One-line, human-readable description of an executed action."""
    where = record.order_id or "no order"
    match record.action:
        case ActionType.REFUND:
            return (
                f"refund - {where} {record.sku}, {record.detail.get('amount')} "
                "to the original payment method"
            )
        case ActionType.EXCHANGE:
            return f"exchange - {where} {record.sku} -> {record.detail.get('new_sku')}"
        case ActionType.CANCELLATION:
            return f"cancellation - {where}"
        case ActionType.ESCALATION:
            summary = str(record.detail.get("summary", ""))
            if len(summary) > 160:
                summary = summary[:160].rsplit(" ", 1)[0] + "..."
            return f"escalation - {where}: {summary}"
    raise AssertionError(record.action)
