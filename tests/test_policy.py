"""Table-driven tests for policy.py. No I/O, no model, no clock: `today` is injected."""

from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import policy
from models import ActionType, Customer, Decision, Outcome, ReasonCode as RC, ReturnReason as RR

GUEST_GINA = "gina.guest@example.com"
GUEST_GARY = "gary.guest@example.com"


@pytest.fixture
def refund(catalog, today):
    def _call(customer, order, sku, reason=RR.CHANGED_MIND, prior=()):
        return policy.check_refund(
            customer, order, sku, reason=reason, today=today, catalog=catalog, prior_actions=prior
        )

    return _call


@pytest.fixture
def exchange(catalog, today):
    def _call(customer, order, sku, new_sku, reason=RR.WRONG_SIZE, prior=()):
        return policy.check_exchange(
            customer, order, sku, new_sku,
            reason=reason, today=today, catalog=catalog, prior_actions=prior,
        )  # fmt: skip

    return _call


@pytest.fixture
def cancel():
    def _call(customer, order, prior=()):
        return policy.check_cancellation(customer, order, prior_actions=prior)

    return _call


def assert_ok(decision: Decision):
    assert decision.allowed and decision.code is RC.OK, decision


def assert_denied(decision: Decision, code: RC):
    assert decision.outcome is Outcome.DENIED and decision.code is code, decision
    assert not decision.allowed and not decision.needs_human
    assert decision.reason


def assert_needs_human(decision: Decision):
    assert decision.outcome is Outcome.NEEDS_HUMAN and decision.code is RC.DEFECT_UNVERIFIED
    assert decision.needs_human and not decision.allowed
    assert "human" in decision.reason


# ------------------------------------------------------------------------- Decision


def test_decision_outcome_must_match_code():
    with pytest.raises(ValidationError):
        Decision(outcome=Outcome.ALLOWED, code=RC.NOT_DELIVERED, reason="x")
    with pytest.raises(ValidationError):
        Decision(outcome=Outcome.DENIED, code=RC.OK, reason="x")
    with pytest.raises(ValidationError):  # needs_human is only for unverified defects
        Decision(outcome=Outcome.NEEDS_HUMAN, code=RC.OUTSIDE_WINDOW, reason="x")
    with pytest.raises(ValidationError):  # and an unverified defect is never "denied"
        Decision(outcome=Outcome.DENIED, code=RC.DEFECT_UNVERIFIED, reason="x")


# --------------------------------------------------------------------------- refund

# (customer key, order, sku, reason, expected)
REFUND_CASES = [
    ("C001", "ORD-1001", "TEE-BLK-M", RR.CHANGED_MIND, RC.OK),  # 15 days, regular item
    ("C001", "ORD-1002", "HOOD-GRY-M", RR.CHANGED_MIND, RC.OUTSIDE_WINDOW),  # 50 days
    ("C001", "ORD-1002", "HOOD-GRY-M", RR.DEFECTIVE, RC.OUTSIDE_WINDOW),  # defect != longer window
    ("C002", "ORD-1003", "JKT-OLV-M", RR.CHANGED_MIND, RC.SALE_ITEM_EXCHANGE_ONLY),
    ("C002", "ORD-1003", "JKT-OLV-M", RR.WRONG_SIZE, RC.SALE_ITEM_EXCHANGE_ONLY),
    ("C002", "ORD-1003", "JKT-OLV-M", RR.OTHER, RC.SALE_ITEM_EXCHANGE_ONLY),
    ("C002", "ORD-1003", "JKT-OLV-M", RR.DEFECTIVE, RC.DEFECT_UNVERIFIED),  # sale + defective
    ("C001", "ORD-1001", "TEE-BLK-M", RR.DEFECTIVE, RC.OK),  # non-sale defective: normal refund
    ("C002", "ORD-1004", "JEAN-IND-32", RR.CHANGED_MIND, RC.NOT_DELIVERED),  # shipped
    ("C003", "ORD-1005", "SNK-WHT-9", RR.CHANGED_MIND, RC.NOT_DELIVERED),  # processing
    ("C003", "ORD-1006", "HOOD-GRY-L", RR.CHANGED_MIND, RC.OK),
    ("C003", "ORD-1006", "CAP-NVY-OS", RR.CHANGED_MIND, RC.OK),
    ("C003", "ORD-1006", "TEE-BLK-M", RR.CHANGED_MIND, RC.ITEM_NOT_IN_ORDER),
    ("C004", "ORD-1007", "JEAN-IND-34", RR.CHANGED_MIND, RC.ALREADY_REFUNDED),  # flag in data
    ("C004", "ORD-1008", "TEE-BLK-L", RR.CHANGED_MIND, RC.OK),  # exactly day 30
    ("C004", "ORD-1009", "HOOD-GRY-M", RR.CHANGED_MIND, RC.OUTSIDE_WINDOW),  # exactly day 31
    ("C002", "ORD-1001", "TEE-BLK-M", RR.CHANGED_MIND, RC.ORDER_NOT_OWNED),  # someone else's
    ("C001", "ORD-2001", "TEE-BLK-M", RR.CHANGED_MIND, RC.ORDER_NOT_OWNED),  # a guest's order
    (GUEST_GINA, "ORD-2001", "TEE-BLK-M", RR.CHANGED_MIND, RC.OK),
    (GUEST_GINA, "ORD-1001", "TEE-BLK-M", RR.CHANGED_MIND, RC.ORDER_NOT_OWNED),
    (GUEST_GINA, "ORD-2003", "SNK-WHT-9", RR.CHANGED_MIND, RC.ORDER_NOT_OWNED),  # other guest
    (GUEST_GINA, "ORD-2002", "SCARF-WOOL", RR.CHANGED_MIND, RC.NOT_DELIVERED),
    (GUEST_GARY, "ORD-2003", "SNK-WHT-9", RR.CHANGED_MIND, RC.OUTSIDE_WINDOW),  # 71 days
]


@pytest.mark.parametrize("who,order_id,sku,reason,expected", REFUND_CASES)
def test_refund_scenarios(refund, customers, orders, who, order_id, sku, reason, expected):
    decision = refund(customers[who], orders[order_id], sku, reason)
    if expected is RC.OK:
        assert_ok(decision)
    elif expected is RC.DEFECT_UNVERIFIED:
        assert_needs_human(decision)
    else:
        assert_denied(decision, expected)


def test_refund_sale_item_denial_offers_exchange(refund, customers, orders):
    d = refund(customers["C002"], orders["ORD-1003"], "JKT-OLV-M")
    assert d.alternatives == (ActionType.EXCHANGE,)
    assert "Autumn Clearance" in d.reason


def test_defective_sale_item_is_never_auto_approved(refund, customers, orders):
    d = refund(customers["C002"], orders["ORD-1003"], "JKT-OLV-M", RR.DEFECTIVE)
    assert not d.allowed
    assert_needs_human(d)


def test_defective_sale_item_outside_window_is_denied_not_escalated(
    refund, customers, make_order, today
):
    order = make_order(
        customer_id="C002",
        customer_email="ben.lim@example.com",
        delivered_on=today - timedelta(days=31),
        items=[{"sku": "JKT-OLV-M", "quantity": 1, "unit_price": "84.00"}],
    )
    assert_denied(refund(customers["C002"], order, "JKT-OLV-M", RR.DEFECTIVE), RC.OUTSIDE_WINDOW)


def test_defective_sale_item_already_refunded_is_denied_not_escalated(
    refund, customers, orders, record
):
    prior = [record(ActionType.REFUND, "ORD-1003", "C002")]
    d = refund(customers["C002"], orders["ORD-1003"], "JKT-OLV-M", RR.DEFECTIVE, prior)
    assert_denied(d, RC.ALREADY_REFUNDED)


def test_defective_sale_item_of_someone_else_is_not_owned(refund, customers, orders):
    d = refund(customers["C001"], orders["ORD-1003"], "JKT-OLV-M", RR.DEFECTIVE)
    assert_denied(d, RC.ORDER_NOT_OWNED)


def test_defective_sale_item_can_still_be_exchanged(exchange, customers, orders):
    assert_ok(
        exchange(customers["C002"], orders["ORD-1003"], "JKT-OLV-M", "JKT-OLV-L", RR.DEFECTIVE)
    )


@pytest.mark.parametrize(
    "days,allowed",
    [(0, True), (1, True), (29, True), (30, True), (31, False), (32, False), (365, False)],
)
def test_refund_window_boundary(refund, customers, make_order, today, days, allowed):
    order = make_order(delivered_on=today - timedelta(days=days))
    d = refund(customers["C001"], order, "TEE-BLK-M")
    assert_ok(d) if allowed else assert_denied(d, RC.OUTSIDE_WINDOW)


def test_delivery_date_in_the_future_counts_as_not_delivered(refund, customers, make_order, today):
    order = make_order(delivered_on=today + timedelta(days=1))
    assert_denied(refund(customers["C001"], order, "TEE-BLK-M"), RC.NOT_DELIVERED)


def test_window_is_measured_from_delivery_not_order_date(refund, customers, make_order, today):
    order = make_order(placed_on=today - timedelta(days=200), delivered_on=today - timedelta(days=2))
    assert_ok(refund(customers["C001"], order, "TEE-BLK-M"))


def test_refund_unknown_product_fails_closed(refund, customers, make_order):
    order = make_order(items=[{"sku": "ZZZ-GONE", "quantity": 1, "unit_price": "10.00"}])
    assert_denied(refund(customers["C001"], order, "ZZZ-GONE"), RC.UNKNOWN_PRODUCT)


def test_refund_blocked_by_prior_actions(refund, customers, orders, record):
    alice, order = customers["C001"], orders["ORD-1001"]
    assert_denied(
        refund(alice, order, "TEE-BLK-M", prior=[record(ActionType.REFUND, "ORD-1001")]),
        RC.ALREADY_REFUNDED,
    )
    assert_denied(
        refund(alice, order, "TEE-BLK-M", prior=[record(ActionType.EXCHANGE, "ORD-1001")]),
        RC.ALREADY_EXCHANGED,
    )
    assert_denied(
        refund(alice, order, "TEE-BLK-M", prior=[record(ActionType.CANCELLATION, "ORD-1001")]),
        RC.ALREADY_CANCELLED,
    )


def test_refund_not_blocked_by_unrelated_prior_actions(refund, customers, orders, record):
    prior = [
        record(ActionType.REFUND, "ORD-1002"),  # a different order
        record(ActionType.ESCALATION, "ORD-1001"),  # escalating does not settle an order
    ]
    assert_ok(refund(customers["C001"], orders["ORD-1001"], "TEE-BLK-M", prior=prior))


# ------------------------------------------------------------------------ ownership


def test_ownership_is_checked_first_and_leaks_nothing(refund, customers, orders):
    # ORD-1002 is out of window AND not Ben's: ownership must win, and the reason
    # must not reveal anything about the order (product, age).
    d = refund(customers["C002"], orders["ORD-1002"], "HOOD-GRY-M")
    assert_denied(d, RC.ORDER_NOT_OWNED)
    assert "HOOD" not in d.reason and "days" not in d.reason


def test_guest_using_a_registered_customers_email_does_not_own_their_orders(
    refund, orders
):
    impostor = Customer(customer_id=None, email="alice.tan@example.com")
    assert_denied(refund(impostor, orders["ORD-1001"], "TEE-BLK-M"), RC.ORDER_NOT_OWNED)


def test_guest_email_match_is_case_insensitive(refund, orders):
    guest = Customer(customer_id=None, email="  Gina.Guest@Example.COM ")
    assert_ok(refund(guest, orders["ORD-2001"], "TEE-BLK-M"))


def test_registered_customer_with_guest_orders_email_does_not_own_them(refund, orders):
    # Different customer_id, same email as a guest order: still not theirs.
    other = Customer(customer_id="C777", email=GUEST_GINA)
    assert_denied(refund(other, orders["ORD-2001"], "TEE-BLK-M"), RC.ORDER_NOT_OWNED)


# ---------------------------------------------------------------------------- exchange

EXCHANGE_CASES = [
    ("C001", "ORD-1001", "TEE-BLK-M", "TEE-BLK-L", RC.OK),
    ("C002", "ORD-1003", "JKT-OLV-M", "JKT-OLV-L", RC.OK),  # sale items CAN be exchanged
    ("C001", "ORD-1002", "HOOD-GRY-M", "HOOD-GRY-L", RC.OUTSIDE_WINDOW),
    ("C002", "ORD-1004", "JEAN-IND-32", "JEAN-IND-34", RC.NOT_DELIVERED),
    ("C004", "ORD-1007", "JEAN-IND-34", "JEAN-IND-32", RC.ALREADY_REFUNDED),
    ("C004", "ORD-1008", "TEE-BLK-L", "TEE-BLK-M", RC.OK),  # exactly day 30
    ("C001", "ORD-1003", "JKT-OLV-M", "JKT-OLV-L", RC.ORDER_NOT_OWNED),
    ("C003", "ORD-1006", "TEE-BLK-M", "TEE-BLK-L", RC.ITEM_NOT_IN_ORDER),
    (GUEST_GINA, "ORD-2001", "TEE-BLK-M", "TEE-BLK-L", RC.OK),
    (GUEST_GARY, "ORD-2001", "TEE-BLK-M", "TEE-BLK-L", RC.ORDER_NOT_OWNED),
    # replacement problems
    ("C001", "ORD-1001", "TEE-BLK-M", "TEE-BLK-M", RC.INVALID_EXCHANGE_TARGET),  # same item
    ("C001", "ORD-1001", "TEE-BLK-M", "HOOD-GRY-M", RC.INVALID_EXCHANGE_TARGET),  # other product
    ("C001", "ORD-1001", "TEE-BLK-M", "NOPE-123", RC.INVALID_EXCHANGE_TARGET),  # not in catalogue
]


@pytest.mark.parametrize("who,order_id,sku,new_sku,expected", EXCHANGE_CASES)
def test_exchange_scenarios(exchange, customers, orders, who, order_id, sku, new_sku, expected):
    decision = exchange(customers[who], orders[order_id], sku, new_sku)
    if expected is RC.OK:
        assert_ok(decision)
    else:
        assert_denied(decision, expected)


def test_exchange_to_out_of_stock_variant_denied(exchange, customers, make_order):
    order = make_order(items=[{"sku": "SNK-WHT-9", "quantity": 1, "unit_price": "90.00"}])
    d = exchange(customers["C001"], order, "SNK-WHT-9", "SNK-WHT-10")
    assert_denied(d, RC.INVALID_EXCHANGE_TARGET)
    assert "out of stock" in d.reason


def test_exchange_reason_does_not_change_eligibility(exchange, customers, orders):
    for reason in RR:
        assert_ok(exchange(customers["C002"], orders["ORD-1003"], "JKT-OLV-M", "JKT-OLV-L", reason))


@pytest.mark.parametrize("days,allowed", [(30, True), (31, False)])
def test_exchange_window_boundary(exchange, customers, make_order, today, days, allowed):
    order = make_order(delivered_on=today - timedelta(days=days))
    d = exchange(customers["C001"], order, "TEE-BLK-M", "TEE-BLK-L")
    assert_ok(d) if allowed else assert_denied(d, RC.OUTSIDE_WINDOW)


def test_exchange_blocked_by_prior_actions(exchange, customers, orders, record):
    alice, order = customers["C001"], orders["ORD-1001"]
    args = (alice, order, "TEE-BLK-M", "TEE-BLK-L")
    assert_denied(
        exchange(*args, prior=[record(ActionType.EXCHANGE, "ORD-1001")]), RC.ALREADY_EXCHANGED
    )
    assert_denied(
        exchange(*args, prior=[record(ActionType.REFUND, "ORD-1001")]), RC.ALREADY_REFUNDED
    )
    assert_denied(
        exchange(*args, prior=[record(ActionType.CANCELLATION, "ORD-1001")]), RC.ALREADY_CANCELLED
    )


# ------------------------------------------------------------------------ cancellation

CANCEL_CASES = [
    ("C003", "ORD-1005", RC.OK),  # processing
    (GUEST_GINA, "ORD-2002", RC.OK),  # processing guest order
    ("C002", "ORD-1004", RC.NOT_CANCELLABLE),  # shipped
    ("C001", "ORD-1001", RC.NOT_CANCELLABLE),  # delivered
    ("C001", "ORD-1005", RC.ORDER_NOT_OWNED),
    (GUEST_GARY, "ORD-2002", RC.ORDER_NOT_OWNED),
]


@pytest.mark.parametrize("who,order_id,expected", CANCEL_CASES)
def test_cancellation_scenarios(cancel, customers, orders, who, order_id, expected):
    decision = cancel(customers[who], orders[order_id])
    if expected is RC.OK:
        assert_ok(decision)
    else:
        assert_denied(decision, expected)


def test_cancellation_alternatives(cancel, customers, orders):
    delivered = cancel(customers["C001"], orders["ORD-1001"])
    assert set(delivered.alternatives) == {ActionType.REFUND, ActionType.EXCHANGE}
    assert cancel(customers["C002"], orders["ORD-1004"]).alternatives == ()  # shipped


def test_cancellation_blocked_when_already_cancelled(cancel, customers, orders, record):
    prior = [record(ActionType.CANCELLATION, "ORD-1005", "C003")]
    assert_denied(cancel(customers["C003"], orders["ORD-1005"], prior), RC.ALREADY_CANCELLED)


def test_cancelled_order_cannot_then_be_refunded(refund, customers, orders, record):
    # ORD-1005 is still "processing" in data; the cancellation lives in state.
    prior = [record(ActionType.CANCELLATION, "ORD-1005", "C003")]
    d = refund(customers["C003"], orders["ORD-1005"], "SNK-WHT-9", prior=prior)
    assert_denied(d, RC.ALREADY_CANCELLED)


# --------------------------------------------------------------------------- purity


def test_policy_module_is_pure():
    """policy.py may not import anything with side effects or read the clock."""
    tree = ast.parse(Path(policy.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"__future__", "collections.abc", "datetime", "models"}, imported
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attributes & {"today", "now", "utcnow"}  # the clock is injected


def test_checks_do_not_mutate_inputs(refund, customers, orders, record):
    order = orders["ORD-1001"]
    before = order.model_dump()
    prior = [record(ActionType.ESCALATION, "ORD-1001")]
    refund(customers["C001"], order, "TEE-BLK-M", prior=prior)
    assert order.model_dump() == before and len(prior) == 1
