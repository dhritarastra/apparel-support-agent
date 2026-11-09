"""Business rules, as pure functions.

Every check takes plain facts (customer, order, catalogue, prior actions, and the
business date `today`) and returns a `Decision`. Nothing here does I/O, reads the
clock, or mutates its inputs; the model never decides eligibility, this module does.

Checks fail closed and run in a fixed order. Ownership always comes first, so a
customer probing someone else's order learns nothing about it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from models import (
    ActionRecord,
    ActionType,
    Customer,
    Decision,
    Order,
    OrderStatus,
    Product,
    ReasonCode,
    ReturnReason,
)

RETURN_WINDOW_DAYS = 30  # counted from delivery; day 30 is still inside the window


# ---------------------------------------------------------------------------- helpers


def owns_order(customer: Customer, order: Order) -> bool:
    """Registered customers own orders carrying their customer_id. Guests own only
    guest orders (no customer_id) placed with their exact email."""
    if customer.customer_id is not None:
        return order.customer_id == customer.customer_id
    return order.customer_id is None and order.customer_email == customer.email


def days_since_delivery(order: Order, today: date) -> int | None:
    """Whole days since delivery, or None if the order has not been delivered
    (including a delivery date in the future)."""
    if order.status is not OrderStatus.DELIVERED or order.delivered_on is None:
        return None
    if order.delivered_on > today:
        return None
    return (today - order.delivered_on).days


def _has_prior(order: Order, prior_actions: Sequence[ActionRecord], action: ActionType, sku: str | None = None) -> bool:
    # Old records without a SKU conservatively settle the entire order.
    return any(r.order_id == order.order_id and r.action is action
               and (sku is None or r.sku is None or r.sku == sku) for r in prior_actions)


def _not_owned(order: Order) -> Decision:
    # Deliberately says nothing about the order itself.
    return Decision.deny(
        ReasonCode.ORDER_NOT_OWNED,
        f"Order {order.order_id} is not associated with this customer.",
    )


def _delivery_window_gate(order: Order, today: date) -> Decision | None:
    days = days_since_delivery(order, today)
    if days is None:
        alternatives = (
            (ActionType.CANCELLATION,) if order.status is OrderStatus.PROCESSING else ()
        )
        return Decision.deny(
            ReasonCode.NOT_DELIVERED,
            f"Order {order.order_id} has not been delivered yet (status: {order.status.value}).",
            alternatives,
        )
    if days > RETURN_WINDOW_DAYS:
        return Decision.deny(
            ReasonCode.OUTSIDE_WINDOW,
            f"Order {order.order_id} was delivered {days} days ago; refunds and exchanges "
            f"must be requested within {RETURN_WINDOW_DAYS} days of delivery.",
        )
    return None


def _prior_order_state_gate(
    order: Order, prior_actions: Sequence[ActionRecord], *, acting: ActionType, sku: str
) -> Decision | None:
    """Deny when an earlier action already settled this order."""
    if _has_prior(order, prior_actions, ActionType.CANCELLATION):
        return Decision.deny(
            ReasonCode.ALREADY_CANCELLED, f"Order {order.order_id} has already been cancelled."
        )
    if order.refunded or _has_prior(order, prior_actions, ActionType.REFUND, sku):
        return Decision.deny(
            ReasonCode.ALREADY_REFUNDED, f"Order {order.order_id} has already been refunded."
        )
    if _has_prior(order, prior_actions, ActionType.EXCHANGE, sku):
        return Decision.deny(
            ReasonCode.ALREADY_EXCHANGED,
            f"Order {order.order_id} has already been exchanged"
            + (" and cannot also be refunded." if acting is ActionType.REFUND else "."),
        )
    return None


# ----------------------------------------------------------------------------- checks


def check_refund(
    customer: Customer,
    order: Order,
    sku: str,
    *,
    reason: ReturnReason,
    today: date,
    catalog: Mapping[str, Product],
    prior_actions: Sequence[ActionRecord] = (),
) -> Decision:
    if not owns_order(customer, order):
        return _not_owned(order)

    if order.item(sku) is None:
        return Decision.deny(
            ReasonCode.ITEM_NOT_IN_ORDER, f"Order {order.order_id} does not contain item {sku}."
        )
    product = catalog.get(sku)
    if product is None:
        return Decision.deny(
            ReasonCode.UNKNOWN_PRODUCT, f"Item {sku} has no catalogue record; cannot assess it."
        )

    if (denied := _prior_order_state_gate(order, prior_actions, acting=ActionType.REFUND, sku=sku)):
        return denied
    if (denied := _delivery_window_gate(order, today)):
        return denied

    item = order.item(sku)
    if item.on_sale_at_purchase:
        if item.defect_verified:
            return Decision.allow("Verified defective sale item meets refund policy.")
        if reason is ReturnReason.DEFECTIVE:
            # A defect claim is customer-supplied and unverifiable here, so it never
            # auto-refunds a sale item: a human verifies it first.
            return Decision.needs_human_review(
                f"{product.name} was bought in the '{item.sale_name_at_purchase or 'recorded promotion'}' sale, so it can only be "
                "refunded if it is defective, and a human agent must verify the defect first."
            )
        return Decision.deny(
            ReasonCode.SALE_ITEM_EXCHANGE_ONLY,
            f"{product.name} was bought in the '{item.sale_name_at_purchase or 'recorded promotion'}' sale and is exchange-only "
            "unless it is defective.",
            (ActionType.EXCHANGE,),
        )
    return Decision.allow()


def check_exchange(
    customer: Customer,
    order: Order,
    sku: str,
    new_sku: str,
    *,
    reason: ReturnReason,  # recorded by the caller; does not change eligibility
    today: date,
    catalog: Mapping[str, Product],
    prior_actions: Sequence[ActionRecord] = (),
) -> Decision:
    if not owns_order(customer, order):
        return _not_owned(order)

    if order.item(sku) is None:
        return Decision.deny(
            ReasonCode.ITEM_NOT_IN_ORDER, f"Order {order.order_id} does not contain item {sku}."
        )
    product = catalog.get(sku)
    if product is None:
        return Decision.deny(
            ReasonCode.UNKNOWN_PRODUCT, f"Item {sku} has no catalogue record; cannot assess it."
        )

    if (denied := _prior_order_state_gate(order, prior_actions, acting=ActionType.EXCHANGE, sku=sku)):
        return denied
    if (denied := _delivery_window_gate(order, today)):
        return denied

    target = catalog.get(new_sku)
    if target is None:
        problem = f"{new_sku} is not a product we sell."
    elif new_sku == sku:
        problem = "the replacement is the same item as the one being exchanged."
    elif target.style != product.style:
        problem = f"{target.name} is a different product; exchanges must be the same product."
    elif not target.in_stock:
        problem = f"{target.name} is out of stock."
    else:
        return Decision.allow()
    return Decision.deny(
        ReasonCode.INVALID_EXCHANGE_TARGET,
        f"Cannot exchange {product.name}: {problem}",
        (ActionType.REFUND,) if not order.item(sku).on_sale_at_purchase else (),
    )


def check_cancellation(
    customer: Customer,
    order: Order,
    *,
    prior_actions: Sequence[ActionRecord] = (),
) -> Decision:
    if not owns_order(customer, order):
        return _not_owned(order)

    if _has_prior(order, prior_actions, ActionType.CANCELLATION):
        return Decision.deny(
            ReasonCode.ALREADY_CANCELLED, f"Order {order.order_id} has already been cancelled."
        )
    if order.status is not OrderStatus.PROCESSING:
        alternatives = (
            (ActionType.REFUND, ActionType.EXCHANGE)
            if order.status is OrderStatus.DELIVERED
            else ()
        )
        return Decision.deny(
            ReasonCode.NOT_CANCELLABLE,
            f"Order {order.order_id} is {order.status.value} and can no longer be cancelled; "
            "only processing orders can be.",
            alternatives,
        )
    return Decision.allow()
