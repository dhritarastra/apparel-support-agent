from __future__ import annotations

from datetime import date, datetime

import pytest

import store
from models import ActionRecord, ActionType, Order, idempotency_key


@pytest.fixture(scope="session")
def today() -> date:
    return date(2026, 11, 9)  # the demo business date; test_data pins config to it


@pytest.fixture(scope="session")
def catalog():
    return store.load_catalog()


@pytest.fixture(scope="session")
def orders():
    return store.load_orders()


@pytest.fixture(scope="session")
def customers():
    """Keyed by customer_id, or by email for guests."""
    return {c.key: c for c in store.load_customers()}


@pytest.fixture
def make_order():
    """Build a one-item delivered order, overriding any field."""

    def _make(**overrides) -> Order:
        base = dict(
            order_id="ORD-T1",
            customer_id="C001",
            customer_email="alice.tan@example.com",
            status="delivered",
            placed_on=date(2026, 1, 1),
            delivered_on=date(2026, 10, 10),
            items=[{"sku": "TEE-BLK-M", "quantity": 1, "unit_price": "25.00"}],
        )
        base.update(overrides)
        catalog = store.load_catalog()
        base['items'] = [{**i, 'on_sale_at_purchase': i.get('on_sale_at_purchase', catalog.get(i['sku']).on_sale if i['sku'] in catalog else False),
                          'sale_name_at_purchase': i.get('sale_name_at_purchase', catalog.get(i['sku']).sale_name if i['sku'] in catalog else None)} for i in base['items']]
        return Order.model_validate(base)

    return _make


@pytest.fixture
def record(today):
    """Build a prior ActionRecord for an order."""

    def _record(action: ActionType, order_id: str, customer_key: str = "C001") -> ActionRecord:
        return ActionRecord(
            idempotency_key=idempotency_key(customer_key, order_id, action),
            action=action,
            customer_key=customer_key,
            order_id=order_id,
            as_of=today,
            executed_at=datetime(2026, 11, 9, 9, 0, 0),
        )

    return _record
