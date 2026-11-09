"""Sanity checks on data/*.json and the demo clock."""

from __future__ import annotations

from datetime import date

import pytest

import config
import policy
import store
from models import OrderStatus


def test_default_clock_is_the_fixed_demo_date(today):
    assert config.DEFAULT_AS_OF == today == date(2026, 11, 9)


def test_as_of_env_override(monkeypatch):
    monkeypatch.delenv("AS_OF_DATE", raising=False)
    assert config.get_as_of() == date(2026, 11, 9)
    monkeypatch.setenv("AS_OF_DATE", "2027-01-15")
    assert config.get_as_of() == date(2027, 1, 15)
    monkeypatch.setenv("AS_OF_DATE", "15/01/2027")
    with pytest.raises(ValueError, match="AS_OF_DATE"):
        config.get_as_of()


def test_every_order_references_known_products_and_customers(orders, catalog, customers):
    for order in orders.values():
        for item in order.items:
            assert item.sku in catalog, f"{order.order_id}: unknown sku {item.sku}"
        if order.customer_id:
            owner = customers[order.customer_id]
            assert owner.email == order.customer_email, order.order_id
        else:
            guest = customers.get(order.customer_email)
            assert guest is not None and guest.is_guest, f"{order.order_id}: no guest record"


def test_catalog_has_sale_items_and_an_out_of_stock_variant(catalog):
    sale = [p for p in catalog.values() if p.on_sale]
    assert len(sale) >= 2 and all(p.sale_name for p in sale)
    assert any(not p.in_stock for p in catalog.values())


def test_customers_include_registered_and_guests(customers):
    assert any(not c.is_guest for c in customers.values())
    assert sum(c.is_guest for c in customers.values()) >= 2


def test_orders_cover_all_statuses_and_window_cases(orders, today):
    assert {o.status for o in orders.values()} == set(OrderStatus)
    days = [policy.days_since_delivery(o, today) for o in orders.values()]
    delivered = [d for d in days if d is not None]
    assert any(d <= policy.RETURN_WINDOW_DAYS for d in delivered), "no in-window order"
    assert any(d > policy.RETURN_WINDOW_DAYS for d in delivered), "no out-of-window order"
    assert policy.RETURN_WINDOW_DAYS in delivered, "no order exactly on day 30"
    assert policy.RETURN_WINDOW_DAYS + 1 in delivered, "no order exactly on day 31"
    assert all(o.delivered_on <= today for o in orders.values() if o.delivered_on)


def test_data_covers_edge_scenarios(orders, catalog):
    assert any(o.refunded for o in orders.values()), "no already-refunded order"
    assert any(len(o.items) > 1 for o in orders.values()), "no multi-item order (ambiguity)"
    assert any(catalog[i.sku].on_sale for o in orders.values() for i in o.items)


def test_a_guest_has_several_orders(customers, orders):
    counts = [
        len(store.orders_for_customer(c, orders)) for c in customers.values() if c.is_guest
    ]
    assert max(counts) >= 2  # "which order?" case


def test_find_customer_by_id_or_email():
    assert store.find_customer("C001").email == "alice.tan@example.com"
    assert store.find_customer(" Alice.Tan@Example.com ").customer_id == "C001"
    guest = store.find_customer("gina.guest@example.com")
    assert guest.is_guest and guest.key == "gina.guest@example.com"
    assert store.find_customer("C999") is None
    assert store.find_customer("nobody@example.com") is None


def test_policy_docs_cover_agent_topics():
    topics = set(store.load_policies())
    assert {"returns", "exchanges", "sale_items", "cancellations", "escalation"} <= topics
