"""Data access: read-only data/*.json, plus read/write of state/state.json.

All order reads go through `fetch_order` / `fetch_orders_for`, which honour the
FAIL_ORDER_LOOKUP=1 failure switch, so a simulated outage blocks every path that
would otherwise show or act on order data.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import TypeAdapter

from config import DATA_DIR
from models import Customer, Order, PolicyDoc, Product, State, normalise_email
from policy import owns_order

_PRODUCTS = TypeAdapter(list[Product])
_CUSTOMERS = TypeAdapter(list[Customer])
_ORDERS = TypeAdapter(list[Order])
_POLICIES = TypeAdapter(list[PolicyDoc])


def _unique(items: list, key: str, what: str) -> dict:
    out = {}
    for item in items:
        k = getattr(item, key)
        if k in out:
            raise ValueError(f"duplicate {what}: {k}")
        out[k] = item
    return out


# ------------------------------------------------------------------------------ data


def load_catalog(data_dir: Path = DATA_DIR) -> dict[str, Product]:
    items = _PRODUCTS.validate_json((data_dir / "products.json").read_text())
    return _unique(items, "sku", "sku")


def load_customers(data_dir: Path = DATA_DIR) -> list[Customer]:
    return _CUSTOMERS.validate_json((data_dir / "customers.json").read_text())


def load_orders(data_dir: Path = DATA_DIR) -> dict[str, Order]:
    items = _ORDERS.validate_json((data_dir / "orders.json").read_text())
    return _unique(items, "order_id", "order id")


def load_policies(data_dir: Path = DATA_DIR) -> dict[str, PolicyDoc]:
    items = _POLICIES.validate_json((data_dir / "policies.json").read_text())
    return _unique(items, "topic", "policy topic")


def find_customer(identifier: str, data_dir: Path = DATA_DIR) -> Customer | None:
    """Resolve a customer id (e.g. C001) or an email address. None if unknown."""
    identifier = identifier.strip()
    customers = load_customers(data_dir)
    if "@" in identifier:
        email = normalise_email(identifier)
        return next((c for c in customers if c.email == email), None)
    return next((c for c in customers if c.customer_id == identifier), None)


def orders_for_customer(customer: Customer, orders: dict[str, Order]) -> list[Order]:
    """Orders owned by this customer (same ownership rule as policy.owns_order)."""
    return [o for o in orders.values() if owns_order(customer, o)]


# ---------------------------------------------------------------- order lookup (guarded)


class OrderLookupError(RuntimeError):
    """The order system is unavailable. FAIL_ORDER_LOOKUP=1 simulates this."""


def _require_order_backend() -> None:
    if os.environ.get("FAIL_ORDER_LOOKUP") == "1":
        raise OrderLookupError("order lookup is unavailable (FAIL_ORDER_LOOKUP=1)")


def fetch_order(order_id: str, data_dir: Path = DATA_DIR) -> Order | None:
    _require_order_backend()
    return load_orders(data_dir).get(order_id)


def fetch_orders_for(customer: Customer, data_dir: Path = DATA_DIR) -> list[Order]:
    _require_order_backend()
    return orders_for_customer(customer, load_orders(data_dir))


# ----------------------------------------------------------------------------- state


def load_state(path: Path) -> State:
    """Missing file means a fresh state. A corrupt file raises: silently starting
    over would forget executed actions and allow double refunds."""
    if not path.exists():
        return State()
    return State.model_validate_json(path.read_text())


def save_state(path: Path, state: State) -> None:
    """Atomic write (temp file + rename) so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(state.model_dump_json(indent=2) + "\n")
    os.replace(tmp, path)
