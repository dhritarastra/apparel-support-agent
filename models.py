"""Pydantic models shared by data loading, policy, actions and the agent.

Domain models are frozen and reject unknown fields, so a typo in a data file or
a stray field from the model fails loudly instead of being silently ignored.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def normalise_email(value: str) -> str:
    value = value.strip().lower()
    if "@" not in value:
        raise ValueError(f"not an email address: {value!r}")
    return value


# --------------------------------------------------------------------------- enums


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"


class ActionType(StrEnum):
    REFUND = "refund"
    EXCHANGE = "exchange"
    CANCELLATION = "cancellation"
    ESCALATION = "escalation"


class Outcome(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    NEEDS_HUMAN = "needs_human"  # policy cannot decide; a person must


class ReturnReason(StrEnum):
    """Why the customer wants a refund/exchange. Self-reported, i.e. untrusted."""

    DEFECTIVE = "defective"
    CHANGED_MIND = "changed_mind"
    WRONG_SIZE = "wrong_size"
    OTHER = "other"


class ReasonCode(StrEnum):
    OK = "ok"
    ORDER_NOT_OWNED = "order_not_owned"
    ITEM_NOT_IN_ORDER = "item_not_in_order"
    UNKNOWN_PRODUCT = "unknown_product"
    NOT_DELIVERED = "not_delivered"
    OUTSIDE_WINDOW = "outside_window"
    SALE_ITEM_EXCHANGE_ONLY = "sale_item_exchange_only"
    DEFECT_UNVERIFIED = "defect_unverified"
    ALREADY_REFUNDED = "already_refunded"
    ALREADY_EXCHANGED = "already_exchanged"
    ALREADY_CANCELLED = "already_cancelled"
    NOT_CANCELLABLE = "not_cancellable"
    INVALID_EXCHANGE_TARGET = "invalid_exchange_target"


# ------------------------------------------------------------------------ catalogue


class Product(_Model):
    sku: str
    name: str
    style: str  # variants (sizes/colours) of one product share a style
    price: Decimal = Field(gt=0)  # current selling price (already discounted if on sale)
    on_sale: bool = False
    sale_name: str | None = None
    in_stock: bool = True

    @model_validator(mode="after")
    def _sale_fields_consistent(self) -> Product:
        if self.on_sale and not self.sale_name:
            raise ValueError(f"{self.sku}: on_sale products need a sale_name")
        if not self.on_sale and self.sale_name:
            raise ValueError(f"{self.sku}: sale_name set but on_sale is false")
        return self


class Customer(_Model):
    """A registered customer (has `customer_id`) or a guest (email only)."""

    customer_id: str | None = None
    email: str
    name: str | None = None

    @field_validator("email")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return normalise_email(value)

    @property
    def is_guest(self) -> bool:
        return self.customer_id is None

    @property
    def key(self) -> str:
        """Stable identity used in idempotency keys and conversation history."""
        return self.customer_id or self.email


# ---------------------------------------------------------------------------- orders


class OrderItem(_Model):
    sku: str
    quantity: int = Field(ge=1)
    unit_price: Decimal = Field(gt=0, decimal_places=2)  # price actually paid
    on_sale_at_purchase: bool  # required historical fact, never inferred from today's catalogue
    sale_name_at_purchase: str | None = None
    defect_verified: bool = False  # trusted back-office data; never a model tool argument


class Order(_Model):
    order_id: str
    customer_id: str | None = None  # None for guest orders
    customer_email: str
    status: OrderStatus
    placed_on: date
    shipped_on: date | None = None
    delivered_on: date | None = None
    items: tuple[OrderItem, ...] = Field(min_length=1)
    refunded: bool = False  # refund already issued before this system saw the order

    @field_validator("customer_email")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return normalise_email(value)

    @model_validator(mode="after")
    def _dates_match_status(self) -> Order:
        if len({i.sku for i in self.items}) != len(self.items):
            raise ValueError("order lines must have unique SKUs")
        s = self.status
        if s is OrderStatus.PROCESSING and (self.shipped_on or self.delivered_on):
            raise ValueError(f"{self.order_id}: processing order has ship/delivery dates")
        if s is OrderStatus.SHIPPED and (self.shipped_on is None or self.delivered_on):
            raise ValueError(f"{self.order_id}: shipped order needs shipped_on only")
        if s is OrderStatus.DELIVERED and self.delivered_on is None:
            raise ValueError(f"{self.order_id}: delivered order needs delivered_on")
        if self.refunded and s is not OrderStatus.DELIVERED:
            raise ValueError(f"{self.order_id}: only delivered orders can be refunded")
        return self

    def item(self, sku: str) -> OrderItem | None:
        return next((i for i in self.items if i.sku == sku), None)


class PolicyDoc(_Model):
    topic: str
    title: str
    text: str


# ------------------------------------------------------------------ policy decisions


class Decision(_Model):
    """Result of a policy check.

    ALLOWED exactly when code is OK; NEEDS_HUMAN exactly when code is DEFECT_UNVERIFIED.
    """

    outcome: Outcome
    code: ReasonCode
    reason: str
    alternatives: tuple[ActionType, ...] = ()  # actions the customer could still request

    @property
    def allowed(self) -> bool:
        return self.outcome is Outcome.ALLOWED

    @property
    def needs_human(self) -> bool:
        return self.outcome is Outcome.NEEDS_HUMAN

    @model_validator(mode="after")
    def _outcome_matches_code(self) -> Decision:
        if (self.outcome is Outcome.ALLOWED) != (self.code is ReasonCode.OK):
            raise ValueError("outcome is ALLOWED iff code is OK")
        if (self.outcome is Outcome.NEEDS_HUMAN) != (self.code is ReasonCode.DEFECT_UNVERIFIED):
            raise ValueError("outcome is NEEDS_HUMAN iff code is DEFECT_UNVERIFIED")
        return self

    @classmethod
    def allow(cls, reason: str = "Request meets policy.") -> Decision:
        return cls(outcome=Outcome.ALLOWED, code=ReasonCode.OK, reason=reason)

    @classmethod
    def deny(
        cls, code: ReasonCode, reason: str, alternatives: tuple[ActionType, ...] = ()
    ) -> Decision:
        return cls(outcome=Outcome.DENIED, code=code, reason=reason, alternatives=alternatives)

    @classmethod
    def needs_human_review(cls, reason: str) -> Decision:
        return cls(outcome=Outcome.NEEDS_HUMAN, code=ReasonCode.DEFECT_UNVERIFIED, reason=reason)


# ------------------------------------------------------------------------------ state


def idempotency_key(customer_key: str, order_id: str | None, action: ActionType, sku: str | None = None) -> str:
    key = f"{customer_key}|{order_id or '-'}|{action.value}"
    return f"{key}|{sku}" if sku and action in {ActionType.REFUND, ActionType.EXCHANGE} else key


class ActionRecord(_Model):
    idempotency_key: str
    action: ActionType
    customer_key: str
    order_id: str | None = None
    sku: str | None = None
    reason: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    as_of: date  # business date the action was evaluated against
    executed_at: datetime  # wall-clock audit timestamp


class Turn(_Model):
    role: Literal["customer", "agent"]
    text: str
    at: datetime


class State(BaseModel):
    """Contents of state/state.json (mutable, written only by actions/store)."""

    model_config = ConfigDict(extra="forbid")

    executed_actions: dict[str, ActionRecord] = Field(default_factory=dict)
    history: dict[str, list[Turn]] = Field(default_factory=dict)
