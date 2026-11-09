# Assumptions

Decisions made where the brief was silent. Extended as each layer lands.

## Clock
- Business "today" is the fixed demo date **2026-11-09** (`config.DEFAULT_AS_OF`), not the wall clock, so the
  30-day-window cases in `data/` give the same answer whenever the code is run or graded.
- Override with `AS_OF_DATE=YYYY-MM-DD`. An invalid value raises instead of silently falling back.
- `policy.py` never reads the clock; the date is passed in.

## Identity and ownership
- `--customer` is trusted as the caller's identity (no authentication is in scope). It is resolved by customer ID
  or email against `data/customers.json`; an unknown ID or email is treated as "customer not found".
- Guests are identified by email only. A guest owns only guest orders (no `customer_id`) placed with that exact
  email (case-insensitive). A guest using a registered customer's email does **not** own that customer's orders.
- A registered customer owns orders carrying their `customer_id`.
- Any action needs an order ID that passes the ownership check. If the customer gives none, the agent lists their
  orders and asks which one; it never acts on an order that fails the check. The ownership denial reveals nothing
  about the order.

## Refunds and exchanges
- The 30 days run from `delivered_on`. Day 30 is inside the window, day 31 is outside. This applies to
  exchanges and to defective items too (a defect does not extend the window).
- Undelivered orders (processing/shipped, or a delivery date in the future) cannot be refunded or exchanged.
- Sale eligibility uses required `on_sale_at_purchase` and `sale_name_at_purchase` on the order line.
  Catalogue promotions can change without changing the purchase terms.
- Unverified defect claims on sale items escalate with the customer's description. A trusted back-office
  `defect_verified: true` on an order line permits a refund within the window, still subject to ownership
  and prior-action checks. The model cannot set this field. ORD-1010 demonstrates a verified defect;
  ORD-1003 demonstrates an unverified claim. Verification is supplied by fixture data; a staff approval UI
  is outside this prototype's scope.
- A defective item that is **not** on sale is refunded normally within the window. The `defective` reason is
  still customer-supplied, but it changes nothing for such items, so claiming it gains nothing.
- A defective sale item outside the window, already refunded, or not owned by the customer is a plain denial
  (not an escalation): the other rules are checked first.
- Exchanging a defective sale item is allowed automatically (exchange is already permitted for sale items).
- An exchange must be for the same product (same `style`, e.g. another size), in stock, and a different SKU.
- Refund/exchange idempotency is per canonical customer, order, action and original SKU. Each action
  resolves all units of that SKU; partial quantities require clarification or escalation. Duplicate SKU
  lines are invalid data. Different SKUs in a multi-item order can be handled independently.
- A refunded SKU cannot also be exchanged, and an exchanged SKU cannot be refunded. Cancellation and
  the legacy `refunded: true` order flag settle the whole order. Old action records with no SKU are treated
  conservatively as settling the whole order; old keyed records with a SKU remain deduplicated.
- Escalations do not settle an order and never block a later action.

## Cancellations
- Only `processing` orders can be cancelled. "Cancelled" is recorded in state, not in the data files.

## Escalations
- One escalation per `customer|order` (or `customer|-` with no order), per the idempotency key. Repeating it
  doesn't open a second ticket. A repeat with a *new* summary is appended to the existing record as a
  `follow_ups` entry, so the human still sees the new information.

## State
- `state/state.json` (or `STATE_FILE`, used by the evals) holds executed actions keyed by idempotency key, plus
  each customer's conversation history (last 40 turns stored, last 20 sent to the model). It is written
  atomically after every action and at the end of every turn.
- A missing state file means a fresh start. A **corrupt** state file is an error, not a reset: silently starting
  over would forget executed refunds.
- One process at a time. There is no file locking, so concurrent runs against the same state file aren't
  supported.
- Ownership is checked before idempotency; eligibility is checked before any new execution. A repeat request therefore reports "already done" (never a
  second execution) even if the model doesn't notice the repeat itself.

## Failure handling
- Policy checks fail closed: a product missing from the catalogue is a denial (`unknown_product`), not a crash.
- `FAIL_ORDER_LOOKUP=1` makes every order read fail, not only the `lookup_order` tool: the refund, exchange
  and cancellation tools and the customer's order list all go through the same lookup. On the first failure in
  a turn, code records one escalation (with the customer's message) and tells the model to apologise without
  guessing. No action can be taken during the outage.
- If the model refuses, gives an empty reply, exceeds 10 model calls in a turn, or the API is unreachable
  (rate limit, connection or server error), code escalates and the customer gets a fixed apology. Setup errors
  (bad key, unknown model) exit with code 2 instead.
- The model is `MODEL` (default `claude-sonnet-5-5`) with adaptive thinking and effort `medium`. Server-side
  refusal fallbacks to another model are **not** enabled: you specified one model, so a refusal escalates to a
  human instead.

## Money
- The fictional store currency is SGD (`data/store.json`). Amounts use Decimal with at most two fractional digits.
