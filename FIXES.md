# Review fixes and validation

## Failure reproduced and fixed

The original idempotency key was `customer|order|action`. Refunding the hoodie on ORD-1006
then requesting a refund for the cap returned the hoodie's earlier refund as `already_done`.
The cap was not refunded. This was a real deterministic reproduction, not an intentionally
introduced bug. The original failing evidence and passing rerun are both in `eval_runs/`:

- `20261109T082103754422Z_multi_item.json`: failed, only hoodie 60.00 recorded.
- `20261109T082344801306Z_multi_item.json`: passed, hoodie 60.00 and cap 20.00,
  with a repeated cap request recognized without another refund.

`actions._find()` now matches the canonical customer, order, action and original SKU.
`models.idempotency_key()` includes the SKU for refunds/exchanges. `policy.py` checks
conflicting outcomes on that SKU; whole-order cancellation and legacy full-order refund
flags still block all lines. Existing action records with old keys remain recognized.
Regression tests cover separate disk reloads, different lines, repeats, conflicts and old keys.

## Other corrections

- Order lines contain required sale-at-purchase facts, so a changing catalogue cannot change
  return eligibility. Decimal amounts come from price paid, not the model or current price.
- Trusted `defect_verified` order data enables eligible defective sale refunds. Unverified
  claims still escalate. Tool schemas reject attempts to set the verification field.
- Live evals check exact refund SKU/amount and include successful exchanges, verified defects,
  multi-turn clarification, multiple items and valid guest requests.
- Reports are checkpointed before API calls and after each case/attempt, use unique names,
  and preserve interruption/error status. Concurrent additions from the other coding chat
  (adversarial cases, repeat runs and setup documentation) were retained.

## Where execution occurs

`agent.SupportAgent._t_refund()` fetches an owned order through the failure-aware lookup and
calls `actions.refund()`. That function checks ownership, duplicate state and
`policy.check_refund()`, calculates the amount from the purchased line, then calls `_write()`
to persist the simulated refund. The model cannot supply identity, amount, destination,
sale-at-purchase facts or defect verification. This is a local simulated ledger; it does
not send money to a payment provider. The documented single-process limitation still applies.

## Evidence interpretation

Reports are labelled by kind. `offline_tests` and `deterministic_regression` do not call a
model. `live_model` calls the configured model. Older original reports lack the kind field.
The retained setup-error and APIConnectionError run came from the review environment;
they are not hidden or counted as a successful live validation. Later authorized live runs
verify the implementation. Report source hashes identify the tested code/data snapshot.
