# Loom & Linen support agent

A customer-support agent for a fictional apparel store, built from scratch on the Anthropic Python SDK
and pydantic. Claude talks to the customer and *requests* actions through tools. Plain Python decides
whether each request is allowed, and only then executes and records it.

## Setup

You need:

- [uv](https://docs.astral.sh/uv/) **0.13.7 or newer** (the build backend pins `uv_build>=0.13.7,<0.14.0`).
- An Anthropic API key with access to `claude-sonnet-5-5`.

Python itself is not a prerequisite: `.python-version` pins **3.12**, and uv downloads it if it isn't installed.

1. Install uv, if you don't have it. On macOS with Homebrew:

   ```bash
   brew install uv
   ```

   Or with the official installer (macOS/Linux):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Check that it worked:

   ```bash
   uv --version
   ```

2. Install the dependencies from the project root. This creates `.venv/` and installs Python 3.12 if
   needed, plus `anthropic` and `pydantic` and the `pytest` dev group, at the versions locked in `uv.lock`:

   ```bash
   uv sync
   ```

3. Set your API key. It's read from `ANTHROPIC_API_KEY` in the environment:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

   Add that line to your shell profile (e.g. `~/.zshrc`) to keep it set. A `.env` file is gitignored, but
   nothing loads it automatically. If you keep the key there, export it into your shell first:

   ```bash
   set -a; source .env; set +a
   ```

4. Check the install. The unit tests make no network calls and need no API key:

   ```bash
   uv run pytest
   ```

## Run it

```bash
uv run python main.py --customer <id-or-email> "<message>"
```

It prints the reply, then one line saying what happened:

```text
$ uv run python main.py --customer C001 "I'd like a refund for the black tee in order ORD-1001, I changed my mind."
Your refund has been processed. The 25.00 for the Classic Cotton Tee in black (M) from order ORD-1001 will go back to your original payment method.

Action taken: refund - ORD-1001 TEE-BLK-M, 25.00 to the original payment method
```

`Action taken:` is one of `refund`, `exchange`, `cancellation`, `escalation` (with the order), or `none`.
A repeat of something already done prints `none (already done earlier: ... - not repeated)`.

`--customer` takes a customer ID (`C001`-`C004`) or an email, including guests'
(`gina.guest@example.com`, `gary.guest@example.com`). Add `--verbose` to print the tool-call trace to stderr.

### Switches

| Variable | Effect |
|---|---|
| `FAIL_ORDER_LOOKUP=1` | Order lookup raises (simulated outage). The agent escalates and apologises, and never guesses or acts. |
| `AS_OF_DATE=YYYY-MM-DD` | Overrides the business date. **The default is the fixed demo date `2026-11-09`**, not the real date, so the 30-day-window cases in `data/` never drift. |
| `MODEL` | Model ID (default `claude-sonnet-5-5`). |
| `STATE_FILE` | State file path (default `state/state.json`; the evals use a temp file). |

```bash
FAIL_ORDER_LOOKUP=1 uv run python main.py --customer C003 "Please cancel my order ORD-1005."
```

## How it works

```text
main.py --customer X "msg"
  -> agent.py      Claude tool-use loop; customer text wrapped as untrusted
       -> tools    lookup_customer, lookup_order, get_policy,
                   request_refund, request_exchange, request_cancellation, escalate_to_human
            -> actions.py   1. idempotency check (customer|order|action|sku) -> "already done"
                            2. policy.py  pure rule check -> allowed / denied / needs_human
                            3. only if allowed: execute + write state/state.json (atomic)
```

- **`policy.py`** holds pure functions with no I/O and no clock (`today` is passed in). It checks:
  - the order belongs to this customer (guests: matching email);
  - the 30-day window from delivery (day 30 in, day 31 out);
  - sale items are exchange-only unless defective, and a defect claim on a sale item goes to a human
    (`needs_human` / `defect_unverified`); an order line with a verified defect permits a refund;
  - the SKU is not already refunded/exchanged and the order is not cancelled;
  - only processing orders can be cancelled.
- **`actions.py`** is the only module that writes actions. A denied request returns its reason to the
  model, which must explain it or escalate.
- **Identity** comes from `--customer`, never from the model or the message. No tool takes a customer ID,
  tool inputs reject unknown fields, and another customer's order looks the same as a non-existent one.
- **State** (`state/state.json`, gitignored) holds the executed actions keyed by idempotency key, plus each
  customer's conversation history. A repeated request is recognised in code and never executed twice.
- **Missing information:** with no order ID, or an ambiguous item, the agent asks a clarifying question
  instead of acting. Guests are shown their orders and asked which one.
- **Failures:** an order-system outage, a model refusal, a runaway loop or an unreachable API is escalated by
  code, and the customer gets an apology.

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every judgement call.

## Data

`data/` holds the products (some `on_sale`), customers (4 registered, 2 guests), orders and policies. The
orders are chosen to cover each rule at the fixed date 2026-11-09:

| Order | Customer | Case |
|---|---|---|
| ORD-1001 | C001 | delivered 15 days ago, regular item |
| ORD-1002 | C001 | delivered 50 days ago |
| ORD-1003 | C002 | sale item (field jacket), 7 days |
| ORD-1004 | C002 | shipped |
| ORD-1005 | C003 | processing |
| ORD-1006 | C003 | two items (ambiguity) |
| ORD-1007 | C004 | already refunded |
| ORD-1008 / ORD-1009 | C004 | exactly day 30 / day 31 |
| ORD-2001, ORD-2002 | guest gina | delivered tee / processing sale scarf |
| ORD-2003 | guest gary | delivered 71 days ago |

## Tests and evals

```bash
uv run pytest                          # unit tests: no network, fake model client
```

```bash
uv run python evals/run_evals.py       # end-to-end with the real model
```

The evals run 27 cases, for example a refund in the window, day 30 vs 31, sale items, defect claims, another
customer's order, a duplicate request, the outage switch, a missing order ID, prompt injection, cancellations,
refund after an exchange, and a policy question. Each case:

- runs `main` in-process against a fresh temp state file;
- checks results in code against state and the action records (action type, order, refund count, "must
  not" rules), never against the wording;
- checkpoints `eval_runs/<unique-timestamp>.json` at startup and after each case with the model, per-case pass/fail and reasons, replies, actions,
  traces and a summary.

The command exits 1 if any attempt fails. Pass case IDs to run a subset, and `--repeat N` to run each
case N times and record a pass rate per case:

```bash
uv run python evals/run_evals.py 05 09
```

```bash
uv run python evals/run_evals.py --repeat 3
```

Cases 17-22 are adversarial:

- a guest using someone else's order ID with a plausible story (the eval asserts the ownership check
  was actually reached);
- an inflated refund amount;
- both items of a multi-item order requested at once;
- a "your colleague promised me" exception;
- an injection naming a specific order and reason;
- a currency check (no symbol in the reply that isn't in the data).

The evals call the real model and cost money: one full run is 31 agent turns,
and `--repeat 3` triples that.

## Resetting state

Executed actions and conversation history persist in `state/state.json`. Delete it to start fresh (for
example before a demo, or the second refund will correctly report "already done"):

```bash
rm -f state/state.json
```


## Review fixes and evidence

Refunds are simulated by the durable action record written by `actions.refund()` through `_write()`;
there is no real payment-provider integration. The amount comes from the purchased line's Decimal
unit price times its quantity. The model supplies neither the amount nor payment destination.

Sale status is captured at purchase. `ORD-1010` (C002, jacket, 84.00) has a trusted verified defect;
`ORD-1003` has an unverified defect path. Different items of `ORD-1006` can each be refunded once.
Full SKU quantities are supported; partial quantities need clarification or human handling.

To replay the deterministic failure/fix evaluation without a model key:

```bash
uv run python evals/reproduce_multi_item.py
```

This saves its actual result under `eval_runs/`, clearly labelled `deterministic_regression`.
The retained failing run shows the cap incorrectly reported as the earlier hoodie refund. The fix
scopes repeat detection and conflict checks to the original SKU, while retaining order-wide cancellation
and compatibility with previous persisted records. `tests/test_regressions.py` covers the bug and guards.

Live eval runs are labelled `live_model`. They check exact refund SKU/amount, successful exchange,
verified defective sale refund, clarification then resolution, multiple items and valid guest behavior.
The report is saved before calls begin and after each case; Ctrl-C/SIGTERM preserves completed cases
and marks the report interrupted. A hard kill may leave a running report identifying the active case.
Offline tests use fake clients and do not establish live-model quality.
