"""CLI: uv run python main.py --customer <id-or-email> "<message>"

Prints the agent's reply, then one line: `Action taken: ...`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import anthropic

import actions
import store
from agent import AgentResult, SupportAgent


def action_line(result: AgentResult) -> str:
    new = [actions.describe_record(r) for r in result.new_records]
    if new:
        return "Action taken: " + "; ".join(new)
    repeats = [actions.describe_record(r) for r in result.repeated_records]
    if repeats:
        return (
            "Action taken: none (already done earlier: " + "; ".join(repeats)
            + " - not repeated)"
        )
    return "Action taken: none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loom & Linen customer-support agent")
    parser.add_argument("--customer", required=True, help="customer ID (e.g. C001) or email")
    parser.add_argument("message", help="the customer's message")
    parser.add_argument(
        "--verbose", action="store_true", help="print the tool-call trace to stderr"
    )
    args = parser.parse_args(argv)

    message = args.message.strip()
    if not message:
        parser.error("message must not be empty")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    customer = store.find_customer(args.customer)
    if customer is None:
        print(f"Sorry, we couldn't find a customer account for {args.customer!r}.")
        print("Action taken: none")
        return 1

    try:
        result = SupportAgent(customer).run(message)
    except anthropic.APIStatusError as exc:  # auth/permission/bad request: a setup problem
        print(f"error: Claude API request failed ({exc.status_code}): {exc.message}", file=sys.stderr)
        return 2

    if args.verbose:
        for step in result.trace:
            print(f"[tool] {step['tool']} {json.dumps(step['input'])} -> "
                  f"{json.dumps(step['output'], default=str)}", file=sys.stderr)
        if result.fallback_reason:
            print(f"[fallback] {result.fallback_reason}", file=sys.stderr)

    print(result.reply)
    print()
    print(action_line(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
