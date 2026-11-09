"""Deterministic regression evaluation, no API calls. Run before/after the fix."""
import json
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import actions
import store
from actions import ActionContext
from models import ReturnReason

def run():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        order = store.load_orders()["ORD-1006"]
        results = []
        for sku in ("HOOD-GRY-L", "CAP-NVY-OS", "CAP-NVY-OS"):
            ctx = ActionContext(store.find_customer("C003"), store.load_catalog(),
                                store.load_state(path), path, date(2026, 11, 9))
            out = actions.refund(ctx, order, sku, ReturnReason.CHANGED_MIND)
            results.append({"requested_sku": sku, "status": out.status.value,
                            "record": out.record.model_dump(mode="json") if out.record else None})
        records = list(store.load_state(path).executed_actions.values())
        actual = sorted((r.sku, r.detail.get("amount")) for r in records)
        expected = [("CAP-NVY-OS", "20.00"), ("HOOD-GRY-L", "60.00")]
        passed = actual == expected and [r["status"] for r in results] == ["executed", "executed", "already_done"]
    report = {"kind": "deterministic_regression", "model": None,
              "case": "different items must not be mistaken for duplicate refunds",
              "passed": passed, "expected": expected, "actual": actual, "turns": results,
              "timestamp": datetime.now(UTC).isoformat()}
    dest = ROOT / "eval_runs"
    dest.mkdir(exist_ok=True)
    filename = dest / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "_multi_item.json")
    filename.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "expected": expected, "actual": actual, "report": str(filename)}))
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(run())
