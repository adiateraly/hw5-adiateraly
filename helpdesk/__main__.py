"""Usage:
  python -m helpdesk part1 "I cannot connect to Wi-Fi."
  python -m helpdesk part1 --clarify "My Wi-Fi password does not work."
  python -m helpdesk part1 --all [--clarify]      # the four Part 1 messages
Add --save outputs/part1_S01.json to store the trace.
"""
import argparse
import json
from pathlib import Path

PART1_MESSAGES = [
    "I cannot connect to Wi-Fi.",
    "I forgot my PASSWORD.",
    "My laptop screen is broken.",
    "My Wi-Fi password does not work.",
]


def main() -> None:
    from .part1 import run_with_trace

    p = argparse.ArgumentParser(prog="helpdesk")
    sub = p.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("part1", help="run the fixed-route graph (no model)")
    p1.add_argument("message", nargs="?")
    p1.add_argument("--all", action="store_true")
    p1.add_argument("--clarify", action="store_true",
                    help="use the graph with clarify_problem")
    p1.add_argument("--save", help="write the trace(s) as JSON to this path")
    args = p.parse_args()

    messages = PART1_MESSAGES if args.all else [args.message]
    if not all(messages):
        p.error("give a message or --all")
    results = []
    for m in messages:
        r = run_with_trace(m, args.clarify)
        results.append(r)
        print(f"\nMessage: {m}  [{r['graph']} graph]")
        for step in r["trace"]:
            print(f"  -> {step['node']:16} update={step['update']}")
        print(f"  Route: START -> {' -> '.join(r['nodes_visited'])} -> END")
        print(f"  Reply:\n    " + r["final_state"]["reply"].replace("\n", "\n    "))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
