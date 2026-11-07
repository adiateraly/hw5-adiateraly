"""Commands (run from the repository root):

  python -m helpdesk part1 --all [--clarify] [--save FILE]     Part 1, no model
  python -m helpdesk chat --user demo-user-a --thread t1 --profile normal [--fake-model] [--message TEXT]
  python -m helpdesk scenarios [--id S06 ...] [--fake-model]   run supplied scenarios, write logs
  python -m helpdesk reset                                     delete saved memory (local-storage/)
"""
import argparse
import json
import os
import sys
from pathlib import Path

PART1_MESSAGES = ["I cannot connect to Wi-Fi.", "I forgot my PASSWORD.",
                  "My laptop screen is broken.", "My Wi-Fi password does not work."]


def _model(fake: bool):
    if fake:
        from .fake_model import ScriptedFakeModel
        return ScriptedFakeModel(), "fake model (offline test double, not a real LLM)"
    from .agent import build_real_model
    return build_real_model()


def _print_turn(r: dict) -> None:
    for s in r["steps"]:
        if s.get("tool_requests"):
            for c in s["tool_requests"]:
                print(f"  [model requests] {c['name']}({json.dumps(c['args'], ensure_ascii=False)})")
        for t in s.get("tool_results", []):
            print(f"  [tool result]    {t['name']} -> {json.dumps(t['result'], ensure_ascii=False)[:300]}")
    print(f"  [sources] {r['information_sources'] or 'none'}   loop limit reached: {r['loop_limit_reached']}")
    print(f"Helpdesk: {r['reply']}")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    default_storage = os.environ.get("HELPDESK_STORAGE_DIR", "local-storage")

    p = argparse.ArgumentParser(prog="helpdesk")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("part1")
    p1.add_argument("message", nargs="?")
    p1.add_argument("--all", action="store_true")
    p1.add_argument("--clarify", action="store_true")
    p1.add_argument("--save")

    ch = sub.add_parser("chat")
    ch.add_argument("--user", required=True)
    ch.add_argument("--thread", required=True)
    ch.add_argument("--profile", default="normal",
                    choices=["normal", "wifi_outage", "recovery_outage", "status_unavailable"])
    ch.add_argument("--message", help="send one message and exit (otherwise interactive)")
    ch.add_argument("--fake-model", action="store_true")
    ch.add_argument("--storage", default=default_storage)
    ch.add_argument("--tickets", default="outputs/tickets")

    sc = sub.add_parser("scenarios")
    sc.add_argument("--id", action="append", help="e.g. --id S06 (repeatable); default: all")
    sc.add_argument("--fake-model", action="store_true")
    sc.add_argument("--storage", default=default_storage)
    sc.add_argument("--out")
    sc.add_argument("--tickets", default="outputs/tickets")

    re_ = sub.add_parser("run-events", help="internal: used by `scenarios` for each process")
    re_.add_argument("--events", required=True)
    re_.add_argument("--result", required=True)
    re_.add_argument("--profile", required=True)
    re_.add_argument("--storage", required=True)
    re_.add_argument("--tickets", required=True)
    re_.add_argument("--fake-model", action="store_true")

    rs = sub.add_parser("reset")
    rs.add_argument("--storage", default=default_storage)

    a = p.parse_args()

    if a.cmd == "part1":
        from .part1 import run_with_trace
        msgs = PART1_MESSAGES if a.all else [a.message]
        if not all(msgs):
            p.error("give a message or --all")
        results = []
        for m in msgs:
            r = run_with_trace(m, a.clarify)
            results.append(r)
            print(f"\nMessage: {m}  [{r['graph']} graph]")
            for st in r["trace"]:
                print(f"  -> {st['node']:16} update={st['update']}")
            print(f"  Route: START -> {' -> '.join(r['nodes_visited'])} -> END")
            print("  Reply:\n    " + r["final_state"]["reply"].replace("\n", "\n    "))
        if a.save:
            Path(a.save).parent.mkdir(parents=True, exist_ok=True)
            Path(a.save).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    elif a.cmd == "chat":
        from .app import Helpdesk
        model, name = _model(a.fake_model)
        print(f"Model: {name} | user={a.user} thread={a.thread} profile={a.profile} storage={a.storage}")
        hd = Helpdesk(model, a.storage, a.profile, a.tickets)
        try:
            if a.message:
                _print_turn(hd.send(a.user, a.thread, a.message))
            else:
                print("Type a message (empty line or 'quit' to exit).")
                while True:
                    text = input("You: ").strip()
                    if not text or text.lower() == "quit":
                        break
                    _print_turn(hd.send(a.user, a.thread, text))
        finally:
            hd.close()

    elif a.cmd == "run-events":
        from .app import Helpdesk
        model, name = _model(a.fake_model)
        hd = Helpdesk(model, a.storage, a.profile, a.tickets)
        events = json.loads(Path(a.events).read_text(encoding="utf-8"))
        out = [hd.send(e["user_id"], e["thread_id"], e["message"]) for e in events]
        hd.close()
        Path(a.result).write_text(json.dumps({"model": name, "events": out}, ensure_ascii=False),
                                  encoding="utf-8")

    elif a.cmd == "scenarios":
        from .scenarios import run_scenarios
        out = a.out or ("outputs/fake-model" if a.fake_model else "outputs/real-model")
        run_scenarios(a.id, a.fake_model, a.storage, out, a.tickets)

    elif a.cmd == "reset":
        from .app import reset_storage
        reset_storage(a.storage)
        print(f"Deleted {a.storage}")


if __name__ == "__main__":
    main()
