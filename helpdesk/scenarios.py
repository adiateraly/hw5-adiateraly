"""Runs S01-S14 from data/scenarios.json and writes one log per scenario.

restart_process events really terminate the program: each group of events between
restarts runs in its own `python -m helpdesk run-events` subprocess.
"""
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .app import reset_storage
from .part1 import run_with_trace

DATA = Path(__file__).resolve().parent.parent / "data" / "scenarios.json"


def render_text(log: dict) -> str:
    L = [f"Scenario: {log['id']} - {log['title']}", f"Model: {log['model']}",
         f"Status snapshot: {log['status_profile']}", f"Command: {log['command']}",
         f"Run at: {log['run_at']}", ""]
    for i, ev in enumerate(log["events"], 1):
        if ev.get("action") == "restart_process":
            L.append(f"--- event {i}: PROCESS RESTARTED (storage kept) ---\n")
            continue
        L.append(f"--- event {i}: user={ev.get('user_id')} thread={ev.get('thread_id')}")
        L.append(f"User: {ev['message']}")
        if "trace" in ev:  # Part 1
            L.append(f"Graph: {ev['graph']}")
            for t in ev["trace"]:
                L.append(f"  node {t['node']}: update={json.dumps(t['update'], ensure_ascii=False)}")
        else:
            L.append(f"State messages before this turn: {ev['messages_in_state_before']}")
            for s in ev["steps"]:
                L.append(f"  node {s['node']}: " + json.dumps(
                    {k: v for k, v in s.items() if k != "node"}, ensure_ascii=False))
            L.append(f"Information sources: {ev['information_sources'] or 'none'}")
            L.append(f"Ticket files: {ev['ticket_files'] or 'none'}")
            L.append(f"Loop limit reached: {ev['loop_limit_reached']}")
        L.append(f"Reply: {ev['reply']}\n")
    return "\n".join(L)


def run_scenarios(ids, fake: bool, storage_dir: str, out_dir: str, tickets_dir: str) -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))["scenarios"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = "fake model (offline test double)" if fake else None
    for sc in data:
        if ids and sc["id"] not in ids:
            continue
        if sc.get("reset_before"):
            reset_storage(storage_dir)
        command = (f"python -m helpdesk scenarios --id {sc['id']}" + (" --fake-model" if fake else ""))
        events_out, model_name = [], mode or "part 1: no model"
        if sc["part"] == 1:
            clarify = sc.get("graph_variant") == "with_clarification"
            for ev in sc["events"]:
                r = run_with_trace(ev["message"], clarify)
                events_out.append({**ev, "graph": r["graph"], "trace": r["trace"],
                                   "reply": r["final_state"]["reply"]})
            model_name = "none (Part 1 graph, no LLM)"
        else:
            segment = []
            def flush():
                nonlocal segment, model_name
                if not segment:
                    return
                with tempfile.TemporaryDirectory() as tmp:
                    inp, outp = Path(tmp, "in.json"), Path(tmp, "out.json")
                    inp.write_text(json.dumps(segment), encoding="utf-8")
                    cmd = [sys.executable, "-m", "helpdesk", "run-events", "--events", str(inp),
                           "--result", str(outp), "--profile", sc["status_profile"],
                           "--storage", storage_dir, "--tickets", tickets_dir]
                    if fake:
                        cmd.append("--fake-model")
                    subprocess.run(cmd, check=True)
                    res = json.loads(outp.read_text(encoding="utf-8"))
                    model_name = res["model"]
                    events_out.extend(res["events"])
                segment = []
            for ev in sc["events"]:
                if ev["action"] == "restart_process":
                    flush()
                    events_out.append({"action": "restart_process"})
                else:
                    segment.append(ev)
            flush()
        log = {"id": sc["id"], "title": sc["title"], "model": model_name, "command": command,
               "status_profile": sc["status_profile"],
               "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "events": events_out}
        (out / f"{sc['id']}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / f"{sc['id']}.txt").write_text(render_text(log), encoding="utf-8")
        print(f"{sc['id']} done -> {out / (sc['id'] + '.txt')}")
