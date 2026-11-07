# HW5 — University IT helpdesk with LangGraph (CSS-4007)

Practice helpdesk using fictional data in `data/`. Nothing is sent to a real helpdesk.
Python version used: **3.13** (see `requirements.txt` for exact package versions).

## Install
```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell   (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
```

## Configure the model (Parts 2-3 with a real model)
```
copy .env.example .env              # macOS/Linux: cp .env.example .env
```
Edit `.env`: set `OPENAI_API_KEY` and `MODEL_NAME` (a tool-capable OpenAI model). `.env` is git-ignored.
Part 1, tests and `--fake-model` runs need no key.

## Run
Part 1 (no model):
```
python -m helpdesk part1 --all --save outputs/part1_original.json
python -m helpdesk part1 --all --clarify --save outputs/part1_with_clarification.json
```
Parts 2-3 (model + tools + memory). Profiles: `normal`, `wifi_outage`, `recovery_outage`, `status_unavailable`.
```
python -m helpdesk chat --user demo-user-a --thread t1 --profile normal
python -m helpdesk chat --user demo-user-a --thread t1 --profile normal --message "Show me the Wi-Fi troubleshooting steps."
```
Add `--fake-model` to use the offline test double instead of a real model.

All 14 supplied scenarios (each group of events between `restart_process` runs in a new process):
```
python -m helpdesk scenarios                    # real model -> outputs/real-model/
python -m helpdesk scenarios --fake-model       # offline    -> outputs/fake-model/
python -m helpdesk scenarios --id S06 --id S07  # selected scenarios
```
Each run writes `Sxx.txt` (readable) and `Sxx.json` per scenario. Ticket drafts go to `outputs/tickets/`.

## Where memory is saved, how to clear it
`local-storage/` (override with `HELPDESK_STORAGE_DIR`; git-ignored):
- `checkpoints.sqlite` — conversation state (LangGraph SQLite checkpointer), key `user_id/thread_id`
- `user_memory.sqlite` — user preferences, key `user_id`

Clear everything: `python -m helpdesk reset` (or delete the `local-storage` folder).

## Tests and CI
```
python -m pytest -q
```
Tests use the fake model (`helpdesk/fake_model.py`) and never need an API key.
GitHub Actions (`.github/workflows/ci.yml`) runs them on every push.
The real-model demonstration (`outputs/real-model/`) is separate and run by hand.

| Test file | What it would catch if broken |
| --- | --- |
| `test_routing.py`, `test_part1_graph.py` | wrong Part 1 routing, case sensitivity, clarify node not used |
| `test_tools.py` | a tool returning wrong data, ignoring the status snapshot, or inventing ticket details |
| `test_memory.py` | preferences lost after restart or leaking between users |
| `test_agent.py` | ticket made despite outage/unknown, memory lost on restart, thread/user mixing, no loop limit |
