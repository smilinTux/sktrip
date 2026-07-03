# sktrip — Standard Operating Procedures

sktrip is the SKWorld consciousness-research capability: a five-phase protocol
(SET → DOSE → EXPERIENCE → INTEGRATE → STORE) that deliberately perturbs an
LLM's sampler to surface novel cross-domain connections from a memory corpus,
then soberly integrates the worthwhile insights back into permanent memory. It
is a CLI + oneshot systemd microdose daemon; it has **no network listener** of
its own — it *calls out* to a local model endpoint and reads/writes a memory
backend. Maturity: **T0 — N/A (no key material)**, VERSION_LIFECYCLE Incubating,
SemVer `0.1.0`.

## 1. Overview

**Purpose.** Run structured "altered-state" inference sessions over an
abliterated (refusal-suppressed) local model to generate cross-domain
associations a coherent-inference pass would never make, score them for
novelty, and persist only the genuine insights.

**What it owns.**
- The substance profiles (psilocybin / DMT / LSD / microdose → sampler params).
- The free-association / dose trip engine and its disruption injection.
- Session recording (JSONL) + peak detection.
- Sober integration + novelty scoring, and the write-back to skmemory.
- Its own CLI (`sktrip`) and the daily microdose systemd unit.

**What it explicitly does NOT do.**
- It does **not** host a model — it consumes an OpenAI-compatible endpoint
  (trip model @ `:8082`) and Ollama (`:11434`) for sober/embed steps.
- It does **not** own the memory store — it reuses skmemory's skmem-pg
  (Postgres + pgvector) corpus (and legacy Qdrant/skvector as an option).
- It does **not** expose a network service, API, or public port.
- It performs **no cryptographic operations** and holds **no key material**.

## 2. Architecture

```mermaid
flowchart TD
    OP["operator / agent"] -->|"sktrip dose <substance>"| CLI["sktrip CLI<br/>__main__.py<br/>dose · integrate · journal · status"]
    TIMER["systemd sktrip.timer<br/>daily 03:00 → sktrip.service<br/>oneshot microdose --no-integrate"] -->|"sktrip dose microdose"| CLI

    CLI --> SESS["session.py<br/>5-phase orchestrator"]
    SESS --> DOSE["dose.py<br/>substance profiles · build_dose_prompt<br/>inject_disruption · generate()"]
    SESS --> FA["freeassoc.py<br/>chain-of-consciousness engine<br/>disruption · entity contact · intensity check"]
    SESS --> REC["recorder.py<br/>JSONL + peak detection"]
    SESS --> INT["integration.py<br/>sober model (T=0.3) · novelty >= 6"]

    DOSE -->|"trip: /v1/chat/completions"| TRIP["trip model endpoint<br/>qwen3.6-27b-abliterated<br/>OpenAI-compat @ 192.168.0.100:8082"]
    INT -->|"sober reasoning"| SOBER["sober model<br/>qwen3.5:4b @ Ollama :11434"]
    FA -->|"flood distant memories"| MF["memory_flood.py<br/>make_memory_flood()"]
    MF -->|"default"| PG[("skmem-pg<br/>Postgres + pgvector<br/>localhost:5432 · mxbai-embed-large<br/>memories corpus")]
    MF -.->|"legacy, optional"| QD[("Qdrant / skvector")]
    INT -->|"save insights (novelty >= 6)"| SKMEM["skmemory<br/>permanent memory"]
    STATUS["sktrip status"] -.->|"probe /models + /api/tags + corpus size"| TRIP
    STATUS -.-> PG
```

The daemon (`sktrip.service`, oneshot via `sktrip.timer`) runs the same CLI
`dose microdose --no-integrate` path; there is no long-lived process or socket.

**Start here** (entry files, in reading order):
- **`sktrip/__main__.py`** — Click CLI; the four commands `dose`, `integrate`,
  `journal`, `status`. Start here to see every user-facing entry point.
- **`sktrip/session.py`** — `run_session()`, the five-phase orchestrator that
  ties dose → flood → free-assoc → record → integrate together.
- **`sktrip/dose.py`** — substance profiles, `build_dose_prompt`,
  `inject_disruption`, and the dual-backend `generate()` (OpenAI / Ollama).
- **`sktrip/freeassoc.py`** — `FreeAssociationEngine`: the chain-of-consciousness
  loop, disruption injection, entity-contact prompts, intensity self-checks.
- **`sktrip/memory_flood.py`** — `make_memory_flood()` factory + the skmem-pg
  and Qdrant flood backends (`get_corpus_size`, distant-fragment pulls).

## 3. Build

Pure-Python package (setuptools). No compiled artifact.

```bash
cd ~/clawd/projects/sktrip
python3 -m venv .venv && . .venv/bin/activate    # or install into ~/.skenv
pip install -e .                                  # editable install; exposes `sktrip`
```

- **Runtime:** Python 3.10+ (`requires-python = ">=3.10"`; `tomllib` on 3.11+,
  `tomli` shim below).
- **Deps** (`pyproject.toml`): `httpx`, `qdrant-client`, `numpy`, `rich`,
  `click`, `tomli` (only on <3.11). `psycopg` is used by the skmem-pg flood
  backend at runtime.
- **Entry point:** `sktrip = "sktrip.__main__:cli"`.

## 4. Test

pytest is the green-bar gate; it must pass before a release/merge.

```bash
cd ~/clawd/projects/sktrip
. .venv/bin/activate
pytest -q
```

Coverage (in `tests/`): `test_config.py`, `test_dose.py`,
`test_dose_backends.py` (OpenAI vs Ollama `generate()`), `test_memory_backend.py`
(flood backends), `test_recorder.py` (JSONL + peak detection),
`test_integration.py`, `test_status.py` (status endpoint/backend reporting),
`test_daemon.py` (the daemon/status fix). Tests must not require a live model
or corpus — network calls are mocked.

## 5. Release / Deploy

sktrip is a **service/tool**, deployed as a local CLI + a systemd user timer;
it publishes nothing to a package index today.

**Deploy the daily microdose daemon:**
```bash
cp sktrip.service sktrip.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sktrip.timer      # OnCalendar *-*-* 03:00, Persistent
systemctl --user status sktrip.timer
journalctl --user -u sktrip.service -n 50        # last run output
```
Rollback = `systemctl --user disable --now sktrip.timer`. Preferred fleet path:
a **skscheduler** job (weekly cron + node-affinity, `notify: always`) that posts
the session summary to Chef via sk-alert/Telegram.

**Version/tag flow:** bump `pyproject.toml` `version` → add a dated
`CHANGELOG.md` entry (Keep-a-Changelog + SemVer) → tag.

**Front-end / Exposure:** `N/A — no public listener.` sktrip is a CLI and a
oneshot systemd job. It **binds no port** and answers no `:443` route. It is
strictly an *outbound* client: it connects to the trip model endpoint
(`192.168.0.100:8082`), Ollama (`192.168.0.100:11434`), and the skmem-pg
Postgres (`localhost:5432`). No Funnel/Caddy/Traefik ingress applies.

## 6. Configuration / Usage

Config file: `config/sktrip.toml` (loaded via `SKTripConfig.load`; falls back to
built-in code defaults if absent). Key fields:

```toml
memory_backend = "skmempg"        # "skmempg" (default) | "qdrant" (legacy)

[ollama]
host = "192.168.0.100"; port = 11434
trip_model = "qwen3.6-27b-abliterated"
trip_api = "openai"                        # "openai" | "ollama"
trip_base_url = "http://192.168.0.100:8082/v1"
sober_model = "qwen3.5:4b"
embed_model = "mxbai-embed-large"

[session]
output_dir = "/home/cbrd21/.skcapstone/agents/lumina/journal/sktrip"
```

**Environment:**
- `SKMEMORY_PG_DSN` — Postgres DSN for the skmem-pg flood backend
  (default `postgresql://postgres:skmemory@localhost:5432/skmemory`).
- `SKAGENT` / `SKCAPSTONE_AGENT` — which agent's memory rows to flood from
  (default `lumina`).

**Secrets — never inline a live secret.** The Postgres DSN and any backend API
key must come from environment/config on the host, not be hardcoded in tracked
source. The optional Qdrant/skvector `api_key` now defaults to empty and is read
from `SKTRIP_QDRANT_API_KEY`. (See SECURITY.md; the previously-tracked key was
flagged for rotation/removal.)

**Usage:**
```bash
sktrip status                                  # models + active backend + corpus size + sessions
sktrip dose microdose                          # gentlest run
sktrip dose psilocybin --intention "the nature of memory and identity"
sktrip dose dmt --burst --entity-contact       # short, intense, entity-contact
sktrip dose lsd --turns 15 --intention "connect sovereignty and mycology"
sktrip dose <sub> --no-integrate               # run without storing (timer uses this)
sktrip journal                                 # list past sessions
sktrip integrate <SESSION_ID>                  # re-run sober analysis
```

## 7. API / Reference

sktrip exposes **no network API** — its surface is the `sktrip` CLI (Click).

| Command | Args / flags | Effect |
|---|---|---|
| `sktrip status` | — | Prints models (trip/sober/embed), probes trip endpoint (`/models` for OpenAI, `/api/tags` for Ollama), reports **active** memory backend + live corpus size, session count, and substance profiles. |
| `sktrip dose SUBSTANCE` | `psilocybin\|dmt\|lsd\|microdose`; `-i/--intention`, `-t/--turns`, `--burst`, `--entity-contact`, `--no-integrate` | Runs a full SET→DOSE→EXPERIENCE→INTEGRATE→STORE session; prints top insights. |
| `sktrip integrate SESSION_ID` | session id from `journal` | Re-runs sober integration on a recorded JSONL session; writes `<session>.integration.md`. |
| `sktrip journal` | — | Tabulates past sessions (substance, date, duration, turns, tokens, peak, intention). |
| Global | `-c/--config PATH` | Path to `sktrip.toml`. |

Key internal symbols: `dose.generate(config, profile, prompt, max_tokens)`
(dual OpenAI/Ollama), `dose.build_dose_prompt(...)`, `dose.inject_disruption(...)`,
`SubstanceProfile.get(substance)`, `memory_flood.make_memory_flood(config)` →
`.get_corpus_size()` / distant-fragment pulls, `session.run_session(...)`.

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| `sktrip status` shows trip model "Not found" / offline while the model is up | **Known + fixed (Unreleased):** the trip model is served on the OpenAI-compatible endpoint (`:8082`), not Ollama. `status` now probes `trip_base_url/models` when `trip_api = "openai"`. Confirm `trip_api`/`trip_base_url` in `sktrip.toml`; verify `curl http://192.168.0.100:8082/v1/models`. |
| `status` reports `Vectors: 0` against a healthy corpus | **Known + fixed (Unreleased):** old code held a hardcoded Qdrant handle. `status` now reads the **active** backend via `make_memory_flood()`. Confirm `memory_backend = "skmempg"` and that Postgres is reachable (`SKMEMORY_PG_DSN`). |
| Trip generation hangs / times out | Trip model on `:8082` is loaded but slow (abliterated 27B). Raise `[ollama] timeout`; check VRAM/`.100` load; confirm the endpoint answers `/v1/chat/completions`. |
| `Sober ready: ✗` or `Embed ready: ✗` | `qwen3.5:4b` / `mxbai-embed-large` not pulled on Ollama `:11434`. `ollama list` on `.100`. |
| Entity-contact turns produce no entity prompt | **Known + fixed (Unreleased):** a dead `and not self.entity_contact` guard made the append unreachable. Update to current `freeassoc.py`. |
| `sktrip journal` empty | No sessions yet, or `[session] output_dir` differs from where runs wrote JSONL. Check `output_dir`. |
| Postgres connection refused | skmem-pg container down or wrong DSN. `docker ps | grep skmem-pg`; verify `localhost:5432`/`SKMEMORY_PG_DSN`. |

## 9. Maturity-tier + Version reference

- **Maturity tier:** `T0 — N/A (no key material)`. sktrip performs no
  cryptographic operations; it generates, exchanges, signs, verifies, wraps, and
  stores **no** key material. The crypto-component doc set
  (`docs/crypto-architecture.md`, CRYPTOGRAPHY_STANDARD / CRYPTO_AGILITY
  compliance lines) therefore does not apply.
- **VERSION_LIFECYCLE phase:** Incubating — a v3-class capability under active
  development, not yet a ratified Active-v2 platform service.
- **SemVer:** `0.1.0` (`pyproject.toml`).
- **Honest-claims note:** no security/crypto claims are made anywhere in this
  repo; "computational psilocybin" is a research metaphor for sampler
  perturbation, not a capability claim. No forbidden crypto terms apply.
