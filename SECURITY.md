# Security Policy — sktrip

sktrip is a **T0 — N/A (no key material)** component: it performs no
cryptographic operations and generates, exchanges, signs, verifies, wraps, or
stores no keys. It is, however, a program that **reads a private memory corpus**
and **sends corpus content to a local LLM endpoint** — so its security posture
is about data handling and outbound calls, not crypto.

## Reporting a vulnerability

Report privately to the maintainer (Chef / David) via the SKWorld channels
(Telegram DM / sk-alert) or an encrypted email to the address in
`pyproject.toml`. Do **not** open a public issue for a sensitive report. Please
include: affected file/command, reproduction, and impact. There is no bug-bounty;
this is sovereign research infrastructure.

## Threat model

**Assets.**
- The **memory corpus** (skmemory / skmem-pg `memories` rows) — potentially
  sensitive personal/agent memory that sktrip floods into prompts.
- **Session transcripts** (JSONL under `[session] output_dir`) — verbatim model
  output plus the memory fragments injected. Treat as sensitive; they live
  outside git (`sessions/` gitignored, journal under `~/.skcapstone/...`).
- The **trip/sober/embed model endpoints** and the **Postgres DSN**.

**Trust boundaries.**
- sktrip is an **outbound client only** — it binds no port and exposes no
  network surface. Ingress attack surface is therefore nil.
- It sends corpus fragments to the trip model endpoint
  (`192.168.0.100:8082`, OpenAI-compatible) and to Ollama (`:11434`). These are
  assumed to be **local/tailnet** services on trusted hosts. Corpus content
  leaves the process boundary to those endpoints — do **not** point
  `trip_base_url`/Ollama at an untrusted or public model host.
- It connects to Postgres (`localhost:5432`) with the DSN from
  `SKMEMORY_PG_DSN`.

**Notable risks.**
- **Abliterated model.** The trip model is intentionally refusal-suppressed
  (the point of "trip mode"). Output is unfiltered by design and may be
  disturbing or unsafe to surface unreviewed; the sober-integration pass
  (T=0.3) is the gate that decides what persists. Only novelty ≥ 6 insights are
  written back to permanent memory.
- **Prompt/corpus injection.** Because arbitrary corpus rows are flooded into
  prompts, a poisoned memory row could steer generation. sktrip does not
  sanitize corpus content; the sober-integration + human review are the
  controls.
- **Write-back to memory.** `integrate` persists insights into skmemory. A
  compromised integration model could inject junk into permanent memory; runs
  can be executed with `--no-integrate` to avoid any write.

## Secret handling

- **Never inline a live secret in tracked source.** Secrets (Postgres DSN,
  any backend API key) must come from environment/host config
  (`SKMEMORY_PG_DSN`, `SKAGENT`) or an untracked local config, not hardcoded.
- Session journals may contain sensitive corpus content — they are kept out of
  git by `.gitignore` and written under `~/.skcapstone/agents/<agent>/journal/`.

> **Known issue (flagged for remediation):** the legacy Qdrant backend carries a
> hardcoded `api_key` in `sktrip/config.py` and `config/sktrip.toml`. This is a
> tracked secret and violates the rule above. It should be **rotated and moved
> to environment/config** (or removed with the legacy Qdrant path). The default
> skmem-pg path does not use it.

## Dependency posture

Small dependency set (`httpx`, `qdrant-client`, `numpy`, `rich`, `click`,
`psycopg`, `tomli` on <3.11). Pin/update via `pyproject.toml`; there is no
network listener, so the primary supply-chain concern is the model/DB clients.
No cryptographic dependencies are present.
