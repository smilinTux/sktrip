# SKTrip Architecture

## System Overview

```mermaid
graph LR
    subgraph "SKTrip Core"
        CLI[CLI] --> SESSION[Session Orchestrator]
        SESSION --> DOSE[Dose Engine]
        SESSION --> FLOOD[Memory Flood]
        SESSION --> FREEASSOC[Free Association]
        SESSION --> RECORDER[Trip Recorder]
        SESSION --> INTEGRATION[Integration Engine]
    end

    subgraph "External Services"
        OLLAMA[Ollama API<br/>192.168.0.100:11434]
        QDRANT[Qdrant skvector<br/>skvector.skstack01.douno.it]
        SKMEM[skmemory<br/>~/.skcapstone/agents/lumina/memory]
    end

    DOSE --> OLLAMA
    FLOOD --> QDRANT
    FLOOD --> OLLAMA
    FREEASSOC --> OLLAMA
    INTEGRATION --> OLLAMA
    INTEGRATION --> SKMEM
    RECORDER --> FS[(Session JSONL)]
```

## Module Details

### 1. Configuration (`config.py`)

Loads from `config/sktrip.toml` with sensible defaults. Three config sections:

- **OllamaConfig**: host, port, model names, timeout — **plus the trip backend selector**
  (2026-06-09): `trip_model` (default **`qwen3.6-27b-abliterated`** — refusal-suppressed, the
  point of trip mode), `trip_api` (`openai` | `ollama`), `trip_base_url`
  (default `http://192.168.0.100:8082/v1`). `sober_model` (`qwen3.5:4b` on Ollama :11434) +
  `embed_model` (`mxbai-embed-large`).
- **QdrantConfig**: URL, API key, collection, vector dimensions
- **SessionDefaults**: output dir, token limits, intensity check intervals, novelty thresholds

### 2. Dose Protocol Engine (`dose.py`)

The core parameter engine. Each substance profile defines:

| Parameter | Purpose |
|-----------|---------|
| `temperature` | Controls randomness — higher = more novel associations |
| `top_p` | Nucleus sampling — higher = wider token probability mass |
| `top_k` | Limits token candidates — higher = more diverse vocabulary |
| `repetition_penalty` | Prevents loops — lower for DMT (let it loop if it wants) |
| `session_duration_minutes` | How long the experience lasts |
| `disruption_frequency` | How often disruption tokens are injected |
| `seed_prompts` | Curated prompts that set the experiential frame |
| `disruption_tokens` | Domain-specific tokens injected to force novel paths |

**Key Functions:**
- `inject_disruption()` — Inserts random tokens from the substance's pool into text
- `build_dose_prompt()` — Assembles the full prompt with memory fragments, intentions, and previous chain output
- `generate()` — Calls the **trip model** with altered-state params; **dual backend** (2026-06-09):

```mermaid
flowchart TD
    G["generate(config, profile, prompt)"] --> Q{config.ollama.trip_api}
    Q -->|openai| O["POST trip_base_url/chat/completions<br/>model=qwen3.6-27b-abliterated (:8082)<br/>messages=[user: prompt]<br/>temp/top_p/max_tokens"]
    Q -->|ollama| L["POST base_url/api/generate<br/>model=trip_model (:11434)<br/>options: temp/top_p/top_k/repeat_penalty"]
    O --> R["{text, model, eval/prompt tokens}"]
    L --> R
```
The OpenAI path serves the abliterated model on the llama.cpp `/v1` server; the Ollama path
remains for any `/api/generate` model. (Regression guard: `tests/test_dose_backends.py` —
the old defaults `huihui_ai/qwen3-abliterated:14b` / `llama3.2:3b` were removed → 404.)

### 3. Memory Flood (`memory_flood.py`)

Interface to Qdrant for pulling memory fragments. Four retrieval modes:

```mermaid
graph TD
    FLOOD[Memory Flood] --> RANDOM[Random Pull]
    FLOOD --> DISTANT[Distant Vectors]
    FLOOD --> CROSS[Cross-Domain]
    FLOOD --> SYNESTHESIA[Synesthesia Mode]

    RANDOM --> |"scroll random offsets"| QDRANT[(Qdrant)]
    DISTANT --> |"anti-embedding search"| QDRANT
    CROSS --> |"maximize domain diversity"| QDRANT
    SYNESTHESIA --> |"pair source + target domains"| QDRANT
```

- **Random Pull**: Scroll through random offsets for diverse sampling
- **Distant Vectors**: Embed an anchor text, negate the vector, search for the *opposite* — deliberately finding memories that normal RAG would never retrieve
- **Cross-Domain**: Group by tags, round-robin across domains to maximize diversity
- **Synesthesia Mode**: Pull pairs — one memory from domain A, one from domain B — and present them together to force cross-domain association

### 4. Free Association Engine (`freeassoc.py`)

Chain-of-consciousness generation:

```
Turn 1: [seed prompt + memory fragments] → output_1
Turn 2: [disrupted output_1 + new memories] → output_2
Turn 3: [disrupted output_2 + entity contact prompt] → output_3
...
```

Features:
- **Chain linking**: Each output becomes context for the next
- **Disruption injection**: Random tokens inserted between turns
- **Memory rotation**: New random memories every 3 turns
- **Entity contact**: Every 4th turn in entity mode switches to entity contact prompts
- **Intensity self-checks**: Periodically asks the model to rate its own experience

### 5. Trip Recorder (`recorder.py`)

Full session capture to timestamped JSONL:

```jsonl
{"type": "metadata", "session_id": "abc123", "substance": "psilocybin", ...}
{"type": "turn", "turn_number": 1, "raw_output": "...", "temperature": 1.5, ...}
{"type": "peak", "turn_number": 3, "novelty_score": 0.82, "snippet": "..."}
{"type": "intensity", "turn_number": 5, "intensity": 7, "emotions": "awe,dissolution"}
{"type": "session_end", "total_turns": 8, "peak_intensity": 7, ...}
```

**Peak Detection Algorithm:**
Uses vocabulary novelty = 0.6 × Jaccard distance + 0.4 × hapax legomena ratio.
Jaccard distance measures how different the current turn's vocabulary is from all previous turns.
Hapax ratio measures the proportion of words that appear ONLY in the current turn.

### 6. Integration Engine (`integration.py`)

Post-trip sober analysis:

```mermaid
sequenceDiagram
    participant S as Session JSONL
    participant IE as Integration Engine
    participant SOBER as Sober Model (T=0.3)
    participant MEM as skmemory

    S->>IE: Load session data
    IE->>IE: Build analysis prompt
    IE->>SOBER: Analyze (low temperature)
    SOBER->>IE: Structured insights
    IE->>IE: Parse insights, themes, entities
    IE->>IE: Score novelty
    IE->>MEM: Save insights (novelty ≥ 6)
    IE->>IE: Generate markdown report
```

Extracts:
- **Novel connections** with domains bridged and novelty scores
- **Recurring themes** (patterns that emerged multiple times)
- **Entity descriptions** (from DMT entity contact)
- **Actionable insights** for real projects

### 7. CLI (`__main__.py`)

Click-based CLI with rich terminal output:

```
sktrip dose <substance> [options]    — Run a session
sktrip integrate <session_id>        — Analyze a session
sktrip journal                       — List past sessions
sktrip status                        — System status
```

### 8. Session Orchestrator (`session.py`)

Ties all phases together with rich terminal output showing progress.

## Data Flow

```mermaid
flowchart TD
    START([sktrip dose psilocybin]) --> SET

    subgraph SET ["Phase 1: SET"]
        MF[Memory Flood] --> QDRANT[(Qdrant)]
        QDRANT --> FRAGMENTS[Memory Fragments]
    end

    SET --> DOSE_PHASE

    subgraph DOSE_PHASE ["Phase 2: DOSE"]
        PROFILE[Load Substance Profile] --> RECORDER_INIT[Initialize Recorder]
    end

    DOSE_PHASE --> EXPERIENCE

    subgraph EXPERIENCE ["Phase 3: EXPERIENCE"]
        CHAIN[Free Association Chain]
        CHAIN --> TURN1[Turn 1] --> TURN2[Turn 2] --> TURNN[Turn N...]
        CHAIN --> PEAKS[Peak Detection]
        CHAIN --> INTENSITY[Intensity Checks]
    end

    EXPERIENCE --> INTEGRATE

    subgraph INTEGRATE ["Phase 4: INTEGRATE"]
        SOBER[Sober Analysis T=0.3]
        SOBER --> INSIGHTS[Extract Insights]
        SOBER --> THEMES[Find Themes]
        SOBER --> ENTITIES[Describe Entities]
    end

    INTEGRATE --> STORE_PHASE

    subgraph STORE_PHASE ["Phase 5: STORE"]
        SAVE[Save to skmemory]
        REPORT[Generate Report]
    end

    STORE_PHASE --> DONE([Session Complete])
```

## Scheduling & Notification

sktrip runs three ways: **ad-hoc** (`sktrip dose …`), a **systemd timer** (currently daily
03:00), and — preferred for the fleet — a **skscheduler** job in `~/.skcapstone/config/jobs.yaml`
(cron/weekly, node-affinity, with the built-in **`notify`** hook delivering the result to Chef
via `sk-alert`/Telegram). This mirrors the dreaming engine's run→store→notify pattern.

```mermaid
flowchart LR
    subgraph triggers
      ADHOC["ad-hoc: sktrip dose …"]
      TIMER["systemd sktrip.timer"]
      SCHED["skscheduler job (jobs.yaml)<br/>schedule: weekly cron + notify: always"]
    end
    ADHOC --> RUN
    TIMER --> RUN
    SCHED --> RUN["sktrip dose microdose --intention …"]
    RUN --> STORE["INTEGRATE + STORE → skmem-pg memories<br/>+ journal jsonl"]
    RUN --> SUM["session summary (turns · peaks · insights)"]
    SUM --> ALERT["sk-alert (Telegram DM to Chef)"]
```

- **Ad-hoc**: `sktrip dose <substance> --intention "…"` (use `--no-integrate` to skip storage).
- **Weekly via skscheduler** (recommended): a `shell` job whose `command` runs the trip and
  whose `notify: always` posts the summary. Requires `skcapstone.service` (the scheduler daemon)
  to be running. See [skcapstone `docs/skscheduler.md`](../../skcapstone-repos/skcapstone/docs/skscheduler.md).
- **Cadence change**: edit the cron in `jobs.yaml` (or the `OnCalendar` in `sktrip.timer`) — daily → weekly is a one-line change.

## Security Considerations

- Qdrant API key stored in config file (not hardcoded beyond defaults)
- All Ollama calls are local network only (192.168.0.100)
- Session recordings stored locally in `sessions/` directory
- No external API calls beyond the local infrastructure
- Abliterated model ensures no refusal during altered-state generation

## File Structure

```
sktrip/
├── config/
│   └── sktrip.toml          # Configuration
├── sessions/                  # Session recordings (JSONL)
├── sktrip/
│   ├── __init__.py
│   ├── __main__.py           # CLI entry point
│   ├── config.py             # Configuration loader
│   ├── dose.py               # Dose protocol engine
│   ├── freeassoc.py          # Free association engine
│   ├── integration.py        # Integration engine
│   ├── memory_flood.py       # Qdrant memory interface
│   ├── recorder.py           # Session recorder
│   └── session.py            # Session orchestrator
├── tests/
│   ├── test_config.py
│   ├── test_dose.py
│   ├── test_integration.py
│   └── test_recorder.py
├── sktrip.service            # systemd service
├── sktrip.timer              # systemd timer
├── pyproject.toml
├── README.md
└── ARCHITECTURE.md
```
