# Changelog

All notable changes to sktrip are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-07-03

### Fixed
- `sktrip status` false negatives: the trip model is now probed on the
  correct endpoint. When it is served via the OpenAI-compatible server
  (abliterated qwen3.6-27b @ `:8082`) `status` queries `/models` there
  instead of Ollama's `/api/tags`, so a healthy trip model is no longer
  reported as "Not found". Ollama connectivity is reported separately for
  the sober and embed models.
- `sktrip status` now reports the **active** memory backend. It reads the
  live corpus via `make_memory_flood()` and shows skmem-pg (Postgres +
  pgvector, mxbai-embed-large) by default, or Qdrant when selected —
  fixing a hardcoded Qdrant handle that showed "Vectors: 0" against a
  healthy ~17k-row skmem-pg corpus.
- Removed dead entity-contact branch in `freeassoc.py`: the guard
  `and not self.entity_contact` made the entity-prompt append unreachable
  (`extra_prompt` is only set when entity contact is on). Entity-contact
  turns now correctly append their labeled prompt, and the session
  intention is carried only on the first turn.

### Tests
- Added `tests/test_status.py` and `tests/test_daemon.py` covering the
  status backend/endpoint reporting and the daemon fix.
