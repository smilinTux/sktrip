# Contributing to sktrip

sktrip is part of the SKWorld sovereign ecosystem. Contributions follow the
[sk-standards](https://github.com/smilinTux/sk-standards) doc/SOP bar.

## Branch model

- `main` — stable, releasable tip.
- Work on a feature/fix branch: `feat/<slug>`, `fix/<slug>`, or an
  integration branch (`integrate/<name>`). Never commit directly to `main`.
- Open a PR into `main`; keep branches focused and rebased on the current tip.

## Commit convention

- Use clear, imperative subjects, optionally Conventional-Commit-prefixed
  (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- **Every commit must end with a `Co-Authored-By` trailer** identifying the AI
  co-author, e.g.:

  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## Test gate (required)

The green pytest bar blocks merge. Before opening or updating a PR:

```bash
. .venv/bin/activate
pytest -q          # must be green
```

- Add/extend tests under `tests/` for any behavior change. Tests must not
  require a live model or corpus — mock network/DB calls.
- If you touch `status`, the daemon, or a backend, update the matching
  `tests/test_status.py` / `tests/test_daemon.py` / `tests/test_*backend*.py`.

## Docs gate

Per sk-standards, a change is not "done" until the docs still let a stranger
build/test/deploy the repo:

- Update **`CHANGELOG.md`** (Keep-a-Changelog + SemVer, dated) for any
  user-visible change.
- Update **`SOP.md`** if you change architecture, config, the CLI surface, or
  add a troubleshooting-worthy failure mode.
- Honest-claims gate: no capability/security claim without in-repo evidence;
  no forbidden crypto terms.

## Review path

Open a PR into `main`, ensure pytest is green and docs are updated, and request
review from the maintainer (Chef). Do not force-push shared branches.
