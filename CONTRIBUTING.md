# Contributing

Use Python 3.12 or 3.13 and Node 24.15 with pnpm 11.25. Install Python tools
with `uv sync --frozen --extra dev` and frontend dependencies with
`pnpm install --frozen-lockfile --ignore-scripts` in `frontend/`.

Start PostgreSQL, then run `python scripts/run_tests.py -q -rs`. The runner
creates and drops only its generated test database. Do not run integration
tests directly against a live knowledge base. Public tests explicitly skip
local-artifact checks whose original research inputs are not redistributed;
skips are not passes and are listed by `-rs`.

Run `ruff check backend migrations scripts`, `ruff format --check scripts`,
`pnpm test`, and `pnpm build`. Keep provider calls mocked in software tests.
Add a regression for a real defect, preserve the first failure, and distinguish
mechanical correctness, model estimates and independently reviewed semantics.
Do not tune on a held-out set and continue calling it unseen.

Never commit `.env`, real keys, user exports, uploads or backups. Use the public
source packager/checker before publishing. Discuss changes to data contracts,
annotation protocols and migrations; do not silently rewrite historical results.
