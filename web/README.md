# EyeSim Lab — web app

A deployed, database-backed demo of the `eyesim` pipeline: upload a photo,
apply real prescription physics to it (Simulate mode), or get an
experimental CNN estimate from a real eye photo (Diagnose mode). See
`proposal/code_walkthrough.md` for what's actually running underneath.

## Backend — local dev without Docker (fastest path)

Zero-config for the stateless routes; everything else needs a `.env`.

```bash
cd /path/to/eyesim
pip install -e ".[ml]"                 # eyesim itself (physics + PupilNet)
cd web/backend
pip install -r requirements.txt
cp .env.example .env                   # defaults to a local sqlite file
uvicorn app.main:app --reload --port 8000
```

`GET /api/health` and `GET /api/psf` work immediately (no DB touched).
`/api/cases/simulate`, `/api/cases/diagnose`, `/api/gallery`, `/api/stats`,
`/api/share/*` need the DB — the sqlite default handles that with zero
setup, and images fall back to local-disk storage (served at `/static/...`)
when Supabase isn't configured (see `app/services/storage.py`).

Regenerating `app/services/train_stats.json` (only needed if PupilNet is
retrained on a different dataset — the committed file already reflects the
current `checkpoints/best.pt`):

```bash
python scripts/compute_train_stats.py   # needs data/train.npz present locally
```

Tests:

```bash
pytest -q
```

## Backend — Docker / docker-compose (Postgres parity with prod)

```bash
cd web
docker-compose up --build
```

Runs a local Postgres (`db`) + the backend (`backend`, port 8000) with live
code reload. First run needs the schema created:

```bash
docker-compose run --rm backend alembic upgrade head
```

> Not yet verified in this environment (no `docker` binary available
> during development) — the Dockerfile and compose file are written and
> YAML-validated, but `docker-compose up` itself needs to be run on a
> machine with Docker before trusting it fully. Do that before deploying.

## Frontend

```bash
cd web/frontend
npm install
cp .env.example .env.local     # VITE_API_BASE_URL, defaults to localhost:8000
npm run dev
```

Pages: `/` (3D optics bench), `/simulate`, `/diagnose`, `/gallery`,
`/share/:slug`. All built on `react-three-fiber` (the 3D scene), a raw
WebGL2 shader (`src/shaders/screenFilter.ts`, the live GPU screen filter),
and a hand-rolled dark "lab/HUD" design system in `src/styles/tokens.css`
— no UI framework.

## Deployment

Backend → Fly.io (Dockerized, `web/backend/Dockerfile`, build context = repo
root, config at repo-root `fly.toml`). Database + image storage → Supabase
(Postgres + Storage). Frontend → Vercel (`web/frontend`, SPA rewrites in
`web/frontend/vercel.json`).

`flyctl` and `vercel` CLIs are installed. Steps once accounts/credentials
exist (see the project's top-level conversation for exactly what's needed):

```bash
# 1. Supabase: create a project, a public Storage bucket (e.g. eyesim-cases),
#    and grab the Postgres connection string + service role key.

# 2. Backend, from repo root:
fly launch --no-deploy         # or `fly apps create` if fly.toml's app name is taken
fly secrets set DATABASE_URL='postgresql+psycopg2://...' \
                SUPABASE_URL='https://xxxx.supabase.co' \
                SUPABASE_SERVICE_KEY='...' \
                SUPABASE_BUCKET='eyesim-cases' \
                CORS_ORIGINS='["https://<vercel-domain>"]'
fly deploy
fly ssh console -C "cd /repo/web/backend && alembic upgrade head"

# 3. Frontend:
cd web/frontend
vercel link
vercel env add VITE_API_BASE_URL   # https://<fly-app>.fly.dev
vercel --prod
```



### Redeploying after a change

Live site: https://frontend-dun-ten-37.vercel.app (Vercel project `frontend`),
backed by the API at https://eyesim-lab-api.fly.dev.

**Frontend** (anything under `web/frontend`):

```bash
cd web/frontend
npm run build      # optional: catches type errors before uploading
vercel --prod      # builds on Vercel and updates the live URL
```

That is the whole step. The folder is already linked to the Vercel project
(`web/frontend/.vercel/project.json`), and `VITE_API_BASE_URL` is stored in
Vercel's Production environment, so neither needs repeating. If the CLI says
"No existing credentials found", run `vercel login` once and retry. To undo a
bad deploy, run `vercel rollback`.

If `npm run build` hangs locally, iCloud has evicted files in `node_modules`
(this repo lives on the Desktop). Skip the local build and run `vercel --prod`
directly, or `rm -rf node_modules && npm ci` first.

**Backend** (anything under `web/backend` or the `eyesim` package), from the
repo root:

```bash
fly deploy
fly ssh console -C "cd /repo/web/backend && alembic upgrade head"   # only if you added a migration
```

The frontend does not need redeploying after a backend-only change.
