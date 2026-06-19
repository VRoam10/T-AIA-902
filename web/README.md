# RL Benchmark Dashboard

Static Next.js dashboard that visualises the benchmark results produced by the
Python pipeline. It reads JSON from `public/data` and exports a fully static
site (no server required).

## Stack

- Next 16 (App Router, static export) · React 19 · TypeScript (strict)
- Tailwind 4 · Recharts · Biome · Vitest

## Data flow

```
outputs/benchmarks/   (Python pipeline writes run folders + index.json)
        │  python scripts/sync_web_data.py   (or: npm run sync)
        ▼
web/public/data/      (JSON mirrored here)
        │  next build (output: "export")
        ▼
web/out/              (static site, deployable as-is)
```

The Python side is the single source of truth; the dashboard only reads JSON.

## Develop

```bash
cd web
npm install
npm run sync     # copy outputs/benchmarks JSON into public/data
npm run dev      # http://localhost:3000
```

## Build & deploy

```bash
npm run build    # static export into web/out/
```

`web/out/` is a static site. Deploy options:

- **Vercel** — import the `web/` directory; framework preset Next.js.
- **GitHub Pages** — publish `web/out/`. If served from a sub-path, set
  `NEXT_PUBLIC_BASE_PATH=/<repo>` before building so asset URLs resolve.

## Quality

```bash
npm run lint     # Biome (strict, tab indent, no explicit any)
npm run format   # Biome autofix
npm run test     # Vitest
```

## Pages

- `/` — list of all runs (from `index.json`).
- `/run/[id]` — per-run detail; charts adapt to the benchmark kind
  (single, multi-seed, comparison, grid-search).
- `/compare` — overlay reward curves and eval rewards across selected runs.
