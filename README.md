# stock-screening

Self-hosted SEPA (Mark Minervini) + Qullamaggie screen for the US equity universe.
Runs on GitHub Actions and deploys a Cloudflare Worker that serves the landing page
and proxies scan requests into the GitHub workflow dispatch API.

## Architecture

```
browser ──HTTP──▶ Cloudflare Worker ──GH API──▶ GitHub Actions ──commits──▶ repo
                       │                              │
                       └── serves static /public/     └── writes /public/results/latest/
```

- **Scanner** (`scanner/`): Python. Fetches Nasdaq Trader symbol directory,
  applies liquidity + SEPA trend template + VCP-lite + Deepvue/Minervini
  extensions + optional Qullamaggie momentum layer.
- **Workflow** (`.github/workflows/scan.yml`): `workflow_dispatch` with inputs,
  plus daily cron at 17:15 ET. Commits `public/results/latest/` back to `main`.
- **Landing page** (`public/index.html`): form that posts thresholds to the Worker.
- **Worker** (`worker/`): thin API layer (`/api/scan`, `/api/runs`, `/api/run?id=`)
  that holds the GitHub PAT server-side, and serves the `public/` directory.

## Signal stack

| Layer | What it checks | Source |
|---|---|---|
| Trend template (8 points) | MA alignment, 52w position, RS > 0 vs SPY | Minervini |
| VCP-lite (0-4) | Contracting pullbacks, range tightening, volume dry-up | SEPA literature |
| Deepvue/Minervini (+4) | ATR compression, 5-day tight range, Power Play, breakout confirmation | Deepvue partnership |
| Qullamaggie (0-4) | 1/3/6-mo top-momentum, 10/20 EMA ride, ADR% ≥ 5%, tight consolidation | Kristjan Kullamägi |

Final `setup_score` = 0–8. Shortlist threshold is configurable (default 3).

## Local run

```bash
pip install -r scanner/requirements.txt
python3 scanner/sepa_scan_universe.py --help
python3 scanner/sepa_scan_universe.py \
    --exchanges nasdaq,nyse \
    --min-price 5 \
    --min-adv-usd 50000000 \
    --use-qullamaggie \
    --min-setup-score 3
```

## Deploy (one-time setup)

See `docs/SETUP.md` for the full checklist. Quick version:

1. Create a GitHub fine-grained PAT with `Actions: Read & Write` on this repo.
2. `cd worker && npx wrangler secret put GITHUB_TOKEN` (paste PAT).
3. `cd worker && npx wrangler deploy`.
4. Hit the printed `*.workers.dev` URL — landing page lives there.

## License

MIT, tooling only. Nothing here is investment advice.
