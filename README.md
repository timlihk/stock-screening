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
  applies liquidity, then scores separate breakout families instead of one
  blended momentum bucket: SEPA/VCP, power-play continuation, plus vendored
  Andy-Roger Qullamaggie breakout and episodic-pivot scans.
- **Workflow** (`.github/workflows/scan.yml`): `workflow_dispatch` with inputs,
  plus daily cron at 8:00 AM EST. Commits `public/results/latest/` back to `main`.
- **Landing page** (`public/index.html`): form that posts thresholds to the Worker.
- **Worker** (`worker/`): thin API layer (`/api/scan`, `/api/runs`, `/api/run?id=`)
  that holds the GitHub PAT server-side, and serves the `public/` directory.

## Signal stack

| Layer | What it checks | Source |
|---|---|---|
| Stock quality | Trend template, longer-term RS, distance from highs, orderly structure | Minervini + leadership filters |
| Entry quality | Base quality, pivot proximity, quiet pullback, accumulation vs distribution | First-principles breakout logic |
| Setup families | `sepa_vcp`, `power_play`, `qm_breakout`, `qm_episodic_pivot` | Minervini, Jeff Sun, Kullamägi |
| Regime filter | SPY/QQQ/IWM trend state changes minimum setup and entry thresholds | Breakout tape filter |

The scanner no longer treats every trader input as one additive `setup_score`.
Instead it classifies each ticker across the four setup families, allows
multi-membership when a stock genuinely fits more than one playbook, then ranks
within a regime-aware shortlist using:

- `primary_setup`: best-fitting breakout archetype
- `setup_score`: strength inside that archetype
- `entry_score`: how usable the entry is now
- `leadership_score`: how strong the stock itself is

## Local run

```bash
pip install -r scanner/requirements.txt
python3 scanner/sepa_scan_universe.py --help
python3 scanner/sepa_scan_universe.py \
    --exchanges nasdaq,nyse \
    --min-price 5 \
    --min-adv-usd 50000000 \
    --use-qullamaggie \
    --min-setup-score 6
```

## QM source

The Qullamaggie families are not hand-rolled anymore. They are sourced from a
vendored copy of Andy-Roger's [qullamaggie-scanner](https://github.com/Andy-Roger/qullamaggie-scanner):

- `qm_breakout`: his breakout rules, mapped into this repo's 0-8 family score
- `qm_episodic_pivot`: his premarket EP rules, also mapped into the 0-8 family score

The raw Andy-Roger scores are still kept in the results CSV as
`qm_breakout_vendor_score` and `qm_episodic_pivot_vendor_score`.

## Deploy (one-time setup)

See `docs/SETUP.md` for the full checklist. Quick version:

1. Create a GitHub fine-grained PAT with `Actions: Read & Write` on this repo.
2. `cd worker && npx wrangler secret put GITHUB_TOKEN` (paste PAT).
3. `cd worker && npx wrangler deploy`.
4. Hit the printed `*.workers.dev` URL — landing page lives there.

## License

MIT, tooling only. Nothing here is investment advice.
