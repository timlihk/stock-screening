# Setup

## Prerequisites

- GitHub account with push access to `timlihk/stock-screening`.
- Cloudflare account (free tier works).
- `gh` CLI authenticated (`gh auth status`).
- `npx wrangler` available (Cloudflare Workers CLI).

## 1. GitHub side

### Fine-grained PAT for the Worker

The Worker needs to call GitHub's `workflow_dispatch` API on your behalf.

1. https://github.com/settings/personal-access-tokens/new
2. **Resource owner:** `timlihk` · **Repository access:** only `stock-screening`.
3. **Repository permissions:**
   - `Actions`: Read and write
   - `Metadata`: Read-only (auto)
   - `Contents`: Read-only
4. Copy the token — it starts with `github_pat_`.

### Workflow permissions

Settings → Actions → General → Workflow permissions → **Read and write** (so the
scheduled commits to `public/results/latest/` succeed).

## 2. Cloudflare side

### Install wrangler

```bash
cd worker
npm install -g wrangler   # or use npx
wrangler login            # opens browser
```

### Store the PAT as a Worker secret

```bash
cd worker
wrangler secret put GITHUB_TOKEN
# Paste the fine-grained PAT when prompted.
```

### Deploy the Worker

```bash
cd worker
wrangler deploy
```

First deploy prints your URL (e.g., `https://stock-screening.<account>.workers.dev`).
The Worker serves both the static landing page (`public/`) and the API endpoints.

### Custom domain (optional)

Cloudflare dashboard → Workers & Pages → `stock-screening` → Triggers → Custom
Domains → add `screen.yourdomain.com`.

## 3. First scan

1. Open the Worker URL.
2. Adjust thresholds (defaults: Nasdaq+NYSE, min $5, ADV ≥ $50M, setup ≥ 3, QM on).
3. Click **Queue scan**.
4. "Scan queued" appears; GitHub Actions runs the workflow (~5–15 min).
5. Run row changes from `queued` → `in_progress` → `success`.
6. Latest results auto-populate in the bottom card. Click **Open chart dashboard**.

## 4. Schedule

The workflow runs automatically every US trading day at ~17:15 ET (21:15 UTC,
30 min post-close). Change the cron in `.github/workflows/scan.yml` if desired.

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Worker not configured` | Missing secret | Re-run `wrangler secret put GITHUB_TOKEN` |
| `GitHub dispatch failed (404)` | PAT lacks Actions:Write | Regenerate PAT with correct scope |
| `GitHub dispatch failed (422)` | Workflow file not on `main` | Merge workflow to default branch first |
| Scan commits fail | Workflow permissions read-only | Flip to read/write in repo settings |
| Stale results card | CDN cache | Hard refresh (Cmd+Shift+R) |

## 6. Cost

- GitHub Actions: free tier = 2,000 min/mo for public repos. Each scan is ~10 min.
- Cloudflare Workers: free tier = 100k requests/day. API calls per session < 10.

Well within free tiers for daily personal use.
