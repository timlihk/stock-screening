// Cloudflare Worker: proxies the landing-page form POST to GitHub's
// workflow_dispatch API, keeping the PAT server-side. Also serves the static
// landing page from the linked Pages deployment.
//
// Required secrets (configure via `wrangler secret put`):
//   GITHUB_TOKEN   -- fine-grained PAT with Actions: Read & Write on this repo
//
// Required vars (wrangler.toml):
//   GITHUB_REPO    -- "timlihk/stock-screening"
//   WORKFLOW_FILE  -- "scan.yml"

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    if (url.pathname === "/api/scan" && request.method === "POST") {
      return handleScanDispatch(request, env);
    }

    if (url.pathname === "/api/runs" && request.method === "GET") {
      return handleListRuns(env);
    }

    if (url.pathname === "/api/run" && request.method === "GET") {
      return handleRunStatus(url, env);
    }

    // Proxy /results/* from GitHub raw so CI-committed scan outputs show up
    // without redeploying the Worker. Cached at CF edge for 60s.
    if (url.pathname.startsWith("/results/")) {
      return proxyGithubRaw(url.pathname, env);
    }

    // Fallback: serve bundled static assets (landing page)
    if (env.ASSETS) {
      if (url.pathname.endsWith("/")) url.pathname += "index.html";
      return env.ASSETS.fetch(new Request(url.toString(), request));
    }

    return new Response("Not found", { status: 404, headers: CORS_HEADERS });
  },
};

async function handleScanDispatch(request, env) {
  const { GITHUB_TOKEN, GITHUB_REPO, WORKFLOW_FILE } = env;
  if (!GITHUB_TOKEN || !GITHUB_REPO || !WORKFLOW_FILE) {
    return json({ error: "Worker not configured: missing GITHUB_TOKEN / GITHUB_REPO / WORKFLOW_FILE" }, 500);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Body must be JSON" }, 400);
  }

  const inputs = normalizeInputs(body);
  const err = validateInputs(inputs);
  if (err) return json({ error: err }, 400);

  const ghResp = await fetch(
    `https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`,
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${GITHUB_TOKEN}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "stock-screening-worker",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: "main", inputs }),
    },
  );

  if (!ghResp.ok) {
    const text = await ghResp.text();
    return json({ error: `GitHub dispatch failed (${ghResp.status})`, detail: text }, 502);
  }

  return json({
    ok: true,
    message: "Scan queued. It usually takes 5-15 minutes.",
    inputs,
    runs_url: `https://github.com/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}`,
  });
}

async function handleListRuns(env) {
  const { GITHUB_TOKEN, GITHUB_REPO, WORKFLOW_FILE } = env;
  const resp = await fetch(
    `https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}/runs?per_page=10`,
    {
      headers: {
        "Authorization": `Bearer ${GITHUB_TOKEN}`,
        "Accept": "application/vnd.github+json",
        "User-Agent": "stock-screening-worker",
      },
    },
  );
  if (!resp.ok) return json({ error: `List failed: ${resp.status}` }, 502);
  const data = await resp.json();
  const runs = (data.workflow_runs || []).map((r) => ({
    id: r.id,
    status: r.status,
    conclusion: r.conclusion,
    created_at: r.created_at,
    updated_at: r.updated_at,
    actor: r.actor?.login,
    event: r.event,
    html_url: r.html_url,
  }));
  return json({ runs });
}

async function proxyGithubRaw(pathname, env) {
  const { GITHUB_REPO } = env;
  // Extension-less paths (e.g. /results/latest/charts) — assume .html, since
  // the browser may have stripped the extension via CF Pages' html_handling
  // default on the landing-page link.
  const hasExt = /\.[a-z0-9]{1,5}$/i.test(pathname);
  const resolvedPath = hasExt ? pathname : `${pathname}.html`;
  const target = `https://raw.githubusercontent.com/${GITHUB_REPO}/main/public${resolvedPath}`;
  const resp = await fetch(target, {
    cf: { cacheTtl: 60, cacheEverything: true },
    headers: { "User-Agent": "stock-screening-worker" },
  });
  if (!resp.ok) {
    return new Response(`Not found: ${pathname}`, {
      status: resp.status,
      headers: CORS_HEADERS,
    });
  }
  const contentType = resolvedPath.endsWith(".json") ? "application/json"
    : resolvedPath.endsWith(".csv") ? "text/csv"
    : resolvedPath.endsWith(".html") ? "text/html; charset=utf-8"
    : resolvedPath.endsWith(".log") ? "text/plain"
    : resolvedPath.endsWith(".txt") ? "text/plain"
    : "application/octet-stream";
  return new Response(resp.body, {
    status: 200,
    headers: {
      "content-type": contentType,
      "cache-control": "public, max-age=60",
      ...CORS_HEADERS,
    },
  });
}

async function handleRunStatus(url, env) {
  const id = url.searchParams.get("id");
  if (!id) return json({ error: "Missing id" }, 400);
  const { GITHUB_TOKEN, GITHUB_REPO } = env;
  const resp = await fetch(
    `https://api.github.com/repos/${GITHUB_REPO}/actions/runs/${id}`,
    {
      headers: {
        "Authorization": `Bearer ${GITHUB_TOKEN}`,
        "Accept": "application/vnd.github+json",
        "User-Agent": "stock-screening-worker",
      },
    },
  );
  if (!resp.ok) return json({ error: `Status failed: ${resp.status}` }, 502);
  const r = await resp.json();
  return json({
    id: r.id,
    status: r.status,
    conclusion: r.conclusion,
    html_url: r.html_url,
    updated_at: r.updated_at,
  });
}

function normalizeInputs(body) {
  return {
    exchanges: String(body.exchanges || "nasdaq,nyse"),
    min_price: String(body.min_price ?? "5"),
    min_adv_usd: String(body.min_adv_usd ?? "50000000"),
    min_setup_score: String(body.min_setup_score ?? "4"),
    use_qullamaggie: String(body.use_qullamaggie ?? "true"),
    use_finviz_prefilter: String(body.use_finviz_prefilter ?? "true"),
    universe_source: String(body.universe_source ?? "sec_edgar"),
  };
}

function validateInputs(i) {
  const validExchanges = new Set(["nasdaq", "nyse", "amex", "arca", "bats"]);
  const exchanges = i.exchanges.split(",").map((x) => x.trim().toLowerCase());
  for (const ex of exchanges) if (!validExchanges.has(ex)) return `Invalid exchange: ${ex}`;
  const nPrice = Number(i.min_price);
  if (!Number.isFinite(nPrice) || nPrice < 0 || nPrice > 10000) return "min_price out of range";
  const nAdv = Number(i.min_adv_usd);
  if (!Number.isFinite(nAdv) || nAdv < 0 || nAdv > 1e12) return "min_adv_usd out of range";
  const nScore = Number(i.min_setup_score);
  if (!Number.isInteger(nScore) || nScore < 0 || nScore > 10) return "min_setup_score must be 0-10";
  if (!["true", "false"].includes(i.use_qullamaggie)) return "use_qullamaggie must be true/false";
  if (!["true", "false"].includes(i.use_finviz_prefilter)) return "use_finviz_prefilter must be true/false";
  if (!["sec_edgar", "nasdaq_trader"].includes(i.universe_source)) return "invalid universe_source";
  return null;
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}
