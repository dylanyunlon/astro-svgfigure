/**
 * src/middleware.ts — IP Firewall v2 ("French Beach + Underwear Thief")
 *
 * Three-layer defense:
 *
 *   Layer 0 — Whitelist
 *     Your own servers, private ranges. Always pass, zero cost.
 *
 *   Layer 1 — Behavior traps (the "underwear thief" detector)
 *     Normal users visit /generate, /blog, /docs, /gallery.
 *     Scanners probe /.env, /wp-admin, /actuator, /swagger…
 *     First suspicious path = instant ban + cached for 24h.
 *     Zero API calls, zero external dependencies, catches ANY scanner
 *     regardless of how clean their IP reputation is.
 *
 *   Layer 2 — IP reputation (AbuseIPDB / Scamalytics / proxycheck)
 *     Only queried for IPs that haven't hit a trap and aren't cached.
 *     Catches known botnets / Tor exits before they even try a trap path.
 *
 * Cache: in-memory LRU, 4096 entries, 24h for trap bans, 1h for reputation.
 *
 * File: src/middleware.ts  (Astro auto-discovers)
 */

import { defineMiddleware } from 'astro:middleware';

/* ──────────────────────────────────────────────────────────────────────
 * Configuration
 * ────────────────────────────────────────────────────────────────────── */

const BLOCK_THRESHOLD = Number(import.meta.env.IP_BLOCK_THRESHOLD ?? '75');
const CACHE_TTL_MS = Number(import.meta.env.IP_CACHE_TTL_MS ?? '3600000');
const CACHE_MAX = Number(import.meta.env.IP_CACHE_MAX ?? '4096');

/** Trap ban duration — 24 hours. Scanners don't deserve a quick retry. */
const TRAP_BAN_MS = 86_400_000;

const ABUSEIPDB_KEY = import.meta.env.ABUSEIPDB_API_KEY ?? '';
const SCAMALYTICS_KEY = import.meta.env.SCAMALYTICS_API_KEY ?? '';
const SCAMALYTICS_USER = import.meta.env.SCAMALYTICS_USERNAME ?? '';
const PROXYCHECK_KEY = import.meta.env.PROXYCHECK_API_KEY ?? '';

const ENABLED = (import.meta.env.IP_FIREWALL_ENABLED ?? 'true') === 'true';

/** Paths that bypass ALL checks (static assets the framework itself serves) */
const BYPASS_PREFIXES = ['/_image', '/_astro/', '/favicon', '/fonts/', '/icons/'];

/* ──────────────────────────────────────────────────────────────────────
 * Layer 0 — Whitelist
 *
 * Your own IPs + RFC1918 private ranges. Set IP_WHITELIST in .env as
 * a comma-separated list: IP_WHITELIST=167.160.187.143,1.2.3.4
 * ────────────────────────────────────────────────────────────────────── */

const WHITELIST: ReadonlySet<string> = new Set(
  (import.meta.env.IP_WHITELIST ?? '')
    .split(',')
    .map((s: string) => s.trim())
    .filter(Boolean)
);

const PRIVATE_PREFIXES = [
  '127.', '10.', '192.168.',
  '172.16.', '172.17.', '172.18.', '172.19.',
  '172.20.', '172.21.', '172.22.', '172.23.',
  '172.24.', '172.25.', '172.26.', '172.27.',
  '172.28.', '172.29.', '172.30.', '172.31.',
  '::1', 'fe80:', '0.0.0.0',
];

function isTrustedIp(ip: string): boolean {
  if (WHITELIST.has(ip)) return true;
  return PRIVATE_PREFIXES.some((p) => ip.startsWith(p));
}

/* ──────────────────────────────────────────────────────────────────────
 * Layer 1 — Behavior traps
 *
 * These are paths that NO legitimate visitor would ever request.
 * Hitting any one of them = immediate 24h ban. Like catching someone
 * rummaging through beach bags — you don't need to check their ID,
 * you just saw them do it.
 *
 * Two categories:
 *   TRAP_EXACT  — exact path match (lowercase)
 *   TRAP_PREFIX — starts-with match
 *
 * The lists are derived directly from the 09:45:xx scan burst in
 * production logs. Each entry maps to a real attack vector observed.
 * ────────────────────────────────────────────────────────────────────── */

const TRAP_EXACT: ReadonlySet<string> = new Set([
  // Server info disclosure
  '/server',
  '/server-status',
  '/server-info',
  // Application exploits
  '/login.action',              // Struts / Confluence RCE
  '/debug/default/view',        // Yii debug
  '/trace.axd',                 // ASP.NET trace
  '/info.php',                  // phpinfo()
  '/phpinfo.php',
  '/config.json',               // App config leak
  '/telescope/requests',        // Laravel Telescope
  // Metadata files
  '/.ds_store',
  '/.vscode/sftp.json',         // VSCode SFTP creds
  // API docs (we don't run swagger)
  '/swagger.json',
  '/swagger-ui',
]);

const TRAP_PREFIXES: readonly string[] = [
  // WordPress
  '/wp-json/', '/wp-admin', '/wp-content', '/wp-includes', '/wp-login',
  '/xmlrpc.php',
  // Hosting panels
  '/___proxy_subdomain_whm', '/___proxy_subdomain_cpanel',
  '/cpanel', '/whm',
  // Docker registry
  '/v2/_catalog',
  // Microsoft Exchange
  '/ecp/', '/owa/', '/autodiscover/',
  // Atlassian
  '/s/8323',                    // Jira metadata probe
  // Spring Boot
  '/actuator/',
  // Source / config leak
  '/.env', '/.git/', '/.svn/', '/.hg/',
  '/proc/',                     // /proc/self/environ
  '/etc/passwd',
  // Swagger family
  '/swagger/', '/swagger/v1/', '/api-docs/',
  '/v2/api-docs', '/v3/api-docs', '/webjars/swagger',
  // GraphQL scanning (we don't have graphql)
  '/graphql',
  // Admin panels
  '/phpmyadmin', '/adminer', '/pma/',
  // Node.js internals
  '/node_modules/', '/.npmrc',
];

/**
 * Substrings that indicate traversal or injection in any path position.
 */
const TRAP_SUBSTRINGS: readonly string[] = [
  'META-INF/',
  'WEB-INF/',
  '..%2f',                      // Encoded path traversal
  '..%5c',
  '%00',                        // Null byte injection
  'pom.properties',             // Maven metadata
];

function isTrapPath(pathname: string): boolean {
  const lower = pathname.toLowerCase();

  if (TRAP_EXACT.has(lower)) return true;

  for (const p of TRAP_PREFIXES) {
    if (lower.startsWith(p)) return true;
  }

  for (const s of TRAP_SUBSTRINGS) {
    if (lower.includes(s)) return true;
  }

  return false;
}

/* ──────────────────────────────────────────────────────────────────────
 * LRU Cache (shared across Layer 1 and Layer 2)
 * ────────────────────────────────────────────────────────────────────── */

interface CacheEntry {
  score: number;
  provider: string;       // 'trap', 'abuseipdb', 'proxycheck', etc.
  ts: number;
  blocked: boolean;
  /** Which trap path triggered the ban (for logging) */
  trapPath?: string;
  /** Custom TTL override — trap bans last 24h, reputation 1h */
  ttlMs: number;
}

const cache = new Map<string, CacheEntry>();

function cacheGet(ip: string): CacheEntry | null {
  const entry = cache.get(ip);
  if (!entry) return null;
  if (Date.now() - entry.ts > entry.ttlMs) {
    cache.delete(ip);
    return null;
  }
  cache.delete(ip);
  cache.set(ip, entry);
  return entry;
}

function cacheSet(ip: string, entry: CacheEntry): void {
  if (cache.size >= CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  cache.set(ip, entry);
}

/* ──────────────────────────────────────────────────────────────────────
 * IP extraction
 * ────────────────────────────────────────────────────────────────────── */

function extractIp(request: Request): string {
  const cf = request.headers.get('cf-connecting-ip');
  if (cf) return cf.trim();
  const xff = request.headers.get('x-forwarded-for');
  if (xff) return xff.split(',')[0].trim();
  const real = request.headers.get('x-real-ip');
  if (real) return real.trim();
  return '0.0.0.0';
}

/* ──────────────────────────────────────────────────────────────────────
 * Layer 2 — IP reputation providers
 * ────────────────────────────────────────────────────────────────────── */

async function queryAbuseIPDB(ip: string): Promise<number | null> {
  if (!ABUSEIPDB_KEY) return null;
  try {
    const url = `https://api.abuseipdb.com/api/v2/check?ipAddress=${encodeURIComponent(ip)}&maxAgeInDays=90`;
    const res = await fetch(url, {
      headers: { Key: ABUSEIPDB_KEY, Accept: 'application/json' },
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;
    const data = await res.json() as { data?: { abuseConfidenceScore?: number } };
    return typeof data?.data?.abuseConfidenceScore === 'number'
      ? data.data.abuseConfidenceScore
      : null;
  } catch { return null; }
}

async function queryScamalytics(ip: string): Promise<number | null> {
  if (!SCAMALYTICS_KEY || !SCAMALYTICS_USER) return null;
  try {
    const url = `https://api11.scamalytics.com/v3/${SCAMALYTICS_USER}?key=${SCAMALYTICS_KEY}&ip=${ip}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return null;
    const data = await res.json() as Record<string, unknown>;
    const score = Number(data.score ?? data.risk ?? -1);
    if (score >= 0 && score <= 100) return score;
    const level = String(data.risk ?? '').toLowerCase();
    return { 'very high': 90, 'high': 75, 'medium': 50, 'low': 15 }[level] ?? null;
  } catch { return null; }
}

async function queryProxyCheck(ip: string): Promise<number | null> {
  if (!PROXYCHECK_KEY) return null;
  try {
    const url = `https://proxycheck.io/v2/${ip}?key=${PROXYCHECK_KEY}&risk=1&vpn=1`;
    const res = await fetch(url, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return null;
    const data = await res.json() as Record<string, unknown>;
    const entry = data[ip] as Record<string, unknown> | undefined;
    if (!entry) return null;
    const risk = Number(entry.risk ?? -1);
    return risk >= 0 && risk <= 100 ? risk : null;
  } catch { return null; }
}

/** Query all providers in parallel, return highest score. */
async function getIpScore(ip: string): Promise<{ score: number; provider: string } | null> {
  const queries = await Promise.allSettled([
    queryAbuseIPDB(ip).then((s) => s !== null ? { score: s, provider: 'abuseipdb' } : null),
    queryScamalytics(ip).then((s) => s !== null ? { score: s, provider: 'scamalytics' } : null),
    queryProxyCheck(ip).then((s) => s !== null ? { score: s, provider: 'proxycheck' } : null),
  ]);

  let best: { score: number; provider: string } | null = null;
  for (const q of queries) {
    if (q.status === 'fulfilled' && q.value !== null) {
      if (best === null || q.value.score > best.score) {
        best = q.value;
      }
    }
  }
  return best;
}

/* ──────────────────────────────────────────────────────────────────────
 * Logging
 * ────────────────────────────────────────────────────────────────────── */

function logBlock(ip: string, reason: string, extra?: string): void {
  const ts = new Date().toISOString();
  const suffix = extra ? ` ${extra}` : '';
  console.warn(`[FIREWALL] ${ts} BLOCKED ip=${ip} reason=${reason}${suffix}`);
}

/* ──────────────────────────────────────────────────────────────────────
 * Middleware — the three layers wired together
 * ────────────────────────────────────────────────────────────────────── */

export const onRequest = defineMiddleware(async ({ request }, next) => {
  if (!ENABLED) return next();

  const url = new URL(request.url);
  const pathname = url.pathname;

  // Skip framework-internal static assets
  if (BYPASS_PREFIXES.some((p) => pathname.startsWith(p))) {
    return next();
  }

  const ip = extractIp(request);

  // ── Layer 0: Whitelist ─────────────────────────────────────────────
  if (isTrustedIp(ip)) return next();

  // ── Check cache (covers both trap bans and reputation bans) ────────
  const cached = cacheGet(ip);
  if (cached) {
    if (cached.blocked) {
      // Don't re-log every request from a banned IP — too noisy
      return new Response(null, { status: 403 });
    }
    // Known clean — skip everything
    return next();
  }

  // ── Layer 1: Behavior trap ─────────────────────────────────────────
  // Caught stealing underwear → instant 24h ban, zero API cost
  if (isTrapPath(pathname)) {
    const entry: CacheEntry = {
      score: 100,
      provider: 'trap',
      ts: Date.now(),
      blocked: true,
      trapPath: pathname,
      ttlMs: TRAP_BAN_MS,
    };
    cacheSet(ip, entry);
    logBlock(ip, 'trap', `path=${pathname}`);
    return new Response(null, { status: 403 });
  }

  // ── Layer 2: IP reputation (only for first-time clean-path visitors)
  const result = await getIpScore(ip);

  if (result === null) {
    // All providers failed — fail open, short cache
    cacheSet(ip, {
      score: 0,
      provider: 'fail-open',
      ts: Date.now(),
      blocked: false,
      ttlMs: 300_000,           // retry in 5 min
    });
    return next();
  }

  const blocked = result.score >= BLOCK_THRESHOLD;
  const entry: CacheEntry = {
    score: result.score,
    provider: result.provider,
    ts: Date.now(),
    blocked,
    ttlMs: CACHE_TTL_MS,
  };
  cacheSet(ip, entry);

  if (blocked) {
    logBlock(ip, 'reputation', `score=${result.score} provider=${result.provider}`);
    return new Response(null, { status: 403 });
  }

  return next();
});