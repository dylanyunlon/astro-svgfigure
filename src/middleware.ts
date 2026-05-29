/**
 * src/middleware.ts — IP Firewall v3 ("French Beach + Underwear Thief")
 *
 * Five-layer defense:
 *
 *   Layer 0   — Whitelist
 *     Your own servers, private ranges. Always pass, zero cost.
 *
 *   Layer 0.5 — HTTP Method Filter
 *     Block WebDAV (PROPFIND) and exotic methods instantly. Zero cost.
 *
 *   Layer 1   — Behavior traps (the "underwear thief" detector)
 *     Scanners probe /.env, /wp-admin, /actuator, /swagger…
 *     First suspicious path = instant ban + cached for 24h.
 *
 *   Layer 1.5 — Burst Detection (sliding window rate limiter)
 *     Same IP > 30 requests in 10s → auto-ban 1h.
 *
 *   Layer 2   — IP reputation (AbuseIPDB / Scamalytics / proxycheck)
 *     Catches known botnets / Tor exits before they try a trap path.
 *
 * Cache: in-memory LRU, 4096 entries, 24h for trap bans, 1h for reputation.
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

/** HTTP methods that legitimate browsers actually use. */
const ALLOWED_METHODS: ReadonlySet<string> = new Set([
  'GET', 'POST', 'HEAD', 'OPTIONS',
]);

/** Suspicious methods (WebDAV, exotic verbs) → instant ban. */
const BLOCKED_METHODS: ReadonlySet<string> = new Set([
  'PROPFIND', 'PROPPATCH', 'MKCOL', 'COPY', 'MOVE', 'LOCK', 'UNLOCK',
  'PATCH', 'DELETE', 'TRACE', 'CONNECT', 'SEARCH', 'PURGE',
]);

/* ──────────────────────────────────────────────────────────────────────
 * Layer 0 — Whitelist
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
 * ────────────────────────────────────────────────────────────────────── */

const TRAP_EXACT: ReadonlySet<string> = new Set([
  '/server', '/server-status', '/server-info',
  '/login.action', '/debug/default/view', '/trace.axd',
  '/info.php', '/phpinfo.php', '/config.json', '/telescope/requests',
  '/.ds_store', '/.vscode/sftp.json',
  '/swagger.json', '/swagger-ui',
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
  // Microsoft Exchange / Outlook (observed: /owa/auth/logon.aspx)
  '/ecp/', '/owa/', '/autodiscover/', '/remote/',
  // Atlassian
  '/s/8323',
  // Spring Boot
  '/actuator/',
  // Source / config leak
  '/.env', '/.git/', '/.svn/', '/.hg/',
  '/proc/', '/etc/passwd',
  // Subfolder .env scanning (observed: /public/.env, /shared/.env, etc.)
  '/public/.env', '/shared/.env', '/app/.env', '/web/.env', '/www/.env',
  '/src/.env', '/config/.env', '/backend/.env', '/api/.env',
  // Swagger family
  '/swagger/', '/swagger/v1/', '/api-docs/',
  '/v2/api-docs', '/v3/api-docs', '/webjars/swagger',
  // GraphQL
  '/graphql',
  // Admin panels
  '/phpmyadmin', '/adminer', '/pma/',
  // Node.js internals
  '/node_modules/', '/.npmrc',
  // Login / auth probes
  '/login', '/signin', '/admin', '/administrator',
  '/auth/', '/oauth/', '/sso/',
  // Common CMS
  '/cms/', '/umbraco/', '/sitecore/', '/typo3/',
  // Shell / webshell
  '/shell', '/cmd', '/exec', '/c99',
  // Backup files
  '/backup', '/db.sql', '/dump.sql', '/database.sql',
  // PHP dependency scanners (phpunit RCE CVE-2017-9841 etc.)
  '/vendor/', '/lib/phpunit/', '/phpunit/',
  // ThinkPHP RCE probes
  '/index.php',
  // Docker / container info leak
  '/containers/',
  // Common PHP entry-point probes (hello.world, root POST)
  '/hello.',
];

const TRAP_SUBSTRINGS: readonly string[] = [
  'META-INF/', 'WEB-INF/',
  '..%2f', '..%5c', '%00',
  'pom.properties',
  '.php', 'cgi-bin/',
  '.asp', '.aspx', '.jsp',
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
 * LRU Cache
 * ────────────────────────────────────────────────────────────────────── */

interface CacheEntry {
  score: number;
  provider: string;
  ts: number;
  blocked: boolean;
  trapPath?: string;
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
 * Layer 1.5 — Burst Detection (sliding window rate limiter)
 * ────────────────────────────────────────────────────────────────────── */

const BURST_WINDOW_MS = 10_000;
const BURST_LIMIT = 30;
const BURST_BAN_MS = 3_600_000;

const burstMap = new Map<string, number[]>();

let burstCleanupTimer: ReturnType<typeof setInterval> | null = null;
function ensureBurstCleanup(): void {
  if (burstCleanupTimer) return;
  burstCleanupTimer = setInterval(() => {
    const cutoff = Date.now() - BURST_WINDOW_MS * 2;
    for (const [ip, timestamps] of burstMap) {
      if (timestamps.length === 0 || timestamps[timestamps.length - 1] < cutoff) {
        burstMap.delete(ip);
      }
    }
  }, 60_000);
}

function recordAndCheckBurst(ip: string): boolean {
  ensureBurstCleanup();
  const now = Date.now();
  const cutoff = now - BURST_WINDOW_MS;
  let timestamps = burstMap.get(ip);
  if (!timestamps) {
    timestamps = [];
    burstMap.set(ip, timestamps);
  }
  while (timestamps.length > 0 && timestamps[0] < cutoff) {
    timestamps.shift();
  }
  timestamps.push(now);
  return timestamps.length > BURST_LIMIT;
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
 * Middleware — five layers wired together
 * ────────────────────────────────────────────────────────────────────── */

export const onRequest = defineMiddleware(async ({ request }, next) => {
  if (!ENABLED) return next();

  const url = new URL(request.url);
  const pathname = url.pathname;

  // Skip framework-internal static assets
  if (BYPASS_PREFIXES.some((p) => pathname.startsWith(p))) {
    return next();
  }

  const method = request.method.toUpperCase();

  // ── Layer 0.5: HTTP Method Filter ──────────────────────────────────
  if (BLOCKED_METHODS.has(method)) {
    const ip = extractIp(request);
    cacheSet(ip, {
      score: 100, provider: 'method', ts: Date.now(),
      blocked: true, trapPath: `${method} ${pathname}`, ttlMs: TRAP_BAN_MS,
    });
    logBlock(ip, 'method', `method=${method} path=${pathname}`);
    return new Response(null, { status: 403 });
  }

  if (!ALLOWED_METHODS.has(method)) {
    return new Response(null, { status: 405 });
  }

  const ip = extractIp(request);

  // ── Layer 0: Whitelist ─────────────────────────────────────────────
  if (isTrustedIp(ip)) return next();

  // ── Check cache (covers trap bans, reputation bans, burst bans) ───
  const cached = cacheGet(ip);
  if (cached) {
    if (cached.blocked) {
      return new Response(null, { status: 403 });
    }
    // Known clean — still check burst
    if (recordAndCheckBurst(ip)) {
      cacheSet(ip, {
        score: 100, provider: 'burst', ts: Date.now(),
        blocked: true, ttlMs: BURST_BAN_MS,
      });
      logBlock(ip, 'burst', `limit=${BURST_LIMIT}/${BURST_WINDOW_MS}ms`);
      return new Response(null, { status: 429 });
    }
    return next();
  }

  // ── Layer 1: Behavior trap ─────────────────────────────────────────
  if (isTrapPath(pathname)) {
    const entry: CacheEntry = {
      score: 100, provider: 'trap', ts: Date.now(),
      blocked: true, trapPath: pathname, ttlMs: TRAP_BAN_MS,
    };
    cacheSet(ip, entry);
    logBlock(ip, 'trap', `path=${pathname}`);
    return new Response(null, { status: 403 });
  }

  // ── Layer 1.5: Burst detection ─────────────────────────────────────
  if (recordAndCheckBurst(ip)) {
    cacheSet(ip, {
      score: 100, provider: 'burst', ts: Date.now(),
      blocked: true, ttlMs: BURST_BAN_MS,
    });
    logBlock(ip, 'burst', `limit=${BURST_LIMIT}/${BURST_WINDOW_MS}ms`);
    return new Response(null, { status: 429 });
  }

  // ── Layer 2: IP reputation (only for first-time clean-path visitors)
  const result = await getIpScore(ip);

  if (result === null) {
    cacheSet(ip, {
      score: 0, provider: 'fail-open', ts: Date.now(),
      blocked: false, ttlMs: 300_000,
    });
    return next();
  }

  const blocked = result.score >= BLOCK_THRESHOLD;
  const entry: CacheEntry = {
    score: result.score, provider: result.provider,
    ts: Date.now(), blocked, ttlMs: CACHE_TTL_MS,
  };
  cacheSet(ip, entry);

  if (blocked) {
    logBlock(ip, 'reputation', `score=${result.score} provider=${result.provider}`);
    return new Response(null, { status: 403 });
  }

  return next();
});