// platform.ts — Platform detection, screen lock, cookie/privacy, Metal, font loading

// ---------------------------------------------------------------------------
// Platform
// ---------------------------------------------------------------------------

export interface PlatformInfo {
  isMobile: boolean;
  isIOS: boolean;
  hasWebGL2: boolean;
  hasWebGPU: boolean;
}

export const Platform = (() => {
  let _info: PlatformInfo | null = null;

  function detect(): PlatformInfo {
    if (_info) return _info;

    const ua = typeof navigator !== "undefined" ? navigator.userAgent : "";
    const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
    const isIOS = /iPhone|iPad|iPod/i.test(ua);

    let hasWebGL2 = false;
    let hasWebGPU = false;

    if (typeof document !== "undefined") {
      try {
        const canvas = document.createElement("canvas");
        hasWebGL2 = !!(canvas.getContext("webgl2"));
      } catch (_) {
        hasWebGL2 = false;
      }
    }

    if (typeof navigator !== "undefined") {
      hasWebGPU = "gpu" in navigator;
    }

    _info = { isMobile, isIOS, hasWebGL2, hasWebGPU };
    return _info;
  }

  return {
    get isMobile() { return detect().isMobile; },
    get isIOS() { return detect().isIOS; },
    get hasWebGL2() { return detect().hasWebGL2; },
    get hasWebGPU() { return detect().hasWebGPU; },
    detect,
  };
})();

// ---------------------------------------------------------------------------
// ScreenLock
// ---------------------------------------------------------------------------

export type OrientationLockType =
  | "any"
  | "natural"
  | "landscape"
  | "portrait"
  | "portrait-primary"
  | "portrait-secondary"
  | "landscape-primary"
  | "landscape-secondary";

export const ScreenLock = (() => {
  let _locked = false;

  async function lock(orientation: OrientationLockType = "landscape"): Promise<boolean> {
    if (typeof screen === "undefined" || !screen.orientation) return false;
    try {
      await (screen.orientation as any).lock(orientation);
      _locked = true;
      return true;
    } catch (err) {
      console.warn("[ScreenLock] lock failed:", err);
      return false;
    }
  }

  function unlock(): void {
    if (typeof screen === "undefined" || !screen.orientation) return;
    try {
      (screen.orientation as any).unlock();
      _locked = false;
    } catch (err) {
      console.warn("[ScreenLock] unlock failed:", err);
    }
  }

  return {
    lock,
    unlock,
    get isLocked() { return _locked; },
  };
})();

// ---------------------------------------------------------------------------
// CookieNotice
// ---------------------------------------------------------------------------

const COOKIE_ACCEPTED_KEY = "cookie_notice_accepted";

export const CookieNotice = (() => {
  let _visible = false;
  let _listeners: Array<(accepted: boolean) => void> = [];

  function show(): void {
    _visible = true;
    _dispatch(false);
  }

  function hide(): void {
    _visible = false;
  }

  function accept(): void {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(COOKIE_ACCEPTED_KEY, "1");
    }
    _visible = false;
    _dispatch(true);
  }

  function accepted(): boolean {
    if (typeof localStorage === "undefined") return false;
    return localStorage.getItem(COOKIE_ACCEPTED_KEY) === "1";
  }

  function onAccepted(cb: (accepted: boolean) => void): () => void {
    _listeners.push(cb);
    return () => { _listeners = _listeners.filter(l => l !== cb); };
  }

  function _dispatch(value: boolean): void {
    for (const l of _listeners) l(value);
  }

  return {
    show,
    hide,
    accept,
    accepted,
    onAccepted,
    get isVisible() { return _visible; },
  };
})();

// ---------------------------------------------------------------------------
// Privacy
// ---------------------------------------------------------------------------

export const Privacy = (() => {
  const CONSENT_KEY = "privacy_consent";

  function checkConsent(): boolean {
    if (typeof localStorage === "undefined") return false;
    return localStorage.getItem(CONSENT_KEY) === "granted";
  }

  function grantConsent(): void {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(CONSENT_KEY, "granted");
    }
  }

  function revokeConsent(): void {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(CONSENT_KEY, "denied");
    }
  }

  function clearData(keys?: string[]): void {
    if (typeof localStorage === "undefined") return;
    if (keys && keys.length > 0) {
      for (const k of keys) localStorage.removeItem(k);
    } else {
      localStorage.clear();
    }
    if (typeof sessionStorage !== "undefined") {
      sessionStorage.clear();
    }
  }

  return {
    checkConsent,
    grantConsent,
    revokeConsent,
    clearData,
  };
})();

// ---------------------------------------------------------------------------
// Metal  (Apple GPU – browser heuristic)
// ---------------------------------------------------------------------------

export const Metal = (() => {
  let _checked = false;
  let _supported = false;

  async function isMetalSupported(): Promise<boolean> {
    if (_checked) return _supported;
    _checked = true;

    // WebGPU on Safari/macOS/iOS implies Metal back-end
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      try {
        const adapter = await (navigator as any).gpu.requestAdapter();
        _supported = adapter !== null;
        return _supported;
      } catch (_) {
        /* fall through */
      }
    }
    _supported = false;
    return _supported;
  }

  return { isMetalSupported };
})();

// ---------------------------------------------------------------------------
// NBArchitektStd  (custom font loader via FontFace API)
// ---------------------------------------------------------------------------

export interface FontLoadOptions {
  weight?: string | number;
  style?: string;
  display?: FontDisplay;
}

export const NBArchitektStd = (() => {
  const _loaded = new Map<string, FontFace>();

  async function loadFont(
    url: string,
    options: FontLoadOptions = {}
  ): Promise<FontFace> {
    const cacheKey = `${url}|${options.weight ?? "normal"}|${options.style ?? "normal"}`;
    if (_loaded.has(cacheKey)) {
      return _loaded.get(cacheKey)!;
    }

    if (typeof FontFace === "undefined") {
      throw new Error("[NBArchitektStd] FontFace API not available in this environment.");
    }

    const descriptors: FontFaceDescriptors = {
      weight: String(options.weight ?? "normal"),
      style: options.style ?? "normal",
      display: options.display ?? "swap",
    };

    const font = new FontFace("NBArchitektStd", `url(${url})`, descriptors);
    await font.load();

    if (typeof document !== "undefined" && document.fonts) {
      document.fonts.add(font);
    }

    _loaded.set(cacheKey, font);
    return font;
  }

  function isLoaded(): boolean {
    return _loaded.size > 0;
  }

  function getAll(): FontFace[] {
    return Array.from(_loaded.values());
  }

  return { loadFont, isLoaded, getAll };
})();
