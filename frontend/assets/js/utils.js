/* ============================================================
   DeepShield AI — utils.js
   Global namespace bootstrap + formatting, storage, history,
   settings helpers. Loaded first on every page.
   ============================================================ */

window.DS = window.DS || {};

/* ---- Storage keys (single source of truth) ---- */
DS.KEYS = {
  USER: 'ds_user',         // localStorage  {name, email, loggedInAt}
  SCAN: 'ds_scan',         // sessionStorage — the scan currently in flight / just finished
  HISTORY: 'ds_history',   // localStorage  — array of completed scans (newest first)
  SETTINGS: 'ds_settings', // localStorage  {frameRate, threshold, reducedMotion, autoDelete}
};

/* ---- Safe storage wrappers ---- */
function makeStore(storage) {
  return {
    get(key, fallback = null) {
      try {
        const raw = storage.getItem(key);
        return raw === null ? fallback : JSON.parse(raw);
      } catch { return fallback; }
    },
    set(key, value) {
      try { storage.setItem(key, JSON.stringify(value)); } catch { /* quota/file:// edge */ }
    },
    remove(key) {
      try { storage.removeItem(key); } catch { /* noop */ }
    },
  };
}
DS.store   = makeStore(window.localStorage);
DS.session = makeStore(window.sessionStorage);

/* ---- Formatting ---- */
DS.util = {
  qs:  (sel, root = document) => root.querySelector(sel),
  qsa: (sel, root = document) => [...root.querySelectorAll(sel)],

  formatBytes(bytes) {
    if (!bytes && bytes !== 0) return '—';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0, n = bytes;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
  },

  formatDuration(ms) {
    if (!ms && ms !== 0) return '—';
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(1)} s`;
  },

  formatDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
      + ' · '
      + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  },

  /* SCAN-8F3A2C style id */
  uid(prefix = 'SCAN') {
    const hex = Array.from(crypto.getRandomValues(new Uint8Array(3)))
      .map(b => b.toString(16).padStart(2, '0')).join('').toUpperCase();
    return `${prefix}-${hex}`;
  },

  /* Deterministic 32-bit hash — used to seed the mock verdict */
  hash(str) {
    let h = 2166136261;
    for (let i = 0; i < String(str).length; i++) {
      h ^= String(str).charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  },

  /* "Kumud Ranjan" → "KR", and anything empty → "U".
     Trim first: a name of only spaces is truthy, so the old version reached
     `''[0].toUpperCase()` and threw. That was unreachable while the name was
     derived from an email address; it is reachable now that people type it. */
  initials(name) {
    const words = String(name || '').trim().split(/\s+/).filter(Boolean);
    if (!words.length) return 'U';
    return words.slice(0, 2).map(w => w[0].toUpperCase()).join('');
  },

  greeting() {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    return 'Good evening';
  },

  truncate(s, n) {
    s = String(s);
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
  },

  escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
  },
};

/* ---- Scan history ---- */
DS.history = {
  all() { return DS.store.get(DS.KEYS.HISTORY, []); },
  add(scan) {
    const list = DS.history.all();
    list.unshift(scan);
    DS.store.set(DS.KEYS.HISTORY, list.slice(0, 50)); // cap
  },
  find(id) { return DS.history.all().find(s => s.id === id) || null; },
  clear() { DS.store.remove(DS.KEYS.HISTORY); },
};

/* ---- Settings ---- */
DS.SETTINGS_DEFAULTS = {
  frameRate: 1,        // frames sampled per second (video)
  threshold: 60,       // confidence threshold (%)
  reducedMotion: false,
  autoDelete: true,    // remove uploads after analysis
  effects: 'auto',     // 'auto' | 'full' (glass blur) | 'lite'
};
DS.settings = {
  get() { return { ...DS.SETTINGS_DEFAULTS, ...DS.store.get(DS.KEYS.SETTINGS, {}) }; },
  set(patch) {
    const next = { ...DS.settings.get(), ...patch };
    DS.store.set(DS.KEYS.SETTINGS, next);
    DS.settings.apply(next);
    return next;
  },
  apply(s = DS.settings.get()) {
    const root = document.documentElement;
    root.dataset.reducedMotion = s.reducedMotion ? 'true' : 'false';

    // 'auto' is resolved pre-paint in boot.js — only override on an
    // explicit choice, so we never fight that decision on load.
    if (s.effects === 'full' || s.effects === 'lite') root.dataset.fx = s.effects;
  },
};

/* ---- Auth helpers ---- */
DS.auth = {
  user() { return DS.store.get(DS.KEYS.USER, null); },
  login(name, email) {
    const user = { name, email, loggedInAt: new Date().toISOString() };
    DS.store.set(DS.KEYS.USER, user);
    return user;
  },
  logout() {
    DS.store.remove(DS.KEYS.USER);
    // ?signout=1 tells login.js to also end the Firebase session (if any)
    window.location.href = 'login.html?signout=1';
  },
  /* Call on protected pages. Returns user or redirects. */
  guard() {
    const user = DS.auth.user();
    if (!user) { window.location.replace('login.html'); return null; }
    return user;
  },
};

/* Apply persisted settings ASAP (reduced-motion flag on <html>) */
DS.settings.apply();
