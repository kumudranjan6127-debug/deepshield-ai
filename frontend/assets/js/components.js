/* ============================================================
   DeepShield AI — components.js
   Shared runtime: icons, toasts, modals, sidebar, navbar,
   footer year. Loaded on every page after utils.js + api.js.
   ============================================================ */

/* ---- Icons (Lucide) ---- */
DS.icons = function initIcons() {
  if (window.lucide && typeof lucide.createIcons === 'function') {
    lucide.createIcons({ attrs: { 'stroke-width': 1.75 } });
  }
};

/* ---- Toasts ---- */
DS.toast = function toast(message, type = 'info', opts = {}) {
  const { title = null, duration = 3800 } = opts;

  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    container.setAttribute('aria-live', 'polite');
    document.body.appendChild(container);
  }

  const iconName = {
    success: 'check-circle',
    error: 'alert-triangle',
    warning: 'alert-triangle',
    info: 'info',
  }[type] || 'info';

  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.setAttribute('role', 'status');
  el.innerHTML = `
    <i data-lucide="${iconName}" class="icon"></i>
    <div>
      ${title ? `<div class="toast-title">${title}</div>` : ''}
      <div>${message}</div>
    </div>
    <button class="toast-close" aria-label="Dismiss">
      <i data-lucide="x" class="icon-sm"></i>
    </button>
  `;
  container.appendChild(el);
  DS.icons();

  const dismiss = () => {
    el.classList.add('leaving');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  };
  el.querySelector('.toast-close').addEventListener('click', dismiss);
  if (duration > 0) setTimeout(dismiss, duration);
  return el;
};

/* ---- Modals ----
   Markup contract:
   <div class="modal-overlay" id="my-modal"> <div class="modal">…</div> </div>
   Openers:  [data-modal-open="my-modal"]
   Closers:  [data-modal-close] inside the modal, overlay click, Esc.     */
DS.modal = {
  open(id) {
    const overlay = document.getElementById(id);
    if (!overlay) return;
    overlay.classList.add('open');
    const focusable = overlay.querySelector('button, [href], input, select');
    if (focusable) focusable.focus();
  },
  close(id) {
    const overlay = id
      ? document.getElementById(id)
      : document.querySelector('.modal-overlay.open');
    if (overlay) overlay.classList.remove('open');
  },
  _bind() {
    document.addEventListener('click', e => {
      const opener = e.target.closest('[data-modal-open]');
      if (opener) { DS.modal.open(opener.dataset.modalOpen); return; }
      if (e.target.closest('[data-modal-close]')) { DS.modal.close(); return; }
      if (e.target.classList && e.target.classList.contains('modal-overlay')) DS.modal.close();
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') DS.modal.close();
    });
  },
};

/* ---- Sidebar (mobile off-canvas) ---- */
DS.sidebar = {
  _bind() {
    document.addEventListener('click', e => {
      if (e.target.closest('[data-sidebar-toggle]')) {
        document.body.classList.toggle('sidebar-open');
      } else if (e.target.closest('.sidebar-overlay')) {
        document.body.classList.remove('sidebar-open');
      }
    });
  },
};

/* ---- Cursor glare on cards ----
   One delegated pointermove, rAF-throttled, hover-devices only. Adds the
   .glare class lazily so touch devices and reduced-motion users pay
   nothing at all. */
DS.glare = {
  _bind() {
    const fine = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
    const reduced = document.documentElement.dataset.reducedMotion === 'true'
      || window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!fine || reduced) return;

    let queued = null;
    document.addEventListener('pointermove', e => {
      const card = e.target.closest('.card, .stat-card');
      if (!card) return;
      queued = { card, x: e.clientX, y: e.clientY };
      if (queued.raf) return;
      queued.raf = requestAnimationFrame(() => {
        const { card: c, x, y } = queued;
        const r = c.getBoundingClientRect();
        c.classList.add('glare');
        c.style.setProperty('--mx', `${((x - r.left) / r.width) * 100}%`);
        c.style.setProperty('--my', `${((y - r.top) / r.height) * 100}%`);
        queued.raf = null;
      });
    }, { passive: true });
  },
};

/* ---- Server info: one shared /api/health call per page ----
   Keeps model facts (name, size, accuracy) truthful across versions —
   nothing about the model is hardcoded in the pages any more. Pages
   that need it after load listen for the 'ds:server-ready' event. */
DS.server = {
  _promise: null,

  /* Cached — every caller shares the same request */
  health() {
    if (!DS.server._promise) {
      DS.server._promise = fetch('/api/health', { cache: 'no-store' })
        .then(r => (r.ok ? r.json() : null))
        .catch(() => null); // frontend-only mode (npm run dev / file://)
    }
    return DS.server._promise;
  },

  async hydrate() {
    const health = await DS.server.health();
    if (!health) return; // no backend — neutral placeholders stay

    // The server reports what it actually loaded; the browser never maps
    // an architecture to a name itself.
    Object.assign(DS.api.MODEL, health.model || {});

    const fill = (attr, value) =>
      DS.util.qsa(`[${attr}]`).forEach(el => { el.textContent = value; });

    fill('data-model-name', DS.api.MODEL.name);
    fill('data-model-params', DS.api.MODEL.params);
    fill('data-model-input', DS.api.MODEL.input);
    fill('data-model-backend', DS.api.MODEL.backend);
    fill('data-model-accuracy',
         health.test_accuracy != null ? `${health.test_accuracy}%` : '—');

    document.dispatchEvent(new CustomEvent('ds:server-ready', { detail: health }));
  },
};

/* ---- App-shell hydration (navbar user, logout, footer year) ---- */
DS.shell = {
  hydrate() {
    const user = DS.auth.user();

    // Avatar initials
    const avatar = document.querySelector('[data-avatar]');
    if (avatar && user) avatar.textContent = DS.util.initials(user.name);

    // Any element that wants the user's name
    DS.util.qsa('[data-user-name]').forEach(el => {
      if (user) el.textContent = user.name;
    });

    // Logout buttons
    DS.util.qsa('[data-logout]').forEach(btn =>
      btn.addEventListener('click', () => DS.auth.logout()));

    // Footer year
    DS.util.qsa('[data-year]').forEach(el =>
      el.textContent = new Date().getFullYear());
  },
};

/* ---- Boot ---- */
document.addEventListener('DOMContentLoaded', () => {
  DS.modal._bind();
  DS.sidebar._bind();
  DS.shell.hydrate();
  DS.server.hydrate();
  DS.glare._bind();
  DS.icons();
});
