/* Shared DeepShield UI/runtime helpers. */
DS.icons = function initIcons() {
  if (window.lucide && typeof lucide.createIcons === 'function') {
    lucide.createIcons({ attrs: { 'stroke-width': 1.75 } });
  }
};

DS.toast = function toast(message, type = 'info', opts = {}) {
  const { title = null, duration = 3800 } = opts;
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    container.setAttribute('aria-live', 'polite');
    document.body.appendChild(container);
  }
  const iconName = { success: 'check-circle', error: 'alert-triangle', warning: 'alert-triangle', info: 'info' }[type] || 'info';
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.setAttribute('role', 'status');
  el.innerHTML = `
    <i data-lucide="${iconName}" class="icon"></i>
    <div>${title ? `<div class="toast-title">${title}</div>` : ''}<div>${message}</div></div>
    <button class="toast-close" aria-label="Dismiss"><i data-lucide="x" class="icon-sm"></i></button>`;
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

DS.modal = {
  open(id) {
    const overlay = document.getElementById(id);
    if (!overlay) return;
    overlay.classList.add('open');
    const focusable = overlay.querySelector('button, [href], input, select');
    if (focusable) focusable.focus();
  },
  close(id) {
    const overlay = id ? document.getElementById(id) : document.querySelector('.modal-overlay.open');
    if (overlay) overlay.classList.remove('open');
  },
  _bind() {
    document.addEventListener('click', e => {
      const opener = e.target.closest('[data-modal-open]');
      if (opener) { DS.modal.open(opener.dataset.modalOpen); return; }
      if (e.target.closest('[data-modal-close]')) { DS.modal.close(); return; }
      if (e.target.classList && e.target.classList.contains('modal-overlay')) DS.modal.close();
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') DS.modal.close(); });
  },
};

DS.sidebar = {
  _bind() {
    document.addEventListener('click', e => {
      if (e.target.closest('[data-sidebar-toggle]')) document.body.classList.toggle('sidebar-open');
      else if (e.target.closest('.sidebar-overlay')) document.body.classList.remove('sidebar-open');
    });
  },
};

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
      queued = { card, x: e.clientX, y: e.clientY, raf: queued && queued.raf };
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

DS.server = {
  _promise: null,

  health() {
    if (!DS.server._promise) {
      DS.server._promise = fetch('/api/health', { cache: 'no-store' })
        .then(async r => {
          if (!r.ok) return null;
          try { return await r.json(); } catch { return null; }
        })
        .catch(() => null)
        .then(value => {
          // Cache successes for this page, but never cache an outage. A backend
          // that wakes/restarts after the page loads must be discoverable on
          // the very next user action.
          if (!value) DS.server._promise = null;
          return value;
        });
    }
    return DS.server._promise;
  },

  async hydrate() {
    const health = await DS.server.health();
    if (!health) {
      DS.util.qsa('[data-engine-badge]').forEach(el =>
        DS.server.paintBadge(el, DS.api.isExplicitDemo() ? 'simulated' : 'offline'));
      return;
    }
    Object.assign(DS.api.MODEL, health.model || {});
    if (Array.isArray(health.certainty_bands)) DS.api.CERTAINTY = health.certainty_bands;
    const fill = (attr, value) => DS.util.qsa(`[${attr}]`).forEach(el => { el.textContent = value; });
    fill('data-model-name', DS.api.MODEL.name);
    fill('data-model-params', DS.api.MODEL.params);
    fill('data-model-input', DS.api.MODEL.input);
    fill('data-model-backend', DS.api.MODEL.backend);
    fill('data-model-accuracy', health.test_accuracy != null ? `${health.test_accuracy}%` : '—');
    DS.util.qsa('[data-engine-badge]').forEach(el =>
      DS.server.paintBadge(el, health.engine === 'live' ? 'live' : 'simulated'));
    document.dispatchEvent(new CustomEvent('ds:server-ready', { detail: health }));
  },

  paintBadge(el, engine) {
    if (!el) return;
    if (engine === 'offline') {
      el.className = 'badge badge-danger engine-badge';
      el.title = 'The DeepShield backend cannot be reached. No analysis will be simulated.';
      el.innerHTML = '<i data-lucide="server-off" class="icon-sm"></i>Backend offline';
      DS.icons();
      return;
    }
    const live = engine === 'live';
    el.className = `badge ${live ? 'badge-success' : 'badge-warning'} engine-badge`;
    el.title = live
      ? 'A trained model analysed this media.'
      : 'No model analysed this media — this is an explicit demonstration mode.';
    el.innerHTML = live
      ? '<span class="badge-dot pulse"></span>Live model'
      : '<i data-lucide="flask-conical" class="icon-sm"></i>Simulated — demo only';
    DS.icons();
  },
};

DS.shell = {
  hydrate() {
    const user = DS.auth.user();
    const avatar = document.querySelector('[data-avatar]');
    if (avatar && user) avatar.textContent = DS.util.initials(user.name);
    DS.util.qsa('[data-user-name]').forEach(el => { if (user) el.textContent = user.name; });
    DS.util.qsa('[data-logout]').forEach(btn => btn.addEventListener('click', () => DS.auth.logout()));
    DS.util.qsa('[data-year]').forEach(el => { el.textContent = new Date().getFullYear(); });
  },
};

document.addEventListener('DOMContentLoaded', () => {
  DS.modal._bind();
  DS.sidebar._bind();
  DS.shell.hydrate();
  DS.server.hydrate();
  DS.glare._bind();
  DS.icons();
});
