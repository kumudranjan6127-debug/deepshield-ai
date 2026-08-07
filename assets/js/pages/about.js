/* ============================================================
   DeepShield AI — about.js
   Static page: auth guard only. Shell hydration, icons and
   footer year are handled by components.js.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const user = DS.auth.guard();
  if (!user) return;
});
