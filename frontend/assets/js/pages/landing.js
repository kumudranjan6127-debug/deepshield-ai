/* ============================================================
   DeepShield AI — landing.js
   Public page: mobile menu, sticky-nav shadow, scroll reveals.
   No auth guard — this page is the front door.
   ============================================================ */

/* Mark JS as available so CSS can hide reveal elements. Without JS the
   content stays visible — progressive enhancement, never a blank page. */
document.documentElement.classList.add('js');

document.addEventListener('DOMContentLoaded', () => {
  /* ---- Mobile menu ---- */
  const burger = document.getElementById('lp-burger');
  const links = document.getElementById('lp-links');
  burger.addEventListener('click', () => links.classList.toggle('open'));
  links.addEventListener('click', e => {
    if (e.target.tagName === 'A') links.classList.remove('open');
  });

  /* ---- Sticky nav gets a border once the page scrolls ---- */
  const nav = document.getElementById('lp-nav');
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  /* ---- Reveal on scroll ---- */
  const items = DS.util.qsa('.reveal');
  const reduced =
    document.documentElement.dataset.reducedMotion === 'true' ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (reduced || !('IntersectionObserver' in window)) {
    items.forEach(el => el.classList.add('in'));
    return;
  }

  const io = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('in');
      obs.unobserve(entry.target); // reveal once, then stop watching
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });

  items.forEach(el => io.observe(el));
});
