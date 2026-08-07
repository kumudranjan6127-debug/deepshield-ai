/* ============================================================
   DeepShield AI — boot.js  (loaded BLOCKING in <head>, pre-paint)
   First page of the session gets the fade-up entry animation;
   every navigation after that renders instantly (html.instant).
   Must stay tiny — it runs before first paint on purpose.
   ============================================================ */
(function () {
  try {
    var KEY = 'ds_nav';
    if (sessionStorage.getItem(KEY)) {
      document.documentElement.classList.add('instant');
    } else {
      sessionStorage.setItem(KEY, '1');
    }
  } catch (e) { /* storage blocked — animate every time, no harm */ }
})();
