/* ============================================================
   DeepShield AI — landing.js
   Turns the page into a running product demo:
   mobile menu · scroll progress · scrollspy · reveals · count-up
   stats · auto-playing analysis mockup · pointer tilt ·
   drag-to-compare heatmap · pipeline signal.

   Every animation is transform/opacity only and pauses when off
   screen, and the whole file no-ops under reduced motion.
   ============================================================ */

/* Marks JS as available so CSS can hide reveal elements. Without JS the
   content stays visible — progressive enhancement, never a blank page. */
document.documentElement.classList.add('js');

document.addEventListener('DOMContentLoaded', () => {
  const reduced =
    document.documentElement.dataset.reducedMotion === 'true' ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  initMenu();
  initScroll(reduced);
  initReveals(reduced);
  initDemo(reduced);
  initTilt(reduced);
  initCompare();
  initPipeline(reduced);
});

/* ---- Mobile menu ---- */
function initMenu() {
  const burger = document.getElementById('lp-burger');
  const links = document.getElementById('lp-links');
  burger.addEventListener('click', () => links.classList.toggle('open'));
  links.addEventListener('click', e => {
    if (e.target.tagName === 'A') links.classList.remove('open');
  });
}

/* ---- Sticky nav state, reading progress, scrollspy ----
   One passive scroll listener, all writes batched into a rAF. */
function initScroll(reduced) {
  const nav = document.getElementById('lp-nav');
  const bar = document.getElementById('lp-progress');
  const links = DS.util.qsa('.lp-links a[href^="#"]');
  const sections = links
    .map(a => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  let ticking = false;
  const update = () => {
    const y = window.scrollY;
    nav.classList.toggle('scrolled', y > 8);

    const max = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? Math.min(1, y / max) : 0})`;

    // Current section = last one whose top has passed the nav
    let current = -1;
    sections.forEach((sec, i) => {
      if (sec.getBoundingClientRect().top <= 120) current = i;
    });
    links.forEach((a, i) => a.classList.toggle('current', i === current));

    ticking = false;
  };

  update();
  window.addEventListener('scroll', () => {
    if (ticking) return;
    ticking = true;
    reduced ? update() : requestAnimationFrame(update);
  }, { passive: true });
}

/* ---- Reveal on scroll + count-up numbers ---- */
function initReveals(reduced) {
  const items = DS.util.qsa('.reveal');

  const countUp = el => {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || '';
    const decimals = (el.dataset.count.split('.')[1] || '').length;
    if (reduced) { el.textContent = target.toFixed(decimals) + suffix; return; }

    const start = performance.now();
    const step = now => {
      const t = Math.min(1, (now - start) / 1100);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = (eased * target).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  const activate = el => {
    el.classList.add('in');
    DS.util.qsa('[data-count]', el).forEach(countUp);
  };

  if (reduced || !('IntersectionObserver' in window)) {
    items.forEach(activate);
    return;
  }

  const io = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      activate(entry.target);
      obs.unobserve(entry.target); // reveal once, then stop watching
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });

  items.forEach(el => io.observe(el));
}

/* ---- Hero mockup: an actual analysis loop, not a static picture ----
   Two files cycle (one fake, one real) so visitors see both outcomes. */
function initDemo(reduced) {
  const el = id => document.getElementById(id);
  const ui = {
    body: el('mock-body'), ring: el('mock-ring-value'), pct: el('mock-pct'),
    cap: el('mock-cap'), scan: el('mock-scan'), verdict: el('mock-verdict'),
    votes: DS.util.qsa('#mock-votes .mock-vote i'), file: el('mock-file'),
  };
  if (!ui.ring) return;

  const C = 327;                       // 2πr for r = 52
  const CASES = [
    { file: 'profile_photo.jpg', fake: true,  score: 94, votes: [91, 78, 99] },
    { file: 'family_pic.jpg',    fake: false, score: 97, votes: [4, 12, 2] },
  ];

  const paint = (pct, cls) => {
    ui.ring.style.strokeDashoffset = String(C * (1 - pct / 100));
    ui.ring.className.baseVal = `mock-ring-value${cls ? ' ' + cls : ''}`;
    ui.pct.textContent = `${Math.round(pct)}%`;
  };

  let idx = 0, timers = [], running = false;
  const at = (ms, fn) => timers.push(setTimeout(fn, ms));

  function play() {
    if (running) return;
    running = true;
    timers.forEach(clearTimeout);
    timers = [];

    const c = CASES[idx % CASES.length];
    idx++;

    // reset
    ui.file.textContent = c.file;
    ui.scan.classList.remove('done');
    ui.cap.textContent = 'analyzing';
    ui.verdict.className = 'badge mock-verdict';
    ui.verdict.innerHTML =
      '<span class="loader" style="width:13px;height:13px;border-width:2px"></span> Working…';
    ui.votes.forEach(v => v.style.setProperty('--w', '0%'));
    paint(0, '');

    if (reduced) { finish(c); return; }

    // ramp the ring while "analyzing"
    const start = performance.now();
    const DURATION = 2200;
    const tick = now => {
      const t = Math.min(1, (now - start) / DURATION);
      paint(t * c.score, '');
      if (t < 1 && running) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    c.votes.forEach((v, i) => at(700 + i * 320, () =>
      ui.votes[i].style.setProperty('--w', `${v}%`)));

    at(DURATION + 120, () => finish(c));
    at(DURATION + 3400, () => { running = false; play(); }); // next case
  }

  function finish(c) {
    paint(c.score, c.fake ? 'fake' : 'real');
    ui.cap.textContent = 'confidence';
    ui.scan.classList.add('done');
    ui.verdict.className = `badge mock-verdict ${c.fake ? 'badge-danger' : 'badge-success'}`;
    ui.verdict.innerHTML = c.fake
      ? '<i data-lucide="alert-triangle" class="icon-sm"></i> Likely Deepfake'
      : '<i data-lucide="shield-check" class="icon-sm"></i> Likely Real';
    c.votes.forEach((v, i) => ui.votes[i].style.setProperty('--w', `${v}%`));
    DS.icons();
    if (reduced) running = false;
  }

  // Only animate while the mockup is actually on screen
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) play();
        else { timers.forEach(clearTimeout); timers = []; running = false; }
      });
    }, { threshold: 0.25 }).observe(ui.body);
  } else {
    play();
  }
}

/* ---- Pointer tilt on the hero mockup (hover devices only) ---- */
function initTilt(reduced) {
  const card = document.querySelector('.mock-card');
  const zone = document.querySelector('.lp-hero-visual');
  if (!card || !zone || reduced) return;
  if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;

  let raf = null, px = 0, py = 0;
  zone.addEventListener('pointermove', e => {
    const r = zone.getBoundingClientRect();
    px = (e.clientX - r.left) / r.width - 0.5;
    py = (e.clientY - r.top) / r.height - 0.5;
    if (raf) return;
    raf = requestAnimationFrame(() => {
      card.style.setProperty('--ry', `${px * 7}deg`);
      card.style.setProperty('--rx', `${-py * 7}deg`);
      raf = null;
    });
  }, { passive: true });

  zone.addEventListener('pointerleave', () => {
    card.style.setProperty('--ry', '0deg');
    card.style.setProperty('--rx', '0deg');
  });
}

/* ---- Drag-to-compare: photo ⇄ model attention ---- */
function initCompare() {
  const wrap = document.getElementById('ex-compare');
  const slider = document.getElementById('ex-slider');
  if (!wrap || !slider) return;

  const apply = v => wrap.style.setProperty('--split', `${v}%`);
  apply(slider.value);
  slider.addEventListener('input', () => apply(slider.value));
}

/* ---- Pipeline: a signal walks through the nodes ---- */
function initPipeline(reduced) {
  const nodes = DS.util.qsa('.lp-pipeline .pipe-node');
  if (!nodes.length || reduced || !('IntersectionObserver' in window)) return;

  let timer = null;
  const step = (() => {
    let i = 0;
    return () => {
      nodes.forEach((n, j) => n.classList.toggle('lit', j === i % nodes.length));
      i++;
    };
  })();

  new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting && !timer) {
        step();
        timer = setInterval(step, 700);
      } else if (!e.isIntersecting && timer) {
        clearInterval(timer);
        timer = null;
        nodes.forEach(n => n.classList.remove('lit'));
      }
    });
  }, { threshold: 0.3 }).observe(document.querySelector('.lp-pipeline'));
}
