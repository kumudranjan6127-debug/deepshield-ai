/* ============================================================
   DeepShield AI — login.js

   Two auth modes, decided automatically at load:
   • FIREBASE mode — real accounts (email/password + Google popup)
     when firebase-config.js is filled AND the SDK loaded.
   • DEMO mode — local-only sign-in (any credentials), used while
     the config has PASTE_ placeholders or the CDN is unreachable.

   Either way, the logged-in user is mirrored into localStorage
   (DS.auth.login) — every other page keeps working unchanged.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  /* ---- Which mode are we in? ---- */
  const cfg = window.DS_FIREBASE_CONFIG;
  const firebaseReady =
    typeof firebase !== 'undefined' &&
    cfg && cfg.apiKey && !String(cfg.apiKey).startsWith('PASTE_');

  let fbAuth = null;
  if (firebaseReady) {
    firebase.initializeApp(cfg);
    fbAuth = firebase.auth();
  }

  /* ---- Arrived via logout? Also end the Firebase session ---- */
  if (new URLSearchParams(location.search).get('signout') && fbAuth) {
    fbAuth.signOut().catch(() => {});
  }

  /* ---- Already signed in (local mirror) → straight to the app ---- */
  if (DS.auth.user()) { window.location.replace('dashboard.html'); return; }

  /* ---- Elements ---- */
  const form       = document.getElementById('login-form');
  const emailInput = document.getElementById('email');
  const passInput  = document.getElementById('password');
  const emailField = document.getElementById('field-email');
  const passField  = document.getElementById('field-password');
  const passError  = passField.querySelector('.field-error');
  const submitBtn  = document.getElementById('login-btn');
  const toggleBtn  = document.getElementById('toggle-password');
  const guestBtn   = document.getElementById('guest-btn');
  const googleBtn  = document.getElementById('google-btn');
  const modeToggle = document.getElementById('mode-toggle');
  const title      = document.getElementById('auth-title');
  const sub        = document.getElementById('auth-sub');
  const note       = document.getElementById('auth-note');

  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const MIN_PASS = firebaseReady ? 6 : 4; // Firebase enforces ≥ 6

  let mode = 'signin'; // 'signin' | 'signup' (signup exists only in Firebase mode)

  /* ---- Mode-specific UI ---- */
  if (firebaseReady) {
    googleBtn.hidden = false;
    modeToggle.hidden = false;
    note.textContent = 'Secured by Firebase Authentication.';
    passError.textContent = `Password must be at least ${MIN_PASS} characters.`;
  }

  function applyMode() {
    const signup = mode === 'signup';
    title.textContent = signup ? 'Create your account' : 'Welcome back';
    sub.textContent   = signup ? 'Join DeepShield in seconds.' : 'Sign in to continue to DeepShield.';
    submitBtn.innerHTML = signup
      ? 'Create account <i data-lucide="arrow-right" class="icon-sm"></i>'
      : 'Sign in <i data-lucide="arrow-right" class="icon-sm"></i>';
    modeToggle.innerHTML = signup
      ? 'Already have an account? <span>Sign in</span>'
      : 'New to DeepShield? <span>Create an account</span>';
    passInput.setAttribute('autocomplete', signup ? 'new-password' : 'current-password');
    DS.icons();
  }
  modeToggle.addEventListener('click', () => {
    mode = mode === 'signin' ? 'signup' : 'signin';
    applyMode();
  });

  /* ---- Password visibility toggle ---- */
  toggleBtn.addEventListener('click', () => {
    const show = passInput.type === 'password';
    passInput.type = show ? 'text' : 'password';
    toggleBtn.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
    toggleBtn.innerHTML = `<i data-lucide="${show ? 'eye-off' : 'eye'}" class="icon-sm"></i>`;
    DS.icons();
  });

  /* ---- Clear error state while typing ---- */
  emailInput.addEventListener('input', () => emailField.classList.remove('invalid'));
  passInput.addEventListener('input', () => passField.classList.remove('invalid'));

  /* ---- Busy state (restores the label on failure) ---- */
  let savedLabel = '';
  function setBusy(busy, label) {
    submitBtn.disabled = guestBtn.disabled = googleBtn.disabled = busy;
    if (busy) {
      savedLabel = submitBtn.innerHTML;
      submitBtn.innerHTML = `<span class="loader" aria-hidden="true"></span> ${label}`;
    } else if (savedLabel) {
      submitBtn.innerHTML = savedLabel;
      DS.icons();
    }
  }

  /* ---- Enter the app: mirror the user locally, then go ---- */
  function enter(name, email) {
    DS.auth.login(name, email);
    window.location.href = 'dashboard.html';
  }

  /* ---- Submit (both modes) ---- */
  form.addEventListener('submit', async e => {
    e.preventDefault();
    emailField.classList.remove('invalid');
    passField.classList.remove('invalid');

    const email = emailInput.value.trim();
    const pass  = passInput.value;
    const errors = [];

    if (!EMAIL_RE.test(email)) {
      emailField.classList.add('invalid');
      errors.push('Enter a valid email address.');
    }
    if (pass.length < MIN_PASS) {
      passField.classList.add('invalid');
      errors.push(`Password must be at least ${MIN_PASS} characters.`);
    }
    if (errors.length) { DS.toast(errors[0], 'error'); return; }

    /* DEMO mode — brief spinner, then local sign-in */
    if (!firebaseReady) {
      setBusy(true, 'Signing in…');
      setTimeout(() => enter(displayNameFrom(email), email), 700);
      return;
    }

    /* FIREBASE mode */
    setBusy(true, mode === 'signup' ? 'Creating account…' : 'Signing in…');
    try {
      let cred;
      if (mode === 'signup') {
        cred = await fbAuth.createUserWithEmailAndPassword(email, pass);
        await cred.user.updateProfile({ displayName: displayNameFrom(email) });
      } else {
        cred = await fbAuth.signInWithEmailAndPassword(email, pass);
      }
      enter(cred.user.displayName || displayNameFrom(email), email);
    } catch (err) {
      setBusy(false);
      DS.toast(friendlyError(err), 'error');
    }
  });

  /* ---- Google sign-in (Firebase mode only) ---- */
  googleBtn.addEventListener('click', async () => {
    setBusy(true, 'Waiting for Google…');
    try {
      const provider = new firebase.auth.GoogleAuthProvider();
      const cred = await fbAuth.signInWithPopup(provider);
      enter(cred.user.displayName || displayNameFrom(cred.user.email || 'user@x.co'),
            cred.user.email || '');
    } catch (err) {
      setBusy(false);
      if (err && err.code !== 'auth/popup-closed-by-user') {
        DS.toast(friendlyError(err), 'error');
      }
    }
  });

  /* ---- Guest access (works in both modes, always local) ---- */
  guestBtn.addEventListener('click', () => {
    DS.auth.login('Guest', 'guest@deepshield.local');
    window.location.href = 'dashboard.html';
  });

  applyMode();
});

/* Firebase error codes → human sentences */
function friendlyError(err) {
  const map = {
    'auth/invalid-credential':      'Wrong email or password.',
    'auth/wrong-password':          'Wrong email or password.',
    'auth/user-not-found':          'No account with this email — try Create account.',
    'auth/email-already-in-use':    'Account already exists — try signing in instead.',
    'auth/weak-password':           'Password must be at least 6 characters.',
    'auth/invalid-email':           'Enter a valid email address.',
    'auth/too-many-requests':       'Too many attempts — wait a minute and retry.',
    'auth/network-request-failed':  'Network error — check your internet connection.',
    'auth/unauthorized-domain':     'Open the app at http://localhost:5000 for Google sign-in.',
  };
  return (err && map[err.code]) || 'Sign-in failed. Please try again.';
}

/* "harsh.goswami" → "Harsh Goswami" (split on . _ -, capitalize each) */
function displayNameFrom(email) {
  return email.split('@')[0]
    .split(/[._-]+/)
    .filter(Boolean)
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ') || 'User';
}
