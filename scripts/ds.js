#!/usr/bin/env node
/* ============================================================
   DeepShield AI — server control
     node scripts/ds.js start | stop | restart | status

   Exists because Flask's debug reloader runs two processes: killing
   the child made the parent spawn a replacement, so instances piled
   up silently. Everything here works on the port, not on guesswork.
   ============================================================ */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

const ROOT = path.join(__dirname, '..');
const PORT = Number(process.env.PORT) || 5000;
const IS_WIN = process.platform === 'win32';
const PYTHON = path.join(ROOT, 'venv', IS_WIN ? 'Scripts/python.exe' : 'bin/python');
const LOG = path.join(ROOT, 'backend.log');

const c = {
  dim: s => `\x1b[2m${s}\x1b[0m`,
  green: s => `\x1b[32m${s}\x1b[0m`,
  red: s => `\x1b[31m${s}\x1b[0m`,
  yellow: s => `\x1b[33m${s}\x1b[0m`,
  bold: s => `\x1b[1m${s}\x1b[0m`,
};

/* ---- Who is holding the port? ---- */
function listeners() {
  try {
    if (IS_WIN) {
      const out = execSync('netstat -ano', { encoding: 'utf8' });
      return [...new Set(
        out.split('\n')
          .filter(l => l.includes(`:${PORT}`) && l.includes('LISTENING'))
          .map(l => l.trim().split(/\s+/).pop())
          .filter(p => /^\d+$/.test(p) && p !== '0')
      )];
    }
    return execSync(`lsof -ti:${PORT}`, { encoding: 'utf8' })
      .split('\n').map(s => s.trim()).filter(Boolean);
  } catch {
    return []; // nothing listening
  }
}

function killAll() {
  let killed = 0;
  // Loop: a debug-mode parent can respawn its child once
  for (let round = 0; round < 4; round++) {
    const pids = listeners();
    if (!pids.length) break;
    for (const pid of pids) {
      try {
        execSync(IS_WIN ? `taskkill /PID ${pid} /T /F` : `kill -9 ${pid}`,
                 { stdio: 'ignore' });
        killed++;
      } catch { /* already gone */ }
    }
    sleepSync(600);
  }
  return killed;
}

function sleepSync(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

/* ---- Ask the server what it is ---- */
function health(timeout = 1500) {
  return new Promise(resolve => {
    const req = http.get(
      { host: '127.0.0.1', port: PORT, path: '/api/health', timeout },
      res => {
        let body = '';
        res.on('data', d => (body += d));
        res.on('end', () => { try { resolve(JSON.parse(body)); } catch { resolve(null); } });
      });
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}

/* ---- Commands ---- */
async function status() {
  const pids = listeners();
  const h = await health();

  if (!h && !pids.length) {
    console.log(`\n  ${c.red('●')} DeepShield backend is ${c.bold('stopped')}`);
    console.log(c.dim(`     start it with:  npm start\n`));
    return false;
  }

  if (!h) {
    console.log(`\n  ${c.yellow('●')} Port ${PORT} is held by PID ${pids.join(', ')} but not answering`);
    console.log(c.dim(`     clear it with:  npm run stop\n`));
    return false;
  }

  const live = h.engine === 'live';
  console.log(`\n  ${c.green('●')} DeepShield backend is ${c.bold('running')}   ${c.dim(`http://localhost:${PORT}`)}`);
  console.log(`     engine    ${live ? c.green('live — real model') : c.yellow('simulated (demo)')}`);
  if (live) {
    console.log(`     model     ${h.arch || '—'}  ${c.dim(h.checkpoint || '')}`);
    const bits = [
      h.test_accuracy != null && `test ${h.test_accuracy}%`,
      h.val_accuracy != null && `val ${h.val_accuracy}%`,
      h.tpdn_accuracy != null && `TPDN ${h.tpdn_accuracy}%`,
      h.dfdc_accuracy != null && `DFDC ${h.dfdc_accuracy}%`,
    ].filter(Boolean);
    if (bits.length) console.log(`     accuracy  ${bits.join(c.dim(' · '))}`);
  }
  console.log(`     processes ${pids.length} ${c.dim(`(pid ${pids.join(', ')})`)}\n`);
  return true;
}

function stop() {
  const before = listeners();
  if (!before.length) {
    console.log(`\n  ${c.dim('Backend was not running — nothing to stop.')}\n`);
    return;
  }
  const killed = killAll();
  const left = listeners();
  if (left.length) {
    console.log(`\n  ${c.red('!')} ${left.length} process(es) still holding port ${PORT}: ${left.join(', ')}`);
    console.log(c.dim(`     try again, or:  taskkill /PID ${left[0]} /T /F\n`));
  } else {
    console.log(`\n  ${c.green('✓')} Stopped — ${killed} process(es) terminated, port ${PORT} free.\n`);
  }
}

async function start() {
  if (listeners().length) {
    console.log(`\n  ${c.yellow('!')} Already running — restarting instead.`);
    stop();
  }
  if (!fs.existsSync(PYTHON)) {
    console.log(`\n  ${c.red('✗')} Python venv not found at ${PYTHON}`);
    console.log(c.dim('     create it:  python -m venv venv  &&  venv\\Scripts\\pip install -r requirements.txt\n'));
    process.exit(1);
  }

  const log = fs.openSync(LOG, 'a');
  fs.writeSync(log, `\n=== started ${new Date().toISOString()} ===\n`);
  const child = spawn(PYTHON, [path.join('backend', 'app.py')], {
    cwd: ROOT,
    detached: true,
    stdio: ['ignore', log, log],
    env: { ...process.env, PORT: String(PORT) },
  });
  child.unref();

  process.stdout.write(`\n  ${c.dim('starting backend')}`);
  for (let i = 0; i < 40; i++) {          // models take a few seconds to load
    sleepSync(500);
    process.stdout.write(c.dim('.'));
    if (await health(800)) { process.stdout.write('\n'); return status(); }
  }
  console.log(`\n  ${c.red('✗')} Did not come up in 20s — see ${c.bold('backend.log')}\n`);
}

/* ---- Entry ---- */
(async () => {
  const cmd = (process.argv[2] || 'status').toLowerCase();
  if (cmd === 'start') await start();
  else if (cmd === 'stop') stop();
  else if (cmd === 'restart') { stop(); await start(); }
  else if (cmd === 'status') await status();
  else {
    console.log('\n  usage: node scripts/ds.js start | stop | restart | status\n');
    process.exit(1);
  }
})();
