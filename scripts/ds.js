#!/usr/bin/env node
/* DeepShield local backend controller.

A PID file plus command-line verification establishes ownership. Port discovery
is diagnostic only: this script never kills an unrelated process merely because
it happens to use DeepShield's configured port.
*/
const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

const ROOT = path.join(__dirname, '..');
const PORT = Number(process.env.PORT) || 5000;
const IS_WIN = process.platform === 'win32';
const PYTHON = path.join(ROOT, 'venv', IS_WIN ? 'Scripts/python.exe' : 'bin/python');
const LOG = path.join(ROOT, 'backend.log');
const PID_FILE = path.join(ROOT, '.deepshield.pid');
const APP_MARKERS = ['backend/app.py', 'backend\\app.py'];
const c = {
  dim: s => `\x1b[2m${s}\x1b[0m`, green: s => `\x1b[32m${s}\x1b[0m`,
  red: s => `\x1b[31m${s}\x1b[0m`, yellow: s => `\x1b[33m${s}\x1b[0m`,
  bold: s => `\x1b[1m${s}\x1b[0m`,
};

function sleepSync(ms) { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms); }

function listeners() {
  try {
    if (IS_WIN) {
      const out = execSync('netstat -ano', { encoding: 'utf8' });
      return [...new Set(out.split('\n')
        .filter(l => l.includes(`:${PORT}`) && l.includes('LISTENING'))
        .map(l => l.trim().split(/\s+/).pop())
        .filter(p => /^\d+$/.test(p) && p !== '0'))];
    }
    return execSync(`lsof -ti:${PORT}`, { encoding: 'utf8' })
      .split('\n').map(s => s.trim()).filter(Boolean);
  } catch { return []; }
}

function readPidFile() {
  try {
    const value = JSON.parse(fs.readFileSync(PID_FILE, 'utf8'));
    const pid = Number(value.pid);
    if (!Number.isInteger(pid) || pid <= 0 || Number(value.port) !== PORT) return null;
    return { ...value, pid };
  } catch { return null; }
}

function removePidFile() {
  try { fs.unlinkSync(PID_FILE); } catch { /* absent/stale */ }
}

function pidAlive(pid) {
  try { process.kill(Number(pid), 0); return true; } catch { return false; }
}

function processCommand(pid) {
  try {
    if (IS_WIN) {
      const script = `(Get-CimInstance Win32_Process -Filter \"ProcessId = ${Number(pid)}\").CommandLine`;
      return execSync(`powershell -NoProfile -Command "${script}"`, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
    }
    return execSync(`ps -p ${Number(pid)} -o args=`, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch { return ''; }
}

function ownedProcess() {
  const meta = readPidFile();
  if (!meta) return null;
  if (!pidAlive(meta.pid)) { removePidFile(); return null; }
  const cmd = processCommand(meta.pid);
  if (!APP_MARKERS.some(marker => cmd.includes(marker))) {
    // PID reuse/stale metadata: losing our PID file is safer than killing it.
    removePidFile();
    return null;
  }
  return meta;
}

function writePidFile(child) {
  const meta = {
    pid: child.pid,
    port: PORT,
    startedAt: new Date().toISOString(),
    root: ROOT,
  };
  fs.writeFileSync(PID_FILE, JSON.stringify(meta, null, 2), { encoding: 'utf8', mode: 0o600 });
}

function health(timeout = 1500) {
  return new Promise(resolve => {
    const req = http.get({ host: '127.0.0.1', port: PORT, path: '/api/health', timeout }, res => {
      let body = '';
      res.on('data', d => (body += d));
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch { resolve(null); } });
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}

function signalOwned(meta, force = false) {
  if (!meta || !pidAlive(meta.pid)) return;
  try {
    if (IS_WIN) {
      execSync(`taskkill /PID ${meta.pid} /T${force ? ' /F' : ''}`, { stdio: 'ignore' });
    } else {
      try { process.kill(-meta.pid, force ? 'SIGKILL' : 'SIGTERM'); }
      catch { process.kill(meta.pid, force ? 'SIGKILL' : 'SIGTERM'); }
    }
  } catch { /* it may have exited between checks */ }
}

function stop() {
  const meta = ownedProcess();
  const portPids = listeners();
  if (!meta) {
    if (portPids.length) {
      console.log(`\n  ${c.red('✗')} Refusing to stop PID ${portPids.join(', ')} on port ${PORT}: it is not owned by DeepShield.`);
      console.log(c.dim('     Stop that process yourself or choose a different PORT.\n'));
    } else {
      console.log(`\n  ${c.dim('Backend was not running — nothing to stop.')}\n`);
    }
    return false;
  }

  signalOwned(meta, false);
  for (let i = 0; i < 12 && pidAlive(meta.pid); i++) sleepSync(250);
  if (pidAlive(meta.pid)) signalOwned(meta, true);
  for (let i = 0; i < 8 && pidAlive(meta.pid); i++) sleepSync(250);
  const stopped = !pidAlive(meta.pid);
  if (stopped) removePidFile();
  console.log(stopped
    ? `\n  ${c.green('✓')} Stopped DeepShield PID ${meta.pid}.\n`
    : `\n  ${c.red('!')} DeepShield PID ${meta.pid} did not stop.\n`);
  return stopped;
}

async function status() {
  const pids = listeners();
  const owner = ownedProcess();
  const h = await health();
  if (!h && !pids.length) {
    console.log(`\n  ${c.red('●')} DeepShield backend is ${c.bold('stopped')}`);
    console.log(c.dim('     start it with: npm start\n'));
    return false;
  }
  if (!owner && pids.length) {
    console.log(`\n  ${c.yellow('●')} Port ${PORT} is occupied by unowned PID ${pids.join(', ')}`);
    console.log(c.dim('     DeepShield will not terminate it.\n'));
    return false;
  }
  if (!h) {
    console.log(`\n  ${c.yellow('●')} DeepShield PID ${owner ? owner.pid : '?'} is starting or unhealthy.\n`);
    return false;
  }
  const live = h.engine === 'live';
  console.log(`\n  ${c.green('●')} DeepShield backend is ${c.bold('running')} ${c.dim(`http://localhost:${PORT}`)}`);
  console.log(`     engine    ${live ? c.green('live — real model') : c.yellow('simulated (demo)')}`);
  if (live) {
    console.log(`     model     ${h.arch || '—'} ${c.dim(h.checkpoint || '')}`);
    const bits = [h.test_accuracy != null && `test ${h.test_accuracy}%`,
      h.val_accuracy != null && `val ${h.val_accuracy}%`,
      h.tpdn_accuracy != null && `TPDN ${h.tpdn_accuracy}%`,
      h.dfdc_accuracy != null && `DFDC ${h.dfdc_accuracy}%`].filter(Boolean);
    if (bits.length) console.log(`     accuracy  ${bits.join(c.dim(' · '))}`);
  }
  console.log(`     pid       ${owner ? owner.pid : 'untracked'}\n`);
  return true;
}

async function start() {
  const occupied = listeners();
  const owner = ownedProcess();
  if (occupied.length) {
    if (!owner) {
      console.log(`\n  ${c.red('✗')} Port ${PORT} is already used by PID ${occupied.join(', ')}; refusing to kill it.`);
      process.exitCode = 1;
      return false;
    }
    console.log(`\n  ${c.yellow('!')} DeepShield is already running — restarting its owned process.`);
    if (!stop()) return false;
  } else if (owner) {
    // Owned process exists but is not listening yet/stuck. Stop only that PID.
    if (!stop()) return false;
  }

  if (!fs.existsSync(PYTHON)) {
    console.log(`\n  ${c.red('✗')} Python venv not found at ${PYTHON}`);
    console.log(c.dim('     create it: python -m venv venv && install requirements\n'));
    process.exitCode = 1;
    return false;
  }

  const logFd = fs.openSync(LOG, 'a');
  fs.writeSync(logFd, `\n=== started ${new Date().toISOString()} ===\n`);
  const child = spawn(PYTHON, [path.join('backend', 'app.py')], {
    cwd: ROOT, detached: true, stdio: ['ignore', logFd, logFd],
    env: { ...process.env, PORT: String(PORT) },
  });
  writePidFile(child);
  child.unref();

  process.stdout.write(`\n  ${c.dim('starting backend')}`);
  for (let i = 0; i < 40; i++) {
    sleepSync(500);
    process.stdout.write(c.dim('.'));
    if (await health(800)) { process.stdout.write('\n'); return status(); }
    if (!pidAlive(child.pid)) break;
  }
  console.log(`\n  ${c.red('✗')} Did not come up — see ${c.bold('backend.log')}`);
  if (!pidAlive(child.pid)) removePidFile();
  console.log('');
  return false;
}

async function main() {
  const cmd = (process.argv[2] || 'status').toLowerCase();
  if (cmd === 'start') await start();
  else if (cmd === 'stop') stop();
  else if (cmd === 'restart') { if (stop() || !listeners().length) await start(); }
  else if (cmd === 'status') await status();
  else { console.log('\n  usage: node scripts/ds.js start | stop | restart | status\n'); process.exitCode = 1; }
}

if (require.main === module) main();
module.exports = { listeners, readPidFile, ownedProcess, pidAlive, processCommand, stop, start, status };
