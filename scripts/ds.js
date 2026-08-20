#!/usr/bin/env node
/* ============================================================
   DeepShield AI — server control
     node scripts/ds.js start | stop | restart | status

   The controller owns only processes it started and recorded. A port is
   useful for diagnostics, but it is never proof that a process belongs to
   DeepShield and is therefore never sufficient authority to kill it.
   ============================================================ */

const { execFileSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

const ROOT = path.join(__dirname, '..');
const PORT = Number(process.env.PORT) || 5000;
const IS_WIN = process.platform === 'win32';
const PYTHON = path.join(ROOT, 'venv', IS_WIN ? 'Scripts/python.exe' : 'bin/python');
const LOG = path.join(ROOT, 'backend.log');
const PID_FILE = process.env.DS_PID_FILE || path.join(ROOT, '.deepshield.pid');

const c = {
  dim: s => `\x1b[2m${s}\x1b[0m`,
  green: s => `\x1b[32m${s}\x1b[0m`,
  red: s => `\x1b[31m${s}\x1b[0m`,
  yellow: s => `\x1b[33m${s}\x1b[0m`,
  bold: s => `\x1b[1m${s}\x1b[0m`,
};

function listeners() {
  try {
    if (IS_WIN) {
      const out = execFileSync('netstat.exe', ['-ano'], { encoding: 'utf8' });
      return [...new Set(
        out.split('\n')
          .filter(l => l.includes(`:${PORT}`) && l.includes('LISTENING'))
          .map(l => l.trim().split(/\s+/).pop())
          .filter(p => /^\d+$/.test(p) && p !== '0')
      )];
    }
    return execFileSync('lsof', [`-ti:${PORT}`], { encoding: 'utf8' })
      .split('\n').map(s => s.trim()).filter(Boolean);
  } catch {
    return [];
  }
}

function sleepSync(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function linuxProcessState(pid) {
  if (process.platform !== 'linux') return null;
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    const end = stat.lastIndexOf(')');
    if (end < 0) return null;
    return stat.slice(end + 2).trim().split(/\s+/)[0] || null;
  } catch {
    return null;
  }
}

function processIdentity(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return null;
  try {
    if (process.platform === 'linux') {
      const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
      const end = stat.lastIndexOf(')');
      if (end < 0) return null;
      // After "pid (comm)" the first token is field 3 (state), so field 22
      // (process start time since boot) is token index 19 here. Unlike a PID,
      // it cannot be silently reused by a different process later.
      const fields = stat.slice(end + 2).trim().split(/\s+/);
      return fields[19] ? `linux:${fields[19]}` : null;
    }
    if (IS_WIN) {
      const script = `(Get-Process -Id ${pid}).StartTime.ToUniversalTime().Ticks`;
      const out = execFileSync(
        'powershell.exe', ['-NoProfile', '-Command', script],
        { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }
      ).trim();
      return out ? `win:${out}` : null;
    }
    const out = execFileSync(
      'ps', ['-o', 'lstart=', '-p', String(pid)], { encoding: 'utf8' }
    ).trim();
    return out ? `ps:${out}` : null;
  } catch {
    return null;
  }
}

function pidAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    // A zombie still has a PID and therefore passes signal 0, but it has
    // already exited and cannot serve anything. Treat it as stopped so a
    // parent that has not reaped it yet does not make shutdown look failed.
    if (linuxProcessState(pid) === 'Z') return false;
    return true;
  } catch {
    return false;
  }
}

function removePidFile() {
  try { fs.unlinkSync(PID_FILE); } catch (err) {
    if (err && err.code !== 'ENOENT') throw err;
  }
}

function readOwnedPid() {
  let record;
  try {
    record = JSON.parse(fs.readFileSync(PID_FILE, 'utf8'));
  } catch (err) {
    if (err && err.code !== 'ENOENT') {
      // Corrupt metadata must never become authority to kill a guessed PID.
      try { removePidFile(); } catch { /* best effort */ }
    }
    return null;
  }

  const pid = Number(record && record.pid);
  const port = Number(record && record.port);
  const identity = record && record.identity;
  if (!Number.isInteger(pid) || pid <= 0 || port !== PORT || !identity) {
    try { removePidFile(); } catch { /* best effort */ }
    return null;
  }
  if (!pidAlive(pid) || processIdentity(pid) !== identity) {
    // A stale PID file or a PID that has been recycled is not ownership.
    try { removePidFile(); } catch { /* best effort */ }
    return null;
  }
  return pid;
}

function writeOwnedPid(pid) {
  const identity = processIdentity(pid);
  if (!identity) throw new Error(`Could not identify spawned process ${pid}`);
  const tmp = `${PID_FILE}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify({
    pid,
    port: PORT,
    identity,
    startedAt: new Date().toISOString(),
  }) + '\n', { encoding: 'utf8', mode: 0o600 });
  fs.renameSync(tmp, PID_FILE);
}

function signalOwned(pid, signal) {
  if (IS_WIN) {
    const args = ['/PID', String(pid), '/T'];
    if (signal === 'SIGKILL') args.push('/F');
    try {
      execFileSync('taskkill.exe', args, { stdio: 'ignore' });
      return true;
    } catch {
      return !pidAlive(pid);
    }
  }

  // A detached child is normally the leader of its own process group. Fall
  // back to the PID itself for tests/legacy pidfiles that are not group leaders.
  try {
    process.kill(-pid, signal);
    return true;
  } catch {
    try {
      process.kill(pid, signal);
      return true;
    } catch {
      return !pidAlive(pid);
    }
  }
}

function terminateOwned(pid) {
  if (!pidAlive(pid)) return true;
  signalOwned(pid, 'SIGTERM');
  for (let i = 0; i < 10 && pidAlive(pid); i++) sleepSync(100);
  if (pidAlive(pid)) {
    signalOwned(pid, 'SIGKILL');
    for (let i = 0; i < 10 && pidAlive(pid); i++) sleepSync(100);
  }
  return !pidAlive(pid);
}

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

async function status() {
  const owned = readOwnedPid();
  const pids = listeners();
  const h = await health();

  if (!h && !pids.length) {
    console.log(`\n  ${c.red('●')} DeepShield backend is ${c.bold('stopped')}`);
    console.log(c.dim(`     start it with:  npm start\n`));
    return false;
  }

  if (!h) {
    console.log(`\n  ${c.yellow('●')} Port ${PORT} is held by PID ${pids.join(', ')} but is not answering DeepShield health`);
    console.log(c.dim('     The controller will not terminate an unowned port listener.\n'));
    return false;
  }

  const live = h.engine === 'live';
  const ownership = owned ? `owned pid ${owned}` : 'not owned by this controller';
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
  console.log(`     process   ${ownership}\n`);
  return true;
}

function stop() {
  const pid = readOwnedPid();
  if (!pid) {
    const pids = listeners();
    if (pids.length) {
      console.log(`\n  ${c.yellow('!')} Port ${PORT} is in use by PID ${pids.join(', ')}, but no owned DeepShield PID is recorded.`);
      console.log(c.dim('     Refusing to terminate an unrelated/unowned process.\n'));
    } else {
      console.log(`\n  ${c.dim('Backend was not running — nothing owned to stop.')}\n`);
    }
    return true;
  }

  const stopped = terminateOwned(pid);
  if (stopped) {
    removePidFile();
    console.log(`\n  ${c.green('✓')} Stopped owned DeepShield process ${pid}.\n`);
    return true;
  }

  console.log(`\n  ${c.red('!')} Could not stop owned DeepShield process ${pid}.`);
  console.log(c.dim(`     PID metadata kept at ${PID_FILE}; investigate before forcing termination.\n`));
  return false;
}

async function start() {
  const owned = readOwnedPid();
  if (owned) {
    console.log(`\n  ${c.yellow('!')} Owned DeepShield process ${owned} is already running — restarting it.`);
    if (!stop()) return false;
  } else {
    const occupied = listeners();
    if (occupied.length) {
      console.log(`\n  ${c.red('✗')} Port ${PORT} is already in use by unowned PID ${occupied.join(', ')}.`);
      console.log(c.dim('     Stop that application or choose another PORT; DeepShield will not kill it.\n'));
      return false;
    }
  }

  if (!fs.existsSync(PYTHON)) {
    console.log(`\n  ${c.red('✗')} Python venv not found at ${PYTHON}`);
    console.log(c.dim(`     create it:  python -m venv venv  &&  ${IS_WIN ? 'venv\\Scripts\\pip' : 'venv/bin/pip'} install -r requirements.txt\n`));
    return false;
  }

  const log = fs.openSync(LOG, 'a');
  fs.writeSync(log, `\n=== started ${new Date().toISOString()} ===\n`);
  const child = spawn(PYTHON, [path.join('backend', 'app.py')], {
    cwd: ROOT,
    detached: true,
    stdio: ['ignore', log, log],
    env: { ...process.env, PORT: String(PORT) },
  });
  fs.closeSync(log);
  try {
    writeOwnedPid(child.pid);
  } catch (err) {
    terminateOwned(child.pid);
    throw err;
  }
  child.unref();

  process.stdout.write(`\n  ${c.dim('starting backend')}`);
  for (let i = 0; i < 40; i++) {
    sleepSync(500);
    process.stdout.write(c.dim('.'));
    if (await health(800)) { process.stdout.write('\n'); return status(); }
    if (!pidAlive(child.pid)) break;
  }

  console.log(`\n  ${c.red('✗')} Did not come up in 20s — see ${c.bold('backend.log')}`);
  if (terminateOwned(child.pid)) removePidFile();
  console.log('');
  return false;
}

async function main() {
  const cmd = (process.argv[2] || 'status').toLowerCase();
  let ok = true;
  if (cmd === 'start') ok = await start();
  else if (cmd === 'stop') ok = stop();
  else if (cmd === 'restart') { ok = stop(); if (ok) ok = await start(); }
  else if (cmd === 'status') ok = await status();
  else {
    console.log('\n  usage: node scripts/ds.js start | stop | restart | status\n');
    ok = false;
  }
  if (!ok) process.exitCode = 1;
}

module.exports = {
  listeners, pidAlive, processIdentity, readOwnedPid, writeOwnedPid,
  terminateOwned, health, status, stop, start,
};

if (require.main === module) main();
