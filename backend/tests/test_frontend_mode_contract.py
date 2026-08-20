import http.client
import json
import os
import shutil
import socket
import subprocess
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_node(source):
    completed = subprocess.run(
        [NODE, "-e", source], cwd=ROOT, text=True,
        capture_output=True, timeout=10, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_backend_outage_does_not_resolve_to_simulated():
    script = r"""
const fs = require('fs');
global.location = { protocol: 'https:' };
global.DS = {
  server: { health: async () => null },
  settings: { get: () => ({ frameRate: 1 }) },
  util: { hash: () => 1, formatBytes: () => '' },
};
eval(fs.readFileSync('frontend/assets/js/api.js', 'utf8'));
(async () => {
  const mode = await DS.api.resolveMode();
  if (mode !== 'unavailable') throw new Error(`expected unavailable, got ${mode}`);
  let error = null;
  try { await DS.api.analyze({ fileName: 'x.jpg', fileType: 'image' }); }
  catch (err) { error = err; }
  if (!error || error.code !== 'SERVER_UNAVAILABLE') {
    throw new Error('analysis did not fail closed with SERVER_UNAVAILABLE');
  }
})().catch(err => { console.error(err); process.exit(1); });
"""
    run_node(script)


def test_explicit_echo_health_resolves_to_simulated_demo():
    script = r"""
const fs = require('fs');
global.location = { protocol: 'http:' };
global.DS = {
  server: { health: async () => ({ engine: 'echo' }) },
  settings: { get: () => ({ frameRate: 1 }) },
  util: { hash: () => 1, formatBytes: () => '' },
};
eval(fs.readFileSync('frontend/assets/js/api.js', 'utf8'));
DS.api.resolveMode().then(mode => {
  if (mode !== 'simulated') throw new Error(`expected simulated, got ${mode}`);
}).catch(err => { console.error(err); process.exit(1); });
"""
    run_node(script)


def test_failed_health_is_not_cached_forever():
    script = r"""
const fs = require('fs');
let calls = 0;
global.window = { matchMedia: () => ({ matches: false }) };
global.document = {
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => ({ setAttribute(){}, appendChild(){}, querySelector(){ return { addEventListener(){} }; }, classList:{ add(){} } }),
  body: { appendChild(){} },
  documentElement: { dataset: {} },
  addEventListener() {},
};
global.CustomEvent = function(){};
global.fetch = async () => { calls += 1; throw new Error('offline'); };
global.DS = {
  util: { qsa: () => [] },
  api: { MODEL: {}, CERTAINTY: [] },
  auth: { user: () => null, logout(){} },
};
eval(fs.readFileSync('frontend/assets/js/components.js', 'utf8'));
(async () => {
  await DS.server.health();
  await DS.server.health();
  if (calls !== 2) throw new Error(`failed health was cached; fetch calls=${calls}`);
})().catch(err => { console.error(err); process.exit(1); });
"""
    run_node(script)


def test_toasts_render_untrusted_text_without_html_and_unavailable_is_distinct():
    script = r"""
const fs = require('fs');

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.attributes = {};
    this.className = '';
    this.innerHTML = '';
    this.textContent = '';
    this.classList = { add(){} };
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { this.children.push(...children); }
  addEventListener() {}
  remove() {}
  querySelector() { return null; }
}

let toastContainer = null;
global.window = { matchMedia: () => ({ matches: false }) };
global.document = {
  querySelector: selector => selector === '.toast-container' ? toastContainer : null,
  querySelectorAll: () => [],
  createElement: tag => new Element(tag),
  body: { appendChild(el) { if (el.className === 'toast-container') toastContainer = el; } },
  documentElement: { dataset: {} },
  addEventListener() {},
  dispatchEvent() {},
  getElementById() { return null; },
};
global.CustomEvent = function(){};
global.DS = {
  util: { qsa: () => [] },
  api: { MODEL: {}, CERTAINTY: [] },
  auth: { user: () => null, logout(){} },
};

eval(fs.readFileSync('frontend/assets/js/components.js', 'utf8'));

const payload = '<img src=x onerror=alert(1)>';
const title = '<svg onload=alert(2)>';
const toast = DS.toast(payload, 'error', { title, duration: 0 });
const copy = toast.children[1];
if (copy.children[0].textContent !== title || copy.children[1].textContent !== payload) {
  throw new Error('toast did not preserve untrusted input as literal text');
}
if (toast.innerHTML.includes(payload) || toast.innerHTML.includes(title)) {
  throw new Error('untrusted toast input reached innerHTML');
}

const badge = new Element();
DS.server.paintBadge(badge, 'unavailable');
if (!badge.innerHTML.includes('Model unavailable')) {
  throw new Error('unavailable model was not labelled distinctly');
}
if (badge.innerHTML.includes('Simulated')) {
  throw new Error('unavailable model was presented as simulated');
}
"""
    run_node(script)


def test_static_server_handles_bad_encoding_and_advertises_demo_mode():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = {**os.environ, "PORT": str(port)}
    process = subprocess.Popen(
        [NODE, "serve.js"], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.time() + 5
        while True:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                conn.request("GET", "/api/health")
                response = conn.getresponse()
                payload = json.loads(response.read())
                conn.close()
                if response.status == 200:
                    break
            except OSError:
                if time.time() >= deadline:
                    raise
                time.sleep(0.05)

        assert payload["engine"] == "echo"
        assert response.getheader("X-Content-Type-Options") == "nosniff"

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/%")
        bad = conn.getresponse()
        bad.read()
        conn.close()
        assert bad.status == 400
        assert bad.getheader("X-Content-Type-Options") == "nosniff"

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/%00")
        nul = conn.getresponse()
        nul.read()
        conn.close()
        assert nul.status == 400
        assert nul.getheader("X-Content-Type-Options") == "nosniff"

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/%2e%2e/%2e%2e/package.json")
        traversal = conn.getresponse()
        traversal.read()
        conn.close()
        assert traversal.status == 403
        assert traversal.getheader("X-Content-Type-Options") == "nosniff"

        # The malformed request must not terminate the server.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/api/health")
        alive = conn.getresponse()
        after = json.loads(alive.read())
        conn.close()
        assert alive.status == 200
        assert after["engine"] == "echo"
        assert process.poll() is None
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def test_process_controller_uses_argument_arrays_instead_of_shell_commands():
    with open(os.path.join(ROOT, "scripts", "ds.js"), encoding="utf-8") as file:
        source = file.read()
    assert "execSync" not in source
    assert "execFileSync" in source
    assert "'powershell.exe', ['-NoProfile', '-Command', script]" in source
