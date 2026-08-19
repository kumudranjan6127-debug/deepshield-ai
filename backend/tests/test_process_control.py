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


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_controller(command, *, port, pid_file):
    env = {**os.environ, "PORT": str(port), "DS_PID_FILE": str(pid_file)}
    return subprocess.run(
        [NODE, "scripts/ds.js", command], cwd=ROOT, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
    )


def process_identity(pid):
    source = (
        "const ds=require('./scripts/ds.js');"
        f"process.stdout.write(ds.processIdentity({int(pid)}) || '');"
    )
    result = subprocess.run(
        [NODE, "-e", source], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout
    return result.stdout


def test_stop_terminates_only_the_recorded_owned_pid(tmp_path):
    port = free_port()
    pid_file = tmp_path / "owned.pid"
    child = subprocess.Popen([NODE, "-e", "setInterval(() => {}, 1000)"])
    identity = process_identity(child.pid)
    pid_file.write_text(
        json.dumps({"pid": child.pid, "port": port, "identity": identity}),
        encoding="utf-8",
    )

    try:
        result = run_controller("stop", port=port, pid_file=pid_file)
        assert result.returncode == 0, result.stderr or result.stdout
        child.wait(timeout=5)
        assert not pid_file.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


def test_stale_identity_never_authorizes_killing_reused_pid(tmp_path):
    port = free_port()
    pid_file = tmp_path / "stale.pid"
    child = subprocess.Popen([NODE, "-e", "setInterval(() => {}, 1000)"])
    pid_file.write_text(
        json.dumps({"pid": child.pid, "port": port, "identity": "stale-process"}),
        encoding="utf-8",
    )

    try:
        result = run_controller("stop", port=port, pid_file=pid_file)
        assert result.returncode == 0, result.stderr or result.stdout
        assert child.poll() is None, "stale PID metadata authorized a kill"
        assert not pid_file.exists()
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=3)


def test_stop_refuses_to_kill_unowned_port_listener(tmp_path):
    port = free_port()
    pid_file = tmp_path / "missing.pid"
    script = (
        "const http=require('http');"
        f"http.createServer((q,r)=>r.end('other app')).listen({port},'127.0.0.1');"
        "setInterval(()=>{},1000);"
    )
    other = subprocess.Popen([NODE, "-e", script])

    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("unowned test listener did not start")

        result = run_controller("stop", port=port, pid_file=pid_file)
        assert result.returncode == 0, result.stderr or result.stdout
        assert other.poll() is None, "controller killed a process it did not own"
        assert not pid_file.exists()
    finally:
        if other.poll() is None:
            other.terminate()
            try:
                other.wait(timeout=3)
            except subprocess.TimeoutExpired:
                other.kill()
                other.wait(timeout=3)
