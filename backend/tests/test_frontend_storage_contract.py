import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def test_feedback_key_is_registered_and_cleared_with_history():
    script = r"""
const fs = require('fs');
function storage() {
  const data = new Map();
  return {
    getItem: key => data.has(key) ? data.get(key) : null,
    setItem: (key, value) => data.set(key, value),
    removeItem: key => data.delete(key),
    has: key => data.has(key),
  };
}
const localStorage = storage();
const sessionStorage = storage();
global.DS = {};
global.window = { DS: global.DS, localStorage, sessionStorage, location: {} };
global.document = {
  documentElement: { dataset: {} },
  querySelector: () => null,
  querySelectorAll: () => [],
};
global.crypto = require('crypto').webcrypto;
eval(fs.readFileSync('frontend/assets/js/utils.js', 'utf8'));
if (DS.KEYS.FEEDBACK !== 'ds_feedback') throw new Error('feedback key is not registered');
DS.store.set(DS.KEYS.HISTORY, [{id:'scan'}]);
DS.store.set(DS.KEYS.FEEDBACK, [{scanId:'scan', agree:true}]);
DS.history.clear();
if (localStorage.has(DS.KEYS.HISTORY)) throw new Error('history was not cleared');
if (localStorage.has(DS.KEYS.FEEDBACK)) throw new Error('feedback survived clear history');
"""
    completed = subprocess.run(
        [NODE, "-e", script], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
