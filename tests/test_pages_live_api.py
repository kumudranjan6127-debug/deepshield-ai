from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_github_pages_routes_api_calls_to_documented_live_backend():
    utils = (ROOT / "frontend" / "assets" / "js" / "utils.js").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    match = re.search(
        r"'kumudranjan6127-debug\.github\.io'\s*:\s*'(https://[^']+)'",
        utils,
    )
    assert match, "GitHub Pages must have an explicit live API backend"

    backend = match.group(1)
    assert f"**Live:** <{backend}>" in readme, \
        "the Pages API bridge must match the documented live deployment"
    assert "input.startsWith('/api/')" in utils
    assert "nativeFetch(DS.apiUrl(input), init)" in utils


def test_render_allows_only_the_deepshield_pages_origin_for_cross_origin_api():
    blueprint = (ROOT / "render.yaml").read_text(encoding="utf-8")

    assert "- key: DS_CORS_ORIGINS" in blueprint
    assert 'value: "https://kumudranjan6127-debug.github.io"' in blueprint
    assert 'value: "*"' not in blueprint
