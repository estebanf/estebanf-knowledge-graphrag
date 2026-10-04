import importlib.util
import os
import tomllib
from pathlib import Path

import pytest


def _load_script():
    script_path = Path(__file__).parent.parent / "scripts" / "remove_api_key.py"
    assert script_path.exists()
    spec = importlib.util.spec_from_file_location("remove_api_key_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


KEYS = '''# operator keys
[[keys]]
id = "laptop-cli"
token = "tok-a"
scopes = ["read", "ingest", "admin"]

[[keys]]
id = "smoke-cli"
token = "tok-smoke"
scopes = [
    "read",
    "ingest",
    "admin",
]

[[keys]]
id = "mcp"
token = "tok-c"
scopes = ["read"]
'''


def _ids(path):
    return [k["id"] for k in tomllib.loads(path.read_text())["keys"]]


@pytest.mark.parametrize("target", ["laptop-cli", "smoke-cli", "mcp"])
def test_removes_only_target_block_and_preserves_inode(tmp_path, target):
    mod = _load_script()
    path = tmp_path / "api_keys.toml"
    path.write_text(KEYS)
    inode_before = os.stat(path).st_ino

    removed = mod.remove_api_key(path, target)

    assert removed == 1
    assert os.stat(path).st_ino == inode_before
    assert _ids(path) == [i for i in ["laptop-cli", "smoke-cli", "mcp"] if i != target]
    assert path.read_text().startswith("# operator keys")


def test_removed_token_is_gone_and_other_tokens_survive(tmp_path):
    mod = _load_script()
    path = tmp_path / "api_keys.toml"
    path.write_text(KEYS)

    mod.remove_api_key(path, "smoke-cli")

    content = path.read_text()
    assert "tok-smoke" not in content
    parsed = tomllib.loads(content)
    assert [(k["id"], k["token"]) for k in parsed["keys"]] == [("laptop-cli", "tok-a"), ("mcp", "tok-c")]
    assert parsed["keys"][1]["scopes"] == ["read"]


def test_noop_when_id_absent(tmp_path):
    mod = _load_script()
    path = tmp_path / "api_keys.toml"
    path.write_text(KEYS)
    mtime_before = os.stat(path).st_mtime_ns

    assert mod.remove_api_key(path, "does-not-exist") == 0

    assert path.read_text() == KEYS
    assert os.stat(path).st_mtime_ns == mtime_before


def test_noop_when_file_missing(tmp_path):
    mod = _load_script()
    assert mod.remove_api_key(tmp_path / "missing.toml", "smoke-cli") == 0


def test_main_returns_zero_and_rewrites_file(tmp_path, capsys):
    mod = _load_script()
    path = tmp_path / "api_keys.toml"
    path.write_text(KEYS)

    assert mod.main(["--id", "smoke-cli", "--path", str(path)]) == 0

    assert "smoke-cli" not in _ids(path)
    assert "removed 1" in capsys.readouterr().out
