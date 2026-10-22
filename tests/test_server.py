from types import SimpleNamespace

import pytest


def test_resolve_path_stays_inside_workspace(server, tmp_path):
    assert server._resolve_path(".") == tmp_path
    assert server._resolve_path("a/b.txt") == tmp_path / "a" / "b.txt"


def test_list_ollama_models_uses_configured_endpoint(server, monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, host):
            seen["host"] = host

        def list(self):
            return SimpleNamespace(models=[SimpleNamespace(
                model="qwen3:8b",
                details=SimpleNamespace(model_dump=lambda: {"family": "qwen3"}),
                capabilities=["completion", "tools"],
            )])

    monkeypatch.setenv("OLLAMA_MODEL", "https://tunnel.example/api/tags")
    monkeypatch.setattr(server, "Client", FakeClient)

    assert server.list_ollama_models() == [{
        "name": "qwen3:8b",
        "model": "qwen3:8b",
        "details": {"family": "qwen3"},
        "capabilities": ["completion", "tools"],
    }]
    assert seen["host"] == "https://tunnel.example"


@pytest.mark.parametrize(
    "bad", ["../outside.txt", "a/../../outside.txt", "/etc/passwd"]
)
def test_resolve_path_rejects_escapes(server, bad):
    with pytest.raises(ValueError):
        server._resolve_path(bad)


def test_resolve_path_accepts_docker_prefix(server, tmp_path):
    assert server._resolve_path("/workspace") == tmp_path
    assert server._resolve_path("/workspace/x.py") == tmp_path / "x.py"


def test_write_then_read_roundtrip(server, tmp_path):
    assert server.workspace_write_file("pkg/mod.py", "x = 1\n") == "Wrote pkg/mod.py"
    assert (tmp_path / "pkg" / "mod.py").read_text() == "x = 1\n"
    assert server.workspace_read_file("pkg/mod.py") == "x = 1\n"


def test_read_missing_file_raises(server):
    with pytest.raises(FileNotFoundError):
        server.workspace_read_file("nope.txt")


def test_write_outside_workspace_is_blocked(server, tmp_path):
    with pytest.raises(ValueError):
        server.workspace_write_file("../evil.txt", "x")
    assert not (tmp_path.parent / "evil.txt").exists()


def test_list_files_skips_generated_dirs_and_respects_depth(server, tmp_path):
    for rel in ["a.py", "src/b.py", ".git/config", ".venv/x.py",
                "node_modules/m.js", "src/__pycache__/c.pyc", "d1/d2/d3/d4/deep.py"]:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    assert server.workspace_list_files(".", max_depth=1) == ["a.py", "src/b.py"]
    assert "d1/d2/d3/d4/deep.py" in server.workspace_list_files(".", max_depth=10)


def test_list_files_rejects_non_directory(server, tmp_path):
    (tmp_path / "f.txt").write_text("x")
    with pytest.raises(NotADirectoryError):
        server.workspace_list_files("f.txt")


def test_search_is_case_insensitive_and_skips_binary_and_ignored(server, tmp_path):
    (tmp_path / "a.py").write_text("Hello World")
    (tmp_path / "b.py").write_text("nothing")
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00hello")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "c").write_text("hello")
    assert server.workspace_search_files("hello") == ["a.py"]


def test_run_command_splits_single_string(server, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        server.sandbox,
        "run",
        lambda cmd, timeout: seen.update(cmd=cmd, timeout=timeout) or {},
    )
    server.run_command(["python --version"], timeout=5)
    assert seen == {"cmd": ["python", "--version"], "timeout": 5}
    server.run_command(["python", "-c", "print(1)"])
    assert seen["cmd"] == ["python", "-c", "print(1)"]


def test_run_command_rejects_empty(server):
    with pytest.raises(ValueError):
        server.run_command([])
    with pytest.raises(ValueError):
        server.run_command([""])


def test_run_python_file_validation_and_command(server, tmp_path, monkeypatch):
    (tmp_path / "x.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "s.py").write_text("print(1)")
    with pytest.raises(FileNotFoundError):
        server.run_python_file("missing.py")
    with pytest.raises(ValueError):
        server.run_python_file("x.txt")
    seen = {}
    monkeypatch.setattr(server.sandbox, "run",
                        lambda cmd, timeout: seen.update(cmd=cmd) or {})
    server.run_python_file("sub/s.py", ["--flag"])
    assert seen["cmd"] == ["python", "/workspace/sub/s.py", "--flag"]


def test_install_tools_use_network_and_validate(server, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(server.sandbox, "run",
                        lambda cmd, timeout=60, network="none":
                        calls.append((cmd, network)) or {"returncode": 0})
    with pytest.raises(ValueError):
        server.install_python_packages([])
    server.install_python_packages(["requests"])
    assert calls[0] == (["python", "-m", "ensurepip", "--upgrade"], "none")
    assert calls[-1] == (["python", "-m", "pip", "install", "requests"], "bridge")

    assert server.install_project_dependencies()["returncode"] == 1  # nothing declared
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "requirements.txt").write_text("requests\n")
    out = server.install_project_dependencies()
    assert out["returncode"] == 0 and len(out["steps"]) == 2
    assert calls[-3] == (["python", "-m", "ensurepip", "--upgrade"], "none")
    assert all(net == "bridge" for _, net in calls[-2:])
