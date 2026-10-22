import subprocess
from pathlib import Path

import pytest

from sandbox.runner import IMAGE_NAME, PACKAGE_VOLUME, DockerSandbox


class Done:
    def __init__(self, rc=0, out="o", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_run_builds_limited_offline_command(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured.update(cmd=cmd, kw=kw)
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Keep command construction independent from host-specific Docker discovery.
    monkeypatch.setattr(DockerSandbox, "_docker", staticmethod(lambda: "docker"))
    result = DockerSandbox(tmp_path).run(["python", "--version"])
    cmd = captured["cmd"]
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert cmd[cmd.index("--memory") + 1] == "512m"
    assert cmd[cmd.index("--cpus") + 1] == "1"
    assert cmd[cmd.index("--pids-limit") + 1] == "100"
    assert f"{tmp_path}:/workspace" in cmd
    assert f"{PACKAGE_VOLUME}:/opt/venv" in cmd
    assert cmd[-3:] == [IMAGE_NAME, "python", "--version"]
    assert result["returncode"] == 0 and result["stdout"] == "o"


def test_run_validates_input(tmp_path):
    box = DockerSandbox(tmp_path)
    with pytest.raises(ValueError):
        box.run([])
    with pytest.raises(ValueError):
        box.run(["ok", 1])
    with pytest.raises(ValueError):
        box.run(["ok"], network="host")


@pytest.mark.parametrize("given,expected", [(0, 1), (5000, 900), (30, 30)])
def test_timeout_is_clamped(tmp_path, monkeypatch, given, expected):
    seen = {}
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: seen.update(t=kw["timeout"]) or Done())
    DockerSandbox(tmp_path).run(["x"], timeout=given)
    assert seen["t"] == expected


def test_missing_docker_returns_127(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(subprocess, "run", boom)
    assert DockerSandbox(tmp_path).run(["x"])["returncode"] == 127


def test_timeout_returns_124_and_decodes_bytes(tmp_path, monkeypatch):
    def slow(cmd, **k):
        raise subprocess.TimeoutExpired(cmd, 1, output=b"partial", stderr=b"err")
    monkeypatch.setattr(subprocess, "run", slow)
    res = DockerSandbox(tmp_path).run(["x"], timeout=1)
    assert res["returncode"] == 124
    assert res["stdout"] == "partial" and "timed out" in res["stderr"]


def test_check_docker_states(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done(0, "27.0\n"))
    assert DockerSandbox(tmp_path).check_docker()["available"] is True
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done(1, "", "daemon down"))
    assert DockerSandbox(tmp_path).check_docker()["available"] is False

    def missing(*a, **k):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(subprocess, "run", missing)
    assert DockerSandbox(tmp_path).check_docker()["available"] is False


def test_build_image_uses_sandbox_dockerfile(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(DockerSandbox, "_docker", staticmethod(lambda: "docker"))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: seen.update(cmd=cmd) or Done(),
    )
    DockerSandbox(tmp_path).build_image()
    cmd = seen["cmd"]
    assert cmd[:4] == ["docker", "build", "--tag", IMAGE_NAME]
    assert Path(cmd[cmd.index("--file") + 1]).name == "Dockerfile"
