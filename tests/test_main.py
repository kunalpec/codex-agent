import asyncio
from contextlib import asynccontextmanager

from openrouter.errors import OpenRouterError

import main


def test_openrouter_error_is_reported_and_prompt_continues(
    tmp_path, monkeypatch, capsys
):
    prompts = iter(["first task", "exit"])
    monkeypatch.setattr("builtins.input", lambda: next(prompts))
    monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "provider/model:free")

    class FakeOpenRouterError(OpenRouterError):
        def __init__(self):
            self.message = "provider busy"
            Exception.__init__(self, self.message)

    class FakeMcpClient:
        async def list_tools(self):
            return []

    @asynccontextmanager
    async def fake_create_mcp_client(_workspace):
        yield FakeMcpClient()

    class FakeCheckpointer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class FakeSaver:
        @staticmethod
        def from_conn_string(_path):
            return FakeCheckpointer()

    class FakeCodingAgent:
        def __init__(self, *_args):
            pass

        async def stream_events(self, **_kwargs):
            raise FakeOpenRouterError()
            yield  # Make this an async generator.

    monkeypatch.setattr(main, "create_mcp_client", fake_create_mcp_client)
    monkeypatch.setattr(main, "AsyncSqliteSaver", FakeSaver)
    monkeypatch.setattr(main, "CodingAgent", FakeCodingAgent)

    monkeypatch.chdir(tmp_path)
    asyncio.run(main.main(tmp_path))

    output = capsys.readouterr().out
    assert "Model request failed" in output and "provider busy" in output
    assert "Free models are heavily rate limited. Try a paid model." in output
    assert "Goodbye." in output


def test_stubbed_stream_renders_complete_turn(tmp_path, monkeypatch, capsys):
    prompts = iter(["inspect files", "quit"])
    monkeypatch.setattr("builtins.input", lambda: next(prompts))

    class FakeMcpClient:
        async def list_tools(self):
            return []

    @asynccontextmanager
    async def fake_create_mcp_client(_workspace):
        yield FakeMcpClient()

    class FakeCheckpointer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class FakeSaver:
        @staticmethod
        def from_conn_string(_path):
            return FakeCheckpointer()

    class FakeCodingAgent:
        def __init__(self, *_args):
            pass

        async def stream_events(self, **_kwargs):
            yield {"kind": "text", "text": "I checked the files."}
            yield {"kind": "tool_start", "id": "t1", "name": "run_command",
                   "args": {"command": ["python", "-m", "pytest"]}}
            yield {"kind": "tool_result", "id": "t1", "name": "run_command",
                   "content": '{"returncode": 0, "stdout": "3 passed"}',
                   "is_error": False}
            yield {"kind": "text", "text": " All checks passed."}
            yield {"kind": "done"}

    monkeypatch.setattr(main, "create_mcp_client", fake_create_mcp_client)
    monkeypatch.setattr(main, "AsyncSqliteSaver", FakeSaver)
    monkeypatch.setattr(main, "CodingAgent", FakeCodingAgent)

    monkeypatch.chdir(tmp_path)
    asyncio.run(main.main(tmp_path))

    output = capsys.readouterr().out
    assert "I checked the files." in output
    assert "run_command" in output
    assert "All checks passed." in output
    assert "1 tool call, 0 errors" in output
