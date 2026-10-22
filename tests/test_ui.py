import asyncio
import io
import json

from rich.console import Console

from ui.commands import CommandContext, handle_command, parse_command
from ui.renderer import render_banner, render_error, render_help, render_tool_card


def make_console():
    output = io.StringIO()
    return Console(file=output, force_terminal=False, width=100, record=True), output


def output(console):
    return console.export_text()


def test_banner_contains_workspace_and_model():
    console, _ = make_console()
    render_banner(console, r"C:\work\repo", "vendor/model", "thread-1",
                  {"available": True, "version": "27.0"})
    rendered = output(console)
    assert "C:\\work\\repo" in rendered
    assert "vendor/model" in rendered
    assert "Docker ready" in rendered


def test_read_file_card_uses_syntax_and_bounds_lines():
    console, _ = make_console()
    render_tool_card(console, "workspace_read_file", {"path": "app.py"},
                     "def main():\n    return 1\n" + "x = 2\n" * 30)
    rendered = output(console)
    assert "workspace_read_file" in rendered
    assert "def main" in rendered
    assert "more lines" in rendered


def test_write_and_list_cards():
    console, _ = make_console()
    render_tool_card(
        console,
        "workspace_write_file",
        {"path": "a.py", "content": "a\nb\n"},
        "Wrote a.py",
    )
    render_tool_card(
        console,
        "workspace_list_files",
        {},
        json.dumps([f"file-{i}" for i in range(20)]),
    )
    rendered = output(console)
    assert "Wrote a.py (2 lines)" in rendered
    assert "... 5 more" in rendered


def test_command_cards_show_exit_codes():
    console, _ = make_console()
    for code in (0, 1, 124, 127):
        render_tool_card(console, "run_command", {"command": ["cmd"]},
                         f'{{"returncode": {code}, "stdout": "ok", "stderr": "bad"}}')
    rendered = output(console)
    for code in (0, 1, 124, 127):
        assert f"Exit code: {code}" in rendered


def test_tool_error_and_markup_injection_are_safe():
    console, _ = make_console()
    render_tool_card(
        console, "workspace_read_file", {}, "[bold red]x[/]", is_error=True
    )
    rendered = output(console)
    assert "[bold red]x[/]" in rendered
    assert "workspace_read_file" in rendered


def test_tool_stdout_is_truncated():
    console, _ = make_console()
    lines = "\n".join(f"line {i}" for i in range(500))
    render_tool_card(console, "run_command", {},
                     json.dumps({"returncode": 0, "stdout": lines}))
    rendered = output(console)
    assert "... 470 more lines" in rendered
    assert "line 499" not in rendered


def test_help_and_error_panels_render():
    console, _ = make_console()
    render_help(console)
    render_error(console, "Model request failed", "provider busy", "Try again")
    rendered = output(console)
    assert "/help" in rendered and "/new" in rendered
    assert "Model request failed" in rendered and "provider busy" in rendered


def test_slash_command_parsing_and_new_thread(tmp_path):
    assert parse_command("/help") == ("help", "")
    assert parse_command("exit") == ("exit", "")
    assert parse_command("quit") == ("quit", "")
    assert parse_command("work on this") is None
    console, _ = make_console()
    context = CommandContext(console, tmp_path, "model", str(tmp_path), 0,
                             {}, [], lambda: None)

    async def go():
        assert await handle_command("/new", context) == "continue"
        assert await handle_command("/unknown", context) == "continue"
        assert await handle_command("/exit", context) == "exit"

    asyncio.run(go())
    assert context.thread_id == f"{tmp_path}#1"


def test_non_tty_rendering_does_not_raise():
    console, _ = make_console()
    render_banner(console, ".", "model", "thread", {})
    render_tool_card(console, "run_command", {},
                     '{"returncode": 0, "stdout": "ok"}')


def test_narrow_terminal_rendering_does_not_raise():
    console = Console(file=io.StringIO(), force_terminal=False, width=60)
    render_banner(console, r"C:\work\a-long-workspace-path", "vendor/model",
                  "thread", {"available": False})
    render_tool_card(console, "workspace_read_file", {"path": "src/app.py"},
                     "def run():\n    return True\n")
