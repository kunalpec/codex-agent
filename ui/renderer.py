import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ui.theme import CODE, ERROR, MUTED, SUCCESS, WARNING, icon

MAX_LINES = 30


def _lines(value: object, limit: int = MAX_LINES) -> str:
    value = str(value or "")
    if len(value) > 12000:
        value = value[:12000] + "\n... output truncated by character limit"
    lines = [line[:1200] + ("…" if len(line) > 1200 else "")
             for line in value.splitlines()]
    kept = lines[:limit]
    if len(lines) > limit:
        kept.append(f"... {len(lines) - limit} more lines")
    return "\n".join(kept)


def _decoded(content: str) -> object:
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content


def _short(value: object, limit: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _arg_summary(args: object) -> str:
    if not isinstance(args, dict):
        return _short(args)
    return "  ".join(f"{key}={_short(value)}" for key, value in args.items())


def render_banner(console, workspace, model, thread_id, docker_status) -> None:
    try:
        app_version = version("project2")
    except PackageNotFoundError:
        app_version = "dev"
    if isinstance(docker_status, dict) and docker_status.get("available"):
        docker = Text(
            f"Docker ready (v{docker_status.get('version', 'unknown')})",
            style=SUCCESS,
        )
    else:
        error = (docker_status.get("error") if isinstance(docker_status, dict)
                 else None)
        detail = f" — {error}" if error else ""
        docker = Text(
            f"Docker unavailable{detail}; start Docker Desktop", style=WARNING
        )
    body = Group(
        Text(f"Coding agent  v{app_version}", style="bold bright_white"),
        Text(f"Workspace  {workspace}"),
        Text(f"Model      {model}"),
        Text(f"Thread     {thread_id}"),
        docker,
    )
    console.print(
        Panel(body, title="codex-agent", border_style="cyan", box=box.ROUNDED)
    )
    console.print("Type a task, /help for commands, Ctrl+C to cancel.", style=MUTED)


def render_tool_card(
    console, name, args, content, is_error: bool = False, elapsed: float | None = None
) -> None:
    parsed = _decoded(content)
    heading = Text()
    heading.append(f"{icon(console, 'error' if is_error else 'run')} ",
                   style=ERROR if is_error else SUCCESS)
    heading.append(name, style="bold")
    summary = _arg_summary(args)
    if summary:
        heading.append("  " + summary, style=MUTED)
    parts = [heading]

    if is_error:
        parts.append(Text(_lines(content, 15), style=ERROR))
    elif name == "workspace_read_file":
        source = parsed if isinstance(parsed, str) else json.dumps(parsed, indent=2)
        source = source[:6000]
        lines = source.splitlines()
        shown = "\n".join(lines[:25])
        lexer = Path(str((args or {}).get("path", ""))).suffix.lstrip(".") or "text"
        parts.append(
            Syntax(shown, lexer, theme=CODE, line_numbers=True, word_wrap=True)
        )
        if len(lines) > 25:
            parts.append(Text(f"... {len(lines) - 25} more lines", style=MUTED))
    elif name == "workspace_write_file":
        path = _short((args or {}).get("path", "file"), 100)
        line_count = len(str((args or {}).get("content", "")).splitlines())
        parts.append(Text(f"Wrote {path} ({line_count} lines)", style=SUCCESS))
    elif name == "workspace_create_csv" and isinstance(parsed, dict):
        headers = parsed.get("headers", [])
        rows = parsed.get("preview", [])
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
        for header in headers:
            table.add_column(_short(header, 60), overflow="fold")
        for row in rows:
            table.add_row(*[_short(value, 100) for value in row])
        parts.append(Text(f"Created {parsed.get('path', 'CSV file')}", style=SUCCESS))
        parts.append(table)
        row_count = parsed.get("row_count", len(rows))
        if row_count > len(rows):
            parts.append(Text(f"Showing {len(rows)} of {row_count} rows", style=MUTED))
    elif name in {"workspace_list_files", "workspace_search_files"}:
        entries = parsed if isinstance(parsed, list) else str(parsed).splitlines()
        parts.extend(
            Text(f"• {_short(item, 180)}", style=MUTED) for item in entries[:15]
        )
        if len(entries) > 15:
            parts.append(Text(f"... {len(entries) - 15} more", style=MUTED))
    elif name in {
        "run_command",
        "run_python_file",
        "run_project_file",
        "run_project_command",
        "run_project_check",
        "install_packages",
        "install_python_packages",
        "install_project_dependencies",
    }:
        data = parsed if isinstance(parsed, dict) else {}
        if "steps" in data:
            data = data["steps"][-1] if data["steps"] else data
        code = data.get("returncode", data.get("exit_code", "?"))
        color = SUCCESS if code == 0 else WARNING if code in (124, 127) else ERROR
        parts.append(Text(f"Exit code: {code}", style=color))
        if data.get("stdout"):
            parts.extend(
                [
                    Text("stdout", style="bold"),
                    Text(_lines(data["stdout"]), style=MUTED),
                ]
            )
        if data.get("stderr"):
            parts.extend(
                [
                    Text("stderr", style="bold red"),
                    Text(_lines(data["stderr"]), style=ERROR),
                ]
            )
    elif name in {"check_docker", "check_sandbox_image", "build_sandbox_image"}:
        good = False
        if isinstance(parsed, dict):
            if "available" in parsed:
                good = bool(parsed["available"])
            elif "returncode" in parsed:
                good = parsed["returncode"] == 0
            else:
                good = bool(parsed.get("success", False))
        parts.append(Text(_lines(parsed, 5), style=SUCCESS if good else WARNING))
    elif name == "web_search" and isinstance(parsed, dict):
        results = parsed.get("results", [])
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Title", overflow="fold")
        table.add_column("URL", overflow="fold")
        table.add_column("Snippet", overflow="fold")
        for item in results[:10]:
            table.add_row(
                Text(_short(item.get("title", ""), 200)),
                Text(_short(item.get("url", ""), 200)),
                Text(_short(item.get("content", ""), 200)),
            )
        parts.append(table)
    elif name == "web_extract" and isinstance(parsed, dict):
        for item in parsed.get("results", [])[:10]:
            parts.append(Text(f"{_short(item.get('url', ''), 240)} "
                              f"({len(item.get('content') or '')} chars)", style=MUTED))
    else:
        parts.append(Text(_lines(parsed, 15), style=MUTED))

    if elapsed is not None:
        parts.append(Text(f"{elapsed:.2f}s", style=MUTED))
    console.print(
        Panel(
            Group(*parts),
            border_style=ERROR if is_error else "grey50",
            box=box.SQUARE,
            padding=(0, 1),
        )
    )


def render_error(console, title, message, suggestion=None, debug=False) -> None:
    body = Text(_lines(message, 15), style=ERROR)
    if suggestion:
        body.append("\n" + suggestion, style=MUTED)
    console.print(Panel(body, title=title, border_style=ERROR, box=box.SQUARE))


def render_help(console) -> None:
    table = Table(box=box.SIMPLE_HEAD, title="Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    for command, description in COMMANDS:
        table.add_row(command, description)
    console.print(table)


COMMANDS = [
    ("/help", "Show available commands"),
    ("/exit, /quit", "Exit the agent"),
    ("/clear", "Clear the screen and redraw the banner"),
    ("/new", "Start a fresh conversation thread"),
    ("/tools", "List available MCP tools"),
    ("/model [provider:model]", "Show or change the active model"),
    ("/models", "List models available from the Ollama endpoint"),
    ("/workspace", "Show the workspace and its files"),
    ("/docker", "Check Docker availability"),
]


def render_tools(console, tools) -> None:
    table = Table(box=box.SIMPLE_HEAD, title="Available tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for tool in tools:
        description = getattr(tool, "description", "") or ""
        table.add_row(Text(str(getattr(tool, "name", ""))),
                      Text(description.splitlines()[0][:160]))
    console.print(table)


def render_footer(console, tool_calls: int, errors: int, elapsed: float) -> None:
    call_label = "tool call" if tool_calls == 1 else "tool calls"
    error_label = "error" if errors == 1 else "errors"
    console.print(
        f"{tool_calls} {call_label}, {errors} {error_label}, "
        f"elapsed {elapsed:.1f}s",
        style=MUTED,
    )


def render_rule(console) -> None:
    console.print(Rule(style="grey35"))
