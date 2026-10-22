import os
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable

from ollama import Client
from rich import box
from rich.table import Table
from rich.text import Text

from ollama_config import ollama_base_url

from ui.renderer import render_banner, render_help, render_tools
from ui.theme import SUCCESS, WARNING


def _model_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else {}


def _model_info_value(model_info: dict[str, Any], key: str) -> Any:
    if key in model_info:
        return model_info[key]
    return next(
        (value for name, value in model_info.items() if name.endswith(f".{key}")),
        None,
    )


def _format_model_size(size: int | None) -> str | None:
    if size is None:
        return None
    return f"{int(size) / (1024 ** 3):.1f} GiB"


def parse_command(value: str):
    """Return (command, arguments), or None when input is a task."""
    value = value.strip()
    if value.lower() in {"exit", "quit"}:
        return value.lower(), ""
    if not value.startswith("/"):
        return None
    bits = value[1:].split(maxsplit=1)
    return bits[0].lower(), bits[1] if len(bits) > 1 else ""


@dataclass
class CommandContext:
    console: Any
    workspace: Path
    model: str
    thread_id: str
    thread_number: int
    docker_status: object
    tools: list[Any]
    check_docker: Callable[[], Any]
    select_model: Callable[[str, str], None] | None = None


async def handle_command(raw: str, context: CommandContext) -> str:
    """Handle a slash command. Returns `exit` or `continue`."""
    parsed = parse_command(raw)
    if parsed is None:
        return "task"
    command, args = parsed
    console = context.console
    if command in {"exit", "quit"}:
        console.print("Goodbye.", style="grey50")
        return "exit"
    if command == "help":
        render_help(console)
    elif command == "clear":
        console.clear()
        render_banner(console, context.workspace, context.model,
                      context.thread_id, context.docker_status)
    elif command == "new":
        context.thread_number += 1
        context.thread_id = f"{context.workspace}#{context.thread_number}"
        console.print(
            f"Started a fresh conversation: {context.thread_id}", style=SUCCESS
        )
    elif command == "tools":
        render_tools(console, context.tools)
    elif command == "model":
        if not args:
            console.print(
                Text(
                    f"Model: {context.model}\n"
                    "Use /model ollama:<name> to switch to an Ollama model."
                )
            )
        elif args.startswith("ollama:"):
            name = args[len("ollama:"):].strip()
            if context.select_model:
                context.select_model("ollama", name)
                context.model = f"ollama:{name}"
                console.print(Text(f"Switched to {context.model}"), style=SUCCESS)
        else:
            if context.select_model:
                context.select_model("openrouter", args.strip())
                context.model = args.strip()
                console.print(Text(f"Switched to {context.model}"), style=SUCCESS)
    elif command == "models":
        try:
            client = Client(host=ollama_base_url())
            models = client.list().models
            if not models:
                console.print("No Ollama models are available.", style=WARNING)
            else:
                console.print("Ollama models (switch with /model ollama:<name>):")
                table = Table(box=box.SIMPLE_HEAD, header_style="bold")
                table.add_column("Model")
                table.add_column("Details", overflow="fold")
                for model in models:
                    name = (
                        getattr(model, "model", None)
                        or getattr(model, "name", "unknown")
                    )
                    listed_details = _model_mapping(getattr(model, "details", None))
                    try:
                        details = client.show(name)
                        shown = _model_mapping(details)
                    except Exception:
                        shown = {}
                    model_info = (
                        shown.get("model_info", shown.get("modelinfo", {})) or {}
                    )
                    model_info = _model_mapping(model_info)
                    model_details = dict(listed_details)
                    model_details.update(_model_mapping(shown.get("details")))
                    capabilities = shown.get("capabilities") or []
                    if isinstance(capabilities, str):
                        capabilities = [capabilities]
                    fields = [
                        ("Family", model_details.get("family")),
                        ("Parameters", model_details.get("parameter_size")),
                        ("Quantization", model_details.get("quantization_level")),
                        ("Context", model_details.get("context_length") or
                         _model_info_value(model_info, "context_length")),
                        ("Embedding", model_details.get("embedding_length") or
                         _model_info_value(model_info, "embedding_length")),
                        ("Format", model_details.get("format")),
                        ("Runner", model_details.get("runner")),
                        ("Size", _format_model_size(getattr(model, "size", None))),
                        ("Features", ", ".join(str(feature) for feature in capabilities)
                         or "Not reported"),
                    ]
                    summary = "  ·  ".join(
                        f"{label}: {value:,}" if label in {"Context", "Embedding"}
                        and isinstance(value, int) else f"{label}: {value}"
                        for label, value in fields if value is not None
                    )
                    table.add_row(str(name), summary or "Details not reported")
                console.print(table)
        except Exception as error:
            console.print(Text(f"Could not load Ollama models: {error}"), style=WARNING)
    elif command == "workspace":
        console.print(Text(f"Workspace: {context.workspace}"))
        entries = sorted(
            islice(context.workspace.iterdir(), 31),
            key=lambda path: path.name.casefold(),
        )
        has_more = len(entries) > 30
        entries = entries[:30]
        for item in entries:
            console.print(Text(f"{'[dir]' if item.is_dir() else '[file]'} {item.name}"))
        if has_more:
            console.print("... more entries", style="grey50")
    elif command == "docker":
        context.docker_status = await context.check_docker()
        render_banner(console, context.workspace, context.model,
                      context.thread_id, context.docker_status)
    else:
        console.print("Unknown command. Type /help.", style=WARNING)
    return "continue"
