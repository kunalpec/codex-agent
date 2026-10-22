import argparse
import asyncio
import json
import os
import re
import warnings
from pathlib import Path
from runpy import run_path

from langchain_core._api.beta_decorator import LangChainBetaWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainBetaWarning,
    message="`langchain.mcp` is in beta.*",
)

from agent import CodingAgent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openrouter.errors import OpenRouterError
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from ui.activity import ActivityStatus
from ui.commands import CommandContext, handle_command
from ui.console import create_console
from ui.renderer import (
    render_banner,
    render_error,
    render_footer,
    render_rule,
    render_tool_card,
)
from ollama_config import ollama_chat_model
from ui.theme import MUTED, WARNING, icon


PROJECT_ROOT = Path(__file__).resolve().parent

# The local folder is named `mcp`, which is also the MCP SDK package name.
create_mcp_client = run_path(
    str(PROJECT_ROOT / "mcp" / "client.py")
)["create_mcp_client"]


def _tool_by_name(tools: list, name: str):
    return next((tool for tool in tools if getattr(tool, "name", None) == name), None)


def _tool_activity_label(name: str) -> str:
    labels = {
        "workspace_list_files": "Exploring workspace",
        "workspace_search_files": "Searching workspace",
        "workspace_read_file": "Reading project files",
        "workspace_write_file": "Updating project files",
        "workspace_rename_file": "Organizing project files",
        "workspace_rename_folder": "Organizing project folders",
        "check_docker": "Checking Docker",
        "check_sandbox_image": "Checking sandbox image",
        "build_sandbox_image": "Building sandbox",
        "run_command": "Running command",
        "run_python_file": "Running Python",
        "run_project_file": "Running project file",
        "run_project_command": "Running project command",
        "run_project_check": "Checking project",
        "install_packages": "Installing packages",
        "install_python_packages": "Installing packages",
        "install_project_dependencies": "Installing project dependencies",
        "web_search": "Searching the web",
        "web_extract": "Reading documentation",
        "list_ollama_models": "Loading available models",
    }
    action = labels.get(name, f"Using {name}")
    return f"• Executing: {action}"


def _response_activity(text: str, subagent: str | None) -> str:
    """Describe activity visible in the response stream and agent events."""
    if subagent:
        return f"Using agent: {subagent}"
    if not text.strip():
        return "Thinking"

    response = text.lstrip()
    starts_with_plan = re.match(
        r"(?:#{1,6}\s*)?(?:\*\*)?plan\b", response, flags=re.IGNORECASE
    )
    next_section = re.search(r"\n\s*(?:#{1,6}\s+|\*\*)", response)
    if starts_with_plan and not next_section:
        return "Planning"
    return "Responding"


async def _call_tool(tool):
    if tool is None:
        return {"available": False, "error": "check_docker tool unavailable"}
    result = await tool.ainvoke({})
    if hasattr(result, "content"):
        result = result.content
    if isinstance(result, list):
        result = "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in result
        )
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            pass
    return result


async def _read_task(console):
    if console.is_terminal:
        # prompt_toolkit enables bracketed paste and avoids the Windows
        # console's short line-input limit for large prompts.
        from prompt_toolkit import PromptSession

        session = PromptSession()
        line = await asyncio.to_thread(session.prompt, "❯ ")
        return line.strip()

    parts = []
    prompt = f"[prompt]{icon(console, 'run')}[/prompt] "
    while True:
        line = await asyncio.to_thread(console.input, prompt)
        if line.endswith("\\"):
            parts.append(line[:-1])
            prompt = "[muted]…[/muted] "
            continue
        parts.append(line)
        return "\n".join(parts).strip()


async def _run_turn(
    console,
    coding_agent,
    task: str,
    thread_id: str,
    model: str,
    debug: bool,
    fallback=None,
    allow_fallback: bool = True,
) -> None:
    started = asyncio.get_running_loop().time()
    tool_starts = {}
    tool_args = {}
    tool_calls = 0
    errors = 0
    buffer = ""
    subagent = None
    live = None
    activity = "Thinking"
    terminal = console.is_terminal

    def start_live():
        nonlocal live
        if terminal and live is None:
            live = Live(
                ActivityStatus(activity),
                console=console,
                auto_refresh=False,
                vertical_overflow="visible",
            )
            live.start(refresh=True)

    def stop_live():
        nonlocal live
        if live is not None:
            live.stop()
            live = None

    def show_markdown():
        nonlocal live
        if terminal and live is not None:
            label = f"Agent · subagent: {subagent}" if subagent else "Agent"
            live.update(
                Group(
                    Text(label, style="bold cyan"),
                    ActivityStatus(activity),
                    Markdown(buffer),
                ),
                refresh=True,
            )

    def update_activity(label: str) -> None:
        nonlocal activity
        if activity == label:
            return
        activity = label
        if terminal and live is not None:
            if buffer:
                show_markdown()
            else:
                live.update(ActivityStatus(activity), refresh=True)
        elif not terminal:
            console.print(ActivityStatus(activity))

    start_live()
    if not terminal:
        console.print(ActivityStatus(activity))
    try:
        async for event in coding_agent.stream_events(task=task, thread_id=thread_id):
            kind = event.get("kind")
            if kind == "text":
                text = event.get("text", "")
                subagent = event.get("subagent")
                if not text:
                    continue
                buffer += text
                update_activity(_response_activity(buffer, subagent))
                if not terminal:
                    console.print(Text(text), end="")
                else:
                    show_markdown()
            elif kind == "tool_start":
                tool_calls += 1
                tool_starts[event.get("id", "")] = asyncio.get_running_loop().time()
                tool_args[event.get("id", "")] = event.get("args", {})
                if live is not None:
                    stop_live()
                subagent = event.get("subagent")
                activity = _tool_activity_label(event.get("name", "tool"))
                suffix = f" · subagent: {subagent}" if subagent else ""
                console.print(
                    Text(
                        f"{icon(console, 'run')} {event.get('name', 'tool')}  "
                        f"{suffix} {str(event.get('args', {}))[:120]}",
                        style=MUTED,
                    )
                )
                buffer = ""
                start_live()
            elif kind == "tool_result":
                stop_live()
                subagent = event.get("subagent")
                duration = asyncio.get_running_loop().time() - tool_starts.pop(
                    event.get("id", ""), asyncio.get_running_loop().time())
                failed = bool(event.get("is_error"))
                errors += int(failed)
                args = tool_args.pop(event.get("id", ""), {})
                if terminal:
                    render_tool_card(console, event.get("name", "tool"), args,
                                     event.get("content", ""), failed, duration)
                else:
                    console.print(
                        Text(
                            f"Tool {event.get('name', 'tool')}: "
                            f"{'error' if failed else 'complete'} ({duration:.2f}s)"
                        )
                    )
                update_activity(
                    f"Using agent: {subagent}" if subagent else "Thinking"
                )
                start_live()
            elif kind == "done":
                break
        stop_live()
        if not terminal and buffer:
            console.print()
        render_footer(console, tool_calls, errors,
                      asyncio.get_running_loop().time() - started)
    except OpenRouterError as error:
        stop_live()
        if allow_fallback and fallback and not model.startswith("ollama:"):
            console.print("OpenRouter is unavailable; retrying this task with Ollama.",
                          style=WARNING)
            try:
                local_agent, local_model = fallback()
                await _run_turn(console, local_agent, task,
                                thread_id + "#ollama-fallback", local_model,
                                debug, fallback, allow_fallback=False)
                return
            except Exception as fallback_error:
                render_error(console, "Ollama fallback failed", fallback_error,
                             "Check OLLAMA_BASE_URL and OLLAMA_CHAT_MODEL in .env.")
                return
        suggestion = (
            "Rate limited or provider busy. Wait a moment or change "
            "OPENROUTER_MODEL in .env."
        )
        if model.endswith(":free"):
            suggestion += "\nFree models are heavily rate limited."
        render_error(console, "Model request failed", error, suggestion)
    except KeyboardInterrupt:
        stop_live()
        console.print("Cancelled.", style=WARNING)
    except asyncio.CancelledError:
        stop_live()
        console.print("Cancelled.", style=WARNING)
    except Exception as error:
        stop_live()
        render_error(console, "Unexpected error", f"{type(error).__name__}: {error}")
        if debug:
            console.print_exception(show_locals=False)


async def main(workspace: Path | None = None, debug: bool = False) -> None:
    workspace = (workspace or Path.cwd()).expanduser().resolve()
    console = create_console()
    if not workspace.is_dir():
        render_error(console, "Workspace unavailable",
                     f"Workspace folder does not exist: {workspace}")
        return
    os.chdir(workspace)
    os.environ["MCP_WORKSPACE_ROOT"] = str(workspace)
    checkpoint_file = workspace / "checkpoints.sqlite"
    provider = os.environ.get("MODEL_PROVIDER", "ollama").lower()
    model = os.environ.get(
        "OLLAMA_MODEL" if provider == "ollama" else "OPENROUTER_MODEL",
        "qwen3:8b" if provider == "ollama" else "openai/gpt-4o",
    )
    if provider == "ollama" and model.startswith(("http://", "https://")):
        model = ollama_chat_model()
    thread_id = str(workspace)
    thread_number = 0

    try:
        async with create_mcp_client(workspace) as mcp_client:
            tools = await mcp_client.list_tools()
            if _tool_by_name(tools, "workspace_create_csv") is None:
                console.print(
                    "CSV export tool is unavailable. Restart or update codex-agent.",
                    style=WARNING,
                )
            check_docker_tool = _tool_by_name(tools, "check_docker")
            try:
                docker_status = await _call_tool(check_docker_tool)
            except Exception as error:
                docker_status = {"available": False, "error": str(error)}
            async with AsyncSqliteSaver.from_conn_string(
                str(checkpoint_file)
            ) as checkpointer:
                coding_agent = CodingAgent(checkpointer, tools, workspace)

                def select_model(provider, model_name):
                    nonlocal coding_agent
                    os.environ["MODEL_PROVIDER"] = provider
                    model_key = (
                        "OLLAMA_MODEL" if provider == "ollama" else "OPENROUTER_MODEL"
                    )
                    os.environ[model_key] = model_name
                    coding_agent = CodingAgent(
                        checkpointer, tools, workspace, provider, model_name
                    )

                def fallback_to_ollama():
                    local_model = ollama_chat_model()
                    local_agent = CodingAgent(
                        checkpointer, tools, workspace, "ollama", local_model
                    )
                    return local_agent, f"ollama:{local_model}"

                context = CommandContext(
                    console,
                    workspace,
                    model,
                    thread_id,
                    thread_number,
                    docker_status,
                    tools,
                    lambda: _call_tool(check_docker_tool),
                    select_model,
                )
                context.model = f"ollama:{model}" if provider == "ollama" else model
                model = context.model
                render_banner(
                    console, workspace, model, context.thread_id, docker_status
                )
                if model.endswith(":free"):
                    console.print(
                        "Free models are heavily rate limited. Try a paid model.",
                        style=WARNING,
                    )
                while True:
                    try:
                        task = await _read_task(console)
                    except (EOFError, KeyboardInterrupt):
                        console.print("\nGoodbye.", style=MUTED)
                        return
                    if not task:
                        console.print("Please enter a coding task.", style=WARNING)
                        continue
                    command = await handle_command(task, context)
                    thread_id = context.thread_id
                    if command == "exit":
                        return
                    if context.model != model:
                        model = context.model
                    if command == "continue":
                        render_rule(console)
                        continue
                    render_rule(console)
                    await _run_turn(console, coding_agent, task, thread_id, model,
                                    debug, fallback_to_ollama)
    except (EOFError, KeyboardInterrupt):
        console.print("\nGoodbye.", style=MUTED)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Start the terminal coding agent.")
    parser.add_argument(
        "--workspace", "-w", type=Path,
        default=Path(os.environ.get("MCP_WORKSPACE_ROOT", Path.cwd())),
        help="Folder where the agent should read and write project files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show tracebacks for unexpected errors.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.workspace, args.debug))


if __name__ == "__main__":
    cli()
