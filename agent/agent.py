import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter

from agent.state import CodingAgentState
from ollama_config import ollama_base_url, ollama_chat_model


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


CORE_SUBAGENTS = [
    {
        "name": "explorer",
        "description": "Search the codebase and trace execution paths.",
        "system_prompt": (
            "Explore the active workspace to find relevant files, symbols, "
            "dependencies, and execution paths. Support conclusions with file "
            "paths and code references. Do not edit files or run commands. "
            "Use workspace tools with workspace-relative paths."
        ),
    },
    {
        "name": "planner",
        "description": "Break requirements into clear implementation steps.",
        "system_prompt": (
            "Turn the request into a concise sequence of implementation steps. "
            "Call out constraints, dependencies, affected areas, and risks. "
            "Do not edit files or run commands. Base the plan on the request "
            "and available project context."
        ),
    },
    {
        "name": "researcher",
        "description": "Investigate libraries, APIs, and technical solutions.",
        "system_prompt": (
            "Research relevant libraries, APIs, and technical approaches. Use "
            "web_search and web_extract for external documentation when useful. "
            "Prefer official sources, include source URLs, and distinguish "
            "verified facts from recommendations. Do not edit project files."
        ),
    },
    {
        "name": "coder",
        "description": "Implement focused changes in project source code.",
        "system_prompt": (
            "Implement the assigned change in application code while preserving "
            "existing behavior outside its scope. Inspect surrounding code and "
            "project conventions first. Use workspace_read_file(path) and "
            "workspace_write_file(path, content) with relative paths; read the "
            "whole file before writing its complete updated contents. Do not "
            "run commands or use unsupported workspace tools."
        ),
    },
    {
        "name": "debugger",
        "description": "Find root causes and fix defects.",
        "system_prompt": (
            "Trace the reported failure to its root cause using code, errors, "
            "and logs. Check Docker and the sandbox image, building the image "
            "if it is missing. Install project dependencies when needed. Make "
            "the smallest reliable fix and verify it with a "
            "focused command in the Docker sandbox. Use run_project_file for "
            "supported source files and run_project_command for project "
            "commands. Read and write files using workspace tools and relative "
            "paths. Never run project code on the host."
        ),
    },
    {
        "name": "test_engineer",
        "description": "Write and execute focused tests for requested behavior.",
        "system_prompt": (
            "Inspect the existing test conventions, add or update focused tests "
            "for the assigned behavior. Check Docker and the sandbox image, "
            "build it if missing, install project dependencies when needed, "
            "and run tests in the Docker sandbox. "
            "Use run_project_check or run_project_command and report the exact "
            "command and observed result. Use workspace tools with relative "
            "paths and never claim a check you did not run."
        ),
    },
    {
        "name": "reviewer",
        "description": "Identify logic errors, regressions, and quality issues.",
        "system_prompt": (
            "Review the relevant changes for correctness, regressions, and "
            "maintainability. Report only actionable findings, ordered by "
            "severity, with file paths and line references when available. Do "
            "not edit files. If no issues are found, say so clearly."
        ),
    },
    {
        "name": "security_auditor",
        "description": "Find vulnerabilities and unsafe handling of data or secrets.",
        "system_prompt": (
            "Audit relevant changes for concrete security risks, including "
            "exposed secrets, unsafe path handling, injection, weak input "
            "validation, and insecure process or network use. Report evidence, "
            "severity, and file references. Do not edit files or speculate "
            "without evidence."
        ),
    },
    {
        "name": "performance_analyzer",
        "description": "Find bottlenecks and inefficient execution paths.",
        "system_prompt": (
            "Analyze the relevant execution paths for measurable bottlenecks, "
            "unnecessary work, and scaling risks. Support findings with code "
            "references or profiling evidence. Separate measured results from "
            "hypotheses. Do not edit files or claim performance improvements "
            "without measurement."
        ),
    },
    {
        "name": "verifier",
        "description": "Independently validate completed changes and claims.",
        "system_prompt": (
            "Independently inspect the completed change and verify its stated "
            "behavior using appropriate checks in the Docker sandbox. Check "
            "Docker and the sandbox image, building it if missing. Look for "
            "missed requirements and report exact commands and results. Do not "
            "edit files or rely only on another agent's summary."
        ),
    },
    {
        "name": "documenter",
        "description": "Update documentation and examples to match the code.",
        "system_prompt": (
            "Make focused updates to user or developer documentation and "
            "examples so they accurately reflect implemented behavior. Do not "
            "change application or test code. Use workspace_read_file(path) and "
            "workspace_write_file(path, content) with relative paths."
        ),
    },
    {
        "name": "shell_agent",
        "description": "Run project commands, builds, and development tools.",
        "system_prompt": (
            "Run commands only through the project's Docker sandbox tools. "
            "Check Docker and the sandbox image, building it when missing. Use "
            "run_command or run_project_command with argument lists, without "
            "shell operators, and report exact commands and output. Do not edit "
            "project files or execute code directly on the host."
        ),
    },
    {
        "name": "browser_agent",
        "description": "Inspect websites and assess browser workflows.",
        "system_prompt": (
            "Inspect public websites using web_search and web_extract when "
            "useful. Summarize page content and documented workflows with "
            "source URLs. This agent has no interactive browser-control tool: "
            "do not claim to click through, submit forms, or test a live browser "
            "workflow. State when interactive browser testing is unavailable."
        ),
    },
]


class AgentEvent(TypedDict):
    """A normalized event emitted while the agent runs."""

    kind: str
    text: NotRequired[str]
    id: NotRequired[str]
    name: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    content: NotRequired[str]
    is_error: NotRequired[bool]
    subagent: NotRequired[str]


class CodingAgent:
    """Create and run the coding agent with the supplied MCP tools."""

    def __init__(
        self,
        checkpointer: Any,
        tools: list[Any],
        workspace: Path,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.tools = tools
        self.workspace = workspace
        self.provider = provider or os.environ.get("MODEL_PROVIDER", "ollama").lower()
        self.model_name = model_name or os.environ.get(
            "OLLAMA_MODEL" if self.provider == "ollama" else "OPENROUTER_MODEL",
            "qwen3:8b" if self.provider == "ollama" else "openai/gpt-4o",
        )
        if self.provider == "ollama" and self.model_name.startswith(
            ("http://", "https://")
        ):
            self.model_name = ollama_chat_model()
        self._build()

    def _build(self) -> None:
        provider, model_name = self.provider, self.model_name
        middleware = []
        if provider == "ollama":
            primary_model = ChatOllama(
                base_url=ollama_base_url(),
                model=model_name,
                temperature=0.2,
            )
            fallback_model = ChatOpenRouter(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"),
                temperature=0.2,
                max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "8192")),
                max_retries=0,
            )
            model = primary_model
            middleware.append(ModelFallbackMiddleware(fallback_model))
        else:
            model = ChatOpenRouter(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                model=model_name,
                temperature=0.2,
                max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "8192")),
                max_retries=5,
            )
        self.agent = create_deep_agent(
            model=model,
            name="coding-agent",
            state_schema=CodingAgentState,
            checkpointer=self.checkpointer,
            backend=FilesystemBackend(root_dir=self.workspace),
            tools=self.tools,
            middleware=middleware,
            subagents=CORE_SUBAGENTS,
            system_prompt=(
                "You are a coding agent that carefully follows the user's request. "
                "Treat the full latest user message and its conversation context "
                "as the source of truth. Identify every requested outcome, "
                "constraint, and question before acting; do not silently drop "
                "requirements or substitute a different task. If the request is "
                "clear, proceed and complete it instead of only giving advice. "
                "If a detail is missing, make a sensible assumption unless it "
                "would materially change the result; ask a focused question only "
                "when you cannot safely or correctly proceed. For informational "
                "questions, answer the question directly and use repository tools "
                "only when project-specific facts are needed. In the final reply, "
                "address the original request, summarize the concrete result, "
                "and report checks only when they were actually run. Be clear and "
                "complete without adding unrelated explanation. "
                "For non-trivial requests, begin with "
                "a short user-facing plan under a 'Plan' heading with 2 to 5 "
                "bullet points before using tools. Before each substantial "
                "action, give one short progress bullet describing the action; "
                "then summarize completed work and checks in bullets. Share "
                "actionable status, not private chain-of-thought or hidden "
                "internal reasoning. The active workspace is the only "
                "folder where project files should be created or changed. Use "
                "workspace_list_files(directory, max_depth), "
                "workspace_glob_search(pattern, directory, max_results), "
                "workspace_search_code(query, directory, max_results), "
                "workspace_search_files(query, directory), and "
                "workspace_read_file(path) to inspect files. Use "
                "workspace_create_file(path, content) for a new file and "
                "workspace_write_file(path, content) to create or replace one. "
                "When the user requests a CSV export and the data is available, "
                "use workspace_create_csv(path, headers, rows); its result is "
                "shown as a table preview. This tool creates new CSV files and "
                "does not overwrite existing files. Never claim CSV files "
                "cannot be created: use this tool, and report its saved path or "
                "the actual error it returns. For any request to create, save, "
                "or export a CSV file, you MUST call workspace_create_csv before "
                "saying the file was saved. Do not substitute CSV text in the "
                "response for creating the requested file. If the tool is "
                "unavailable or fails, explain that accurately and never claim "
                "the file was created. "
                "Read the whole file before writing its complete updated "
                "contents. Call workspace_delete_file(path, confirm=true) only "
                "when the user explicitly requested deleting that file. Use "
                "workspace_rename_file(path, new_path) or "
                "workspace_rename_folder(path, new_path) to rename entries. "
                "Do not invent workspace_edit_file or unsupported arguments. "
                "Use list_ollama_models when asked to inspect available Ollama models. "
                "Workspace files are shared with VS Code and Docker; write code "
                "with workspace tools, then test it in Docker. Create a file "
                "before running it, confirm it exists with a workspace file "
                "tool, then call run_python_file or run_project_file with its "
                "project-relative path as appropriate. "
                "These tools take paths relative to the "
                "workspace, and '.' means its root. Their /workspace prefix is "
                "accepted as a convenience. Deep Agents' built-in filesystem "
                "tools use '/' as the workspace root; never use '/workspace' "
                "with those tools or a Windows absolute path. '/workspace' is "
                "only the path inside Docker commands. get_project_info "
                "returns '.' as the file-tool root; the CLI displays the real "
                "workspace path. Run "
                "code and commands only in the Docker "
                "sandbox. Before testing, call check_docker and "
                "check_sandbox_image; build the sandbox image if it is missing. "
                "Inspect project dependency "
                "files and call install_project_dependencies before running "
                "the project. If a script reports a missing Python package, "
                "install it with install_python_packages, then rerun the "
                "script. Package installs persist between calls. Installation "
                "uses network access; Docker commands use bridge networking. "
                "Use install_python_packages for Python packages or "
                "install_packages(packages, ecosystem, timeout) for packages "
                "from python, npm, go, cargo, or maven. Do not use apt-get or "
                "shell command strings for package installs. "
                "install_project_dependencies(timeout, directory) "
                "detects Python, Node, Maven/Gradle, Go, and Rust manifests. "
                "run_command executes arguments directly without a shell: do "
                "not include shell operators such as &&, pipes, or semicolons. "
                "Call each command separately. "
                "Delegate to explorer, planner, researcher, coder, debugger, "
                "test_engineer, reviewer, security_auditor, "
                "performance_analyzer, verifier, documenter, shell_agent, and "
                "browser_agent when useful. "
                "Report only work and test results confirmed by tool output. "
                "When asked to run a Python file, confirm its project-relative "
                "path and use run_python_file. For JavaScript, TypeScript, "
                "Java, Go, C, C++, or Rust use run_project_file(path, args, "
                "timeout). Use "
                "run_project_command(command, directory, timeout) for "
                "project-specific commands such as npm test, cargo test, or "
                "mvn test. If Docker is unavailable, explain "
                "the daemon error instead of retrying the same failed command. "
                "For run_command, pass the command as separate list items, "
                "such as [\"python\", \"--version\"], rather than one "
                "combined string. If Docker says the entire command string "
                "was not found, fix the argument list instead of claiming "
                "the executable is missing from the image. "
                "Use run_project_check(check, command, timeout, directory) for "
                "standard tests, lint, format checks, types, builds, and dependency "
                "security audits. Project checks use Docker bridge networking. Use "
                "git_status, git_diff(path, staged), git_log(max_entries), "
                "and git_branch for read-only repository inspection. Use "
                "inspect_environment for safe configuration status, "
                "inspect_dependencies to list sandbox packages, and "
                "check_dependencies to find dependency conflicts. "
                "Use web_search for documentation, library versions, and error "
                "messages you are unsure about, then web_extract to read a promising "
                "result in full. Interactive browser control, database access, "
                "deployment, and authenticated API tools are not configured. "
                "Docker code execution, dependency installation, and project "
                "checks use bridge networking. "
            ),
        )

    def select_model(self, provider: str, model_name: str) -> None:
        self.provider, self.model_name = provider, model_name
        self._build()

    async def run(self, task: str, thread_id: str = "default") -> dict:
        """Run one task and keep its conversation in the given thread."""
        return await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"configurable": {"thread_id": thread_id}},
        )

    async def stream(
        self,
        task: str,
        thread_id: str = "default",
    ) -> AsyncIterator[str]:
        """Yield response text as the model generates it."""
        config = {"configurable": {"thread_id": thread_id}}
        async for message, _metadata in self.agent.astream(
            {"messages": [{"role": "user", "content": task}]},
            config=config,
            stream_mode="messages",
        ):
            # Tool results also arrive in this stream; only show model text.
            if not isinstance(message, AIMessageChunk):
                continue
            content = message.content
            if isinstance(content, str):
                if content:
                    yield content
                continue

            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    yield block.get("text", "")

    async def stream_events(
        self,
        task: str,
        thread_id: str = "default",
    ) -> AsyncIterator[AgentEvent]:
        """Yield model text and normalized tool lifecycle events."""
        config = {"configurable": {"thread_id": thread_id}}
        pending: dict[object, dict[str, str]] = {}
        calls: dict[str, dict[str, object]] = {}

        def emit_ready(key: object) -> AgentEvent | None:
            call = pending.get(key)
            if not call or not call["id"] or not call["name"]:
                return None
            try:
                args = json.loads(call["args"] or "{}")
            except (json.JSONDecodeError, TypeError):
                return None
            if not isinstance(args, dict):
                args = {"value": args}
            call_id = call["id"]
            calls[call_id] = {"name": call["name"], "args": args}
            pending.pop(key, None)
            return {
                "kind": "tool_start",
                "id": call_id,
                "name": call["name"],
                "args": args,
            }

        async for message, metadata in self.agent.astream(
            {"messages": [{"role": "user", "content": task}]},
            config=config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node") or metadata.get("name")
            subagent_names = {agent["name"] for agent in CORE_SUBAGENTS}
            subagent = node if node in subagent_names else None
            if subagent is None:
                namespace = str(metadata.get("langgraph_checkpoint_ns", ""))
                subagent = next(
                    (
                        name
                        for name in subagent_names
                        if namespace == name
                        or namespace.startswith(name + ":")
                        or f"|{name}:" in namespace
                    ),
                    None,
                )
            if isinstance(message, AIMessageChunk):
                content = message.content
                if isinstance(content, str) and content:
                    event: AgentEvent = {"kind": "text", "text": content}
                    if subagent:
                        event["subagent"] = subagent
                    yield event
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            value = block.get("text", "")
                            if value:
                                event = {"kind": "text", "text": str(value)}
                                if subagent:
                                    event["subagent"] = subagent
                                yield event
                for chunk in message.tool_call_chunks or []:
                    key = chunk.get("index")
                    if key is None:
                        key = chunk.get("id") or f"pending-{len(pending)}"
                    call = pending.setdefault(
                        key, {"id": "", "name": "", "args": ""}
                    )
                    if chunk.get("id"):
                        call["id"] = str(chunk["id"])
                    if chunk.get("name"):
                        call["name"] += str(chunk["name"])
                    if chunk.get("args"):
                        call["args"] += str(chunk["args"])
                    ready = emit_ready(key)
                    if ready:
                        if subagent:
                            ready["subagent"] = subagent
                        yield ready
            elif isinstance(message, ToolMessage):
                for key in list(pending):
                    ready = emit_ready(key)
                    if ready:
                        if subagent:
                            ready["subagent"] = subagent
                        yield ready
                call = calls.get(str(message.tool_call_id), {})
                content = message.content
                if isinstance(content, list):
                    content = "\n".join(
                        str(block.get("text", ""))
                        if isinstance(block, dict)
                        else str(block)
                        for block in content
                    )
                else:
                    content = str(content)
                event = {
                    "kind": "tool_result",
                    "id": str(message.tool_call_id),
                    "name": str(call.get("name", "")),
                    "content": content,
                    "is_error": getattr(message, "status", None) == "error"
                    or content.startswith("Error"),
                }
                if subagent:
                    event["subagent"] = subagent
                yield event

        for key in list(pending):
            ready = emit_ready(key)
            if ready:
                yield ready
        yield {"kind": "done"}
