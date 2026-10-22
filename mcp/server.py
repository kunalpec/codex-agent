"""FastMCP tools for operating on the current project workspace."""

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastmcp import FastMCP
from ollama import Client
from tavily import TavilyClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = Path(os.environ.get("MCP_WORKSPACE_ROOT", Path.cwd())).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ollama_config import ollama_base_url
from sandbox.runner import DockerSandbox


IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__", "node_modules"}
DEFAULT_CHECK_COMMANDS = {
    "tests": ["python", "-m", "pytest"],
    "lint": ["ruff", "check", "."],
    "format": ["ruff", "format", "--check", "."],
    "types": ["mypy", "."],
    "build": ["python", "-m", "build"],
    "security": ["pip-audit"],
}

mcp = FastMCP("coding-agent-workspace")
sandbox = DockerSandbox(WORKSPACE_ROOT)


def _resolve_path(path: str) -> Path:
    # `/workspace` is the Docker mount path. Accept it as a convenience when
    # the agent passes a container path to a host-side workspace tool.
    if path == "/workspace":
        path = "."
    elif path.startswith("/workspace/"):
        path = path.removeprefix("/workspace/")

    target = (WORKSPACE_ROOT / path).resolve()
    if not target.is_relative_to(WORKSPACE_ROOT):
        raise ValueError("Path must stay inside the project workspace.")
    return target


def _is_ignored(path: Path, root: Path) -> bool:
    """Return whether a path is beneath a generated or version-control folder."""
    return any(part in IGNORED_DIRECTORIES for part in path.relative_to(root).parts)


def _run_sandbox_command(
    command: list[str],
    *,
    timeout: int,
    network: str = "bridge",
    directory: str = ".",
) -> dict[str, object]:
    """Run a command in Docker, optionally from a workspace subdirectory."""
    target = _resolve_path(directory)
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    relative_directory = target.relative_to(WORKSPACE_ROOT).as_posix()
    if relative_directory == ".":
        return sandbox.run(command, timeout=timeout, network=network)
    return sandbox.run(
        command,
        timeout=timeout,
        network=network,
        working_directory=relative_directory,
    )


def _nearest_manifest_directory(start: Path, manifest: str) -> Path | None:
    """Find the closest ancestor containing a manifest without leaving the workspace."""
    current = start
    while current.is_relative_to(WORKSPACE_ROOT):
        if (current / manifest).is_file():
            return current
        if current == WORKSPACE_ROOT:
            break
        current = current.parent
    return None


def _run_compiled_source(
    source: Path, compiler: str, args: list[str], timeout: int
) -> dict[str, object]:
    """Compile and run one C, C++, or standalone Rust source file in Docker."""
    source_path = f"/workspace/{source.relative_to(WORKSPACE_ROOT).as_posix()}"
    binary_path = f"/opt/packages/tmp/codex-agent-{uuid4().hex}"
    language = "rust" if compiler == "rustc" else "native"
    directory_result = sandbox.run(
        ["mkdir", "-p", "/opt/packages/tmp"], timeout=10, language=language
    )
    if directory_result["returncode"] != 0:
        return directory_result

    compile_result = sandbox.run(
        [compiler, source_path, "-o", binary_path], timeout=timeout, language=language
    )
    if compile_result["returncode"] != 0:
        sandbox.run(["rm", "-f", binary_path], timeout=10, language=language)
        return {"compile": compile_result, "returncode": compile_result["returncode"]}

    try:
        run_result = sandbox.run(
            [binary_path, *args], timeout=timeout, language=language
        )
    finally:
        sandbox.run(["rm", "-f", binary_path], timeout=10, language=language)
    return {
        "compile": compile_result,
        "execution": run_result,
        "returncode": run_result["returncode"],
    }


@mcp.tool
def workspace_list_files(directory: str = ".", max_depth: int = 3) -> list[str]:
    """List project files beneath a directory, skipping generated folders."""
    root = _resolve_path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    max_depth = max(0, min(max_depth, 10))
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if len(relative.parts) > max_depth + 1:
            continue
        if path.is_file():
            files.append(path.relative_to(WORKSPACE_ROOT).as_posix())
    return sorted(files)


@mcp.tool
def workspace_read_file(path: str) -> str:
    """Read a UTF-8 text file from the project workspace."""
    target = _resolve_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return target.read_text(encoding="utf-8")


@mcp.tool
def workspace_write_file(path: str, content: str) -> str:
    """Create or replace a UTF-8 text file inside the project workspace."""
    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {target.relative_to(WORKSPACE_ROOT).as_posix()}"


@mcp.tool
def workspace_create_file(path: str, content: str) -> str:
    """Create a new UTF-8 file without replacing an existing file."""
    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as file:
        file.write(content)
    return f"Created {target.relative_to(WORKSPACE_ROOT).as_posix()}"


@mcp.tool
def workspace_delete_file(path: str, confirm: bool = False) -> str:
    """Delete one file in the workspace after explicit confirmation."""
    if not confirm:
        raise ValueError("Set confirm=true only when file deletion was requested.")
    target = _resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not target.is_file():
        raise IsADirectoryError(f"Not a file: {path}")
    target.unlink()
    return f"Deleted {target.relative_to(WORKSPACE_ROOT).as_posix()}"


def _rename_workspace_path(source: str, destination: str, *, directory: bool) -> str:
    old = _resolve_path(source)
    new = _resolve_path(destination)
    kind = "Directory" if directory else "File"
    if not old.exists():
        raise FileNotFoundError(f"{kind} not found: {source}")
    if directory and not old.is_dir():
        raise NotADirectoryError(f"Not a directory: {source}")
    if not directory and not old.is_file():
        raise IsADirectoryError(f"Not a file: {source}")
    if new.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if directory and new.is_relative_to(old):
        raise ValueError("A directory cannot be moved inside itself.")
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    old_relative = old.relative_to(WORKSPACE_ROOT).as_posix()
    new_relative = new.relative_to(WORKSPACE_ROOT).as_posix()
    return f"Renamed {old_relative} to {new_relative}"


@mcp.tool
def workspace_rename_file(path: str, new_path: str) -> str:
    """Rename or move a file to another path inside the workspace."""
    return _rename_workspace_path(path, new_path, directory=False)


@mcp.tool
def workspace_rename_folder(path: str, new_path: str) -> str:
    """Rename or move a folder to another path inside the workspace."""
    return _rename_workspace_path(path, new_path, directory=True)


@mcp.tool
def workspace_search_files(query: str, directory: str = ".") -> list[str]:
    """Find workspace text files containing a case-insensitive search string."""
    root = _resolve_path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or any(
            part in IGNORED_DIRECTORIES for part in path.relative_to(root).parts
        ):
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(WORKSPACE_ROOT):
                continue
            if query.casefold() in resolved.read_text(encoding="utf-8").casefold():
                matches.append(path.relative_to(WORKSPACE_ROOT).as_posix())
        except (UnicodeDecodeError, OSError):
            continue
    return sorted(matches)


@mcp.tool
def workspace_glob_search(
    pattern: str, directory: str = ".", max_results: int = 500
) -> list[str]:
    """Find workspace files by a relative glob pattern such as `src/**/*.py`."""
    glob_path = Path(pattern)
    if not pattern.strip() or glob_path.is_absolute() or ".." in glob_path.parts:
        raise ValueError("Provide a relative glob pattern without '..'.")
    root = _resolve_path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    max_results = max(1, min(max_results, 1000))
    matches = []
    for path in root.glob(pattern):
        if not path.is_file() or _is_ignored(path, root):
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(WORKSPACE_ROOT):
                continue
            matches.append(resolved.relative_to(WORKSPACE_ROOT).as_posix())
        except OSError:
            continue
        if len(matches) >= max_results:
            break
    return sorted(matches)


@mcp.tool
def workspace_search_code(
    query: str, directory: str = ".", max_results: int = 100
) -> list[dict[str, object]]:
    """Find case-insensitive text matches with file paths and line numbers."""
    if not query:
        raise ValueError("Query cannot be empty.")
    root = _resolve_path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    max_results = max(1, min(max_results, 1000))
    matches: list[dict[str, object]] = []
    for path in root.rglob("*"):
        if not path.is_file() or _is_ignored(path, root):
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(WORKSPACE_ROOT):
                continue
            if resolved.stat().st_size > 1_000_000:
                continue
            for line_number, line in enumerate(
                resolved.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if query.casefold() in line.casefold():
                    matches.append(
                        {
                            "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= max_results:
                        return matches
        except (UnicodeDecodeError, OSError, ValueError):
            continue
    return matches


@mcp.tool
def get_project_info() -> dict[str, str]:
    """Return the workspace's virtual root and host platform."""
    return {
        "workspace_root": ".",
        "platform": os.name,
    }


@mcp.tool
def check_docker() -> dict[str, object]:
    """Check whether Docker Desktop or the Docker daemon is available."""
    return sandbox.check_docker()


@mcp.tool
def check_sandbox_image(
    language: Literal["python", "node", "java", "go", "rust", "native"] = "python",
) -> dict[str, object]:
    """Check whether a language-specific sandbox image exists locally."""
    return sandbox.check_image(language)


@mcp.tool
def list_ollama_models() -> list[dict[str, object]]:
    """List the models exposed by the configured Ollama `/api/tags` endpoint."""
    response = Client(host=ollama_base_url()).list()
    models = []
    for item in response.models:
        name = getattr(item, "model", None) or getattr(item, "name", "")
        details = getattr(item, "details", None)
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        models.append({
            "name": name,
            "model": name,
            "details": details or {},
            "capabilities": list(getattr(item, "capabilities", []) or []),
        })
    return models


@mcp.tool
def build_sandbox_image(
    language: Literal["python", "node", "java", "go", "rust", "native"] = "python",
    timeout: int = 900,
) -> dict[str, object]:
    """Build one language-specific image for running and testing project code."""
    return sandbox.build_image(language=language, timeout=timeout)


def _ensure_sandbox_pip(timeout: int) -> dict[str, object]:
    """Repair older persistent sandbox volumes that were created without pip."""
    return sandbox.run(
        ["python", "-m", "ensurepip", "--upgrade"],
        timeout=min(timeout, 60),
        network="none",
    )


@mcp.tool
def install_packages(
    packages: list[str],
    ecosystem: Literal["python", "npm", "go", "cargo", "maven"],
    timeout: int = 300,
) -> dict[str, object]:
    """Install packages for Python, npm, Go, Cargo, or Maven in Docker."""
    if not packages or any(not package.strip() for package in packages):
        raise ValueError("Provide one or more package names.")
    if any(package.startswith("-") for package in packages):
        raise ValueError("Package names cannot be command-line options.")

    if ecosystem == "python":
        bootstrap = _ensure_sandbox_pip(timeout)
        if bootstrap["returncode"] != 0:
            return bootstrap
        command = ["python", "-m", "pip", "install", *packages]
    elif ecosystem == "npm":
        command = ["npm", "install", "--global", *packages]
    elif ecosystem == "go":
        command = ["go", "install", *packages]
    elif ecosystem == "cargo":
        command = ["cargo", "install", *packages]
    elif ecosystem == "maven":
        if len(packages) != 1:
            raise ValueError("Install one Maven artifact coordinate at a time.")
        command = ["mvn", "--batch-mode", "dependency:get", f"-Dartifact={packages[0]}"]
    else:
        raise ValueError(f"Unsupported package ecosystem: {ecosystem}")

    return sandbox.run(command, timeout=timeout, network="bridge")


@mcp.tool
def install_python_packages(
    packages: list[str], timeout: int = 300
) -> dict[str, object]:
    """Install named Python packages in the persistent sandbox environment."""
    return install_packages(packages, ecosystem="python", timeout=timeout)


@mcp.tool
def install_project_dependencies(
    timeout: int = 600, directory: str = "."
) -> dict[str, object]:
    """Install declared Python, Node, Java, Go, and Rust project dependencies."""
    project_root = _resolve_path(directory)
    if not project_root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    relative_directory = project_root.relative_to(WORKSPACE_ROOT).as_posix()

    install_steps: list[tuple[str, list[str]]] = []
    if (project_root / "pyproject.toml").is_file():
        install_steps.append(("python", ["python", "-m", "pip", "install", "-e", "."]))
    if (project_root / "requirements.txt").is_file():
        install_steps.append((
            "python",
            ["python", "-m", "pip", "install", "-r", "requirements.txt"],
        ))

    package_json = project_root / "package.json"
    if package_json.is_file():
        lockfiles = [
            name
            for name in (
                "package-lock.json",
                "npm-shrinkwrap.json",
                "pnpm-lock.yaml",
                "yarn.lock",
            )
            if (project_root / name).is_file()
        ]
        lock_managers = {
            "package-lock.json": "npm",
            "npm-shrinkwrap.json": "npm",
            "pnpm-lock.yaml": "pnpm",
            "yarn.lock": "yarn",
        }
        managers = {lock_managers[name] for name in lockfiles}
        if len(managers) > 1:
            raise ValueError("Conflicting JavaScript lockfiles were found.")

        declared_manager = None
        manager_spec = None
        try:
            package_manager_field = json.loads(
                package_json.read_text(encoding="utf-8")
            ).get("packageManager", "")
            if isinstance(package_manager_field, str) and "@" in package_manager_field:
                declared_manager, manager_version = package_manager_field.split("@", 1)
                manager_spec = (
                    f"{declared_manager}@{manager_version.split('+', 1)[0]}"
                )
        except (json.JSONDecodeError, OSError, AttributeError):
            declared_manager = None

        if managers and declared_manager and declared_manager not in managers:
            raise ValueError("packageManager conflicts with the JavaScript lockfile.")
        package_manager = next(iter(managers), declared_manager or "npm")
        if package_manager == "npm":
            npm_command = "ci" if lockfiles else "install"
            if manager_spec and declared_manager == package_manager:
                command = ["npx", "--yes", manager_spec, npm_command]
            else:
                command = ["npm", npm_command]
        elif package_manager in {"pnpm", "yarn"}:
            if declared_manager != package_manager:
                manager_spec = None
            manager_spec = manager_spec or package_manager
            command = [
                "npx", "--yes", manager_spec, "install", "--frozen-lockfile"
            ]
        else:
            raise ValueError(f"Unsupported packageManager: {package_manager}")
        install_steps.append((package_manager, command))

    if (project_root / "pom.xml").is_file():
        install_steps.append(
            ("maven", ["mvn", "--batch-mode", "dependency:go-offline"])
        )

    gradle_files = any(
        (project_root / filename).is_file()
        for filename in ("build.gradle", "build.gradle.kts")
    )
    if gradle_files:
        if (project_root / "gradlew").is_file():
            install_steps.append((
                "gradle",
                ["sh", "./gradlew", "--no-daemon", "dependencies"],
            ))
        else:
            install_steps.append((
                "gradle",
                ["gradle", "--no-daemon", "dependencies"],
            ))

    if (project_root / "go.mod").is_file():
        install_steps.append(("go", ["go", "mod", "download"]))
    if (project_root / "Cargo.toml").is_file():
        cargo_command = ["cargo", "fetch"]
        if (project_root / "Cargo.lock").is_file():
            cargo_command.append("--locked")
        install_steps.append(("cargo", cargo_command))

    if not install_steps:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": "No supported dependency manifest was found.",
        }

    if any(ecosystem == "python" for ecosystem, _command in install_steps):
        bootstrap = _ensure_sandbox_pip(timeout)
        if bootstrap["returncode"] != 0:
            return {"steps": [bootstrap], "returncode": bootstrap["returncode"]}

    results = []
    for ecosystem, command in install_steps:
        result = _run_sandbox_command(
            command,
            timeout=timeout,
            network="bridge",
            directory=relative_directory,
        )
        results.append({"ecosystem": ecosystem, **result})

    return {
        "steps": results,
        "returncode": next(
            (step["returncode"] for step in results if step["returncode"] != 0),
            0,
        ),
    }


@mcp.tool
def run_python_file(
    path: str,
    args: list[str] | None = None,
    timeout: int = 60,
) -> dict[str, object]:
    """Run a Python file in the resource-limited Docker sandbox."""
    script = _resolve_path(path)
    if not script.is_file():
        raise FileNotFoundError(f"Python file not found: {path}")
    if script.suffix.lower() != ".py":
        raise ValueError("Only Python (.py) files can be run with this tool.")

    script_path = script.relative_to(WORKSPACE_ROOT).as_posix()
    return sandbox.run(
        ["python", f"/workspace/{script_path}", *(args or [])],
        timeout,
    )


@mcp.tool
def run_project_file(
    path: str,
    args: list[str] | None = None,
    timeout: int = 60,
) -> dict[str, object]:
    """Run Python, JavaScript, TypeScript, Go, Java, C, C++, or Rust in Docker."""
    source = _resolve_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")
    args = args or []
    if any(not isinstance(argument, str) for argument in args):
        raise ValueError("Arguments must be strings.")

    relative_path = source.relative_to(WORKSPACE_ROOT).as_posix()
    container_path = f"/workspace/{relative_path}"
    extension = source.suffix.lower()
    if extension == ".py":
        return run_python_file(path, args, timeout)
    if extension in {".js", ".mjs", ".cjs"}:
        return _run_sandbox_command(
            ["node", container_path, *args],
            timeout=timeout,
            directory=source.parent.relative_to(WORKSPACE_ROOT).as_posix(),
        )
    if extension in {".ts", ".tsx", ".mts", ".cts"}:
        return _run_sandbox_command(
            ["tsx", container_path, *args],
            timeout=timeout,
            directory=source.parent.relative_to(WORKSPACE_ROOT).as_posix(),
        )
    if extension == ".go":
        project_directory = _nearest_manifest_directory(source.parent, "go.mod")
        if project_directory is None:
            raise ValueError("Go source must be inside a project containing go.mod.")
        return _run_sandbox_command(
            ["go", "run", container_path, *args],
            timeout=timeout,
            directory=project_directory.relative_to(WORKSPACE_ROOT).as_posix(),
        )
    if extension == ".java":
        return _run_sandbox_command(
            ["java", container_path, *args],
            timeout=timeout,
            directory=source.parent.relative_to(WORKSPACE_ROOT).as_posix(),
        )
    if extension == ".c":
        return _run_compiled_source(source, "gcc", args, timeout)
    if extension in {".cc", ".cpp", ".cxx"}:
        return _run_compiled_source(source, "g++", args, timeout)
    if extension == ".rs":
        project_directory = _nearest_manifest_directory(source.parent, "Cargo.toml")
        if project_directory is not None:
            relative_manifest = (project_directory / "Cargo.toml").relative_to(
                WORKSPACE_ROOT
            ).as_posix()
            manifest_path = f"/workspace/{relative_manifest}"
            return _run_sandbox_command(
                ["cargo", "run", "--manifest-path", manifest_path, "--", *args],
                timeout=timeout,
                directory=project_directory.relative_to(WORKSPACE_ROOT).as_posix(),
            )
        return _run_compiled_source(source, "rustc", args, timeout)
    raise ValueError(f"Unsupported source file type: {extension or '(no extension)'}")


@mcp.tool
def run_project_command(
    command: list[str],
    directory: str = ".",
    timeout: int = 60,
) -> dict[str, object]:
    """Run a project-specific command in Docker from a workspace directory."""
    if not command or any(not isinstance(argument, str) for argument in command):
        raise ValueError("Command must be a non-empty list of strings.")
    return _run_sandbox_command(
        command,
        timeout=timeout,
        directory=directory,
    )


@mcp.tool
def run_command(command: list[str], timeout: int = 60) -> dict[str, object]:
    """Run an argument list in Docker without a shell."""
    if not command:
        raise ValueError("Command cannot be empty.")
    # Models sometimes send the entire command as one list item, like
    # ["python --version"]. Split it without starting a shell.
    if len(command) == 1:
        command = shlex.split(command[0])
    if not command:
        raise ValueError("Command cannot be empty.")
    return sandbox.run(command, timeout)


def _run_git(command: list[str]) -> dict[str, object]:
    """Run a read-only Git command and bound the returned output size."""
    result = sandbox.run(["git", *command], timeout=60)
    output_limit = 50_000
    for stream in ("stdout", "stderr"):
        value = str(result.get(stream, ""))
        if len(value) > output_limit:
            result[stream] = value[:output_limit] + "\n... output truncated"
            result[f"{stream}_truncated"] = True
    return result


@mcp.tool
def git_status() -> dict[str, object]:
    """Show the current branch and workspace changes without modifying them."""
    return _run_git(["status", "--short", "--branch"])


@mcp.tool
def git_diff(path: str | None = None, staged: bool = False) -> dict[str, object]:
    """Show unstaged or staged Git changes, optionally limited to one path."""
    command = ["diff", "--no-ext-diff", "--no-color", "--unified=3"]
    if staged:
        command.append("--cached")
    if path is not None:
        target = _resolve_path(path)
        command.extend(["--", target.relative_to(WORKSPACE_ROOT).as_posix()])
    return _run_git(command)


@mcp.tool
def git_log(max_entries: int = 20) -> dict[str, object]:
    """Show recent commit summaries without changing repository state."""
    max_entries = max(1, min(max_entries, 100))
    return _run_git(
        ["log", "--oneline", "--decorate", "-n", str(max_entries)]
    )


@mcp.tool
def git_branch() -> dict[str, object]:
    """Show the current Git branch name without changing repository state."""
    return _run_git(["branch", "--show-current"])


@mcp.tool
def inspect_environment() -> dict[str, object]:
    """Report runtime information and whether supported credentials are set."""
    return {
        "host_platform": os.name,
        "model_provider": os.environ.get("MODEL_PROVIDER", "ollama"),
        "openrouter_model": os.environ.get("OPENROUTER_MODEL"),
        "ollama_model_configured": bool(os.environ.get("OLLAMA_MODEL")),
        "configured": {
            "OLLAMA_BASE_URL": bool(os.environ.get("OLLAMA_BASE_URL")),
            "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
            "TAVILY_API_KEY": bool(os.environ.get("TAVILY_API_KEY")),
        },
    }


@mcp.tool
def inspect_dependencies(timeout: int = 60) -> dict[str, object]:
    """List Python packages installed in the persistent Docker environment."""
    return sandbox.run(["python", "-m", "pip", "list", "--format=json"], timeout)


@mcp.tool
def check_dependencies(timeout: int = 60) -> dict[str, object]:
    """Check installed Python packages for dependency conflicts."""
    return sandbox.run(["python", "-m", "pip", "check"], timeout)


@mcp.tool
def run_project_check(
    check: Literal["tests", "lint", "format", "types", "build", "security"],
    command: list[str] | None = None,
    timeout: int = 300,
    directory: str = ".",
) -> dict[str, object]:
    """Run tests, lint, format, type, build, or security checks inside Docker."""
    if command is not None and not command:
        raise ValueError("Provide a non-empty command argument list.")
    if check not in DEFAULT_CHECK_COMMANDS:
        raise ValueError(f"Unsupported project check: {check}")
    selected_command = command if command is not None else DEFAULT_CHECK_COMMANDS[check]
    if not selected_command or any(not part for part in selected_command):
        raise ValueError("Provide a non-empty command argument list.")
    return _run_sandbox_command(
        selected_command,
        timeout=timeout,
        network="bridge",
        directory=directory,
    )


def _tavily() -> TavilyClient:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set in .env")
    return TavilyClient(api_key=key)


@mcp.tool
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict[str, object]:
    """Search the internet with Tavily. Use for docs, errors, and current info."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if search_depth not in {"basic", "advanced"}:
        raise ValueError("search_depth must be 'basic' or 'advanced'.")
    max_results = max(1, min(max_results, 10))

    response = _tavily().search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=True,
    )
    return {
        "answer": response.get("answer"),
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
            }
            for r in response.get("results", [])
        ],
    }


@mcp.tool
def web_extract(urls: list[str]) -> dict[str, object]:
    """Fetch the full text of one or more web pages with Tavily."""
    if not urls or len(urls) > 5:
        raise ValueError("Provide between 1 and 5 URLs.")
    if any(not u.startswith(("http://", "https://")) for u in urls):
        raise ValueError("URLs must start with http:// or https://.")

    response = _tavily().extract(urls=urls)
    return {
        "results": [
            {"url": r.get("url"), "content": r.get("raw_content")}
            for r in response.get("results", [])
        ],
        "failed": response.get("failed_results", []),
    }
    
if __name__ == "__main__":
    mcp.run(transport="stdio")
