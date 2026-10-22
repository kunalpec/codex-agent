# Codex Agent

Codex Agent is a terminal coding assistant for an existing project. It uses OpenRouter or an Ollama model, gives the agent workspace-scoped file tools, and runs project commands in a Docker sandbox.

## Features

- Reads, searches, writes, and renames files in the selected workspace.
- Streams responses and tool activity in a Rich terminal interface.
- Runs supported language files and project commands in a resource-limited Docker container.
- Saves conversation checkpoints in `checkpoints.sqlite` inside the workspace.
- Supports OpenRouter and local or remote Ollama servers.
- Can search and read web pages with Tavily when configured.
- Uses focused specialist agents when useful:
  - `explorer` traces code paths; `planner` breaks work into steps.
  - `researcher` investigates libraries and APIs; `coder` implements changes.
  - `debugger` finds root causes; `test_engineer` writes and runs tests.
  - `reviewer` checks logic and quality; `security_auditor` checks security.
  - `performance_analyzer` investigates bottlenecks; `verifier` validates results.
  - `documenter` updates docs; `shell_agent` runs sandboxed commands.
  - `browser_agent` inspects web pages and documented workflows. Interactive
    browser control is not available in the current toolset.

The main agent coordinates these specialists and has the same workspace tools. They are delegated as needed; you do not need to start them separately.

## MCP tools

The workspace MCP server provides:

- Workspace file read, create, write, rename, confirmed file deletion, glob search, and code search. CSV exports create a new `.csv` file and show a table preview.
- Docker-sandboxed Python, JavaScript/TypeScript, Java, Go, Rust, C, and C++ execution through `run_project_file` and `run_project_command`.
- Project dependency installation for Python, Node.js, Maven/Gradle, Go, and Rust manifests.
- Package installation for Python, npm, Go, Cargo, and Maven ecosystems. Installed tools and package caches persist in Docker volumes.
- Standard tests, lint, formatting, type, build, and dependency security checks.
- Read-only Git status, diff, branch, and log inspection.
- Dependency and safe environment inspection, plus Ollama model discovery.
- Tavily web search and page extraction when `TAVILY_API_KEY` is configured.

Docker commands use bridge networking by default, so code running in the sandbox can access the network. Interactive browser control, database access, deployment, and authenticated API tools are not configured.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for installing and running the project.
- Docker Desktop on Windows/macOS or Docker Engine on Linux. Docker must be running to execute project code and commands.
- An OpenRouter API key, or access to a local/remote Ollama server.

## Install

### 1. Install prerequisites

Install Python 3.12 or newer and `uv`. On Windows, you can install `uv` with WinGet:

```powershell
winget install --id astral-sh.uv -e
```

On macOS or Linux, use the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install a managed Python 3.12 runtime with:

```powershell
uv python install 3.12
```

Open a new terminal and confirm the tools are available:

```powershell
python --version
uv --version
```

The `uv` commands below use the official installation methods documented by [uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Get the project and install its dependencies

If you are cloning from Git, replace `<repository-url>` with this project's Git URL:

```powershell
git clone <repository-url> codex-agent
Set-Location codex-agent
```

If you already downloaded the project, open PowerShell in its folder instead. Then run:

```powershell
uv sync
Copy-Item .env.example .env
notepad .env
```

For macOS or Linux, use:

```bash
git clone <repository-url> codex-agent
cd codex-agent
uv sync
cp .env.example .env
${EDITOR:-nano} .env
```

If you already downloaded the project, `cd` to its folder before running `uv sync`. Set the provider and its settings in `.env` before starting the app. The application loads this file from the repository/installation directory. Never commit API keys or other secrets.

### OpenRouter setup

Get an API key from [OpenRouter](https://openrouter.ai/), then set:

```dotenv
MODEL_PROVIDER=openrouter
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=openai/gpt-4o
OPENROUTER_MAX_TOKENS=8192
```

Choose a model ID supported by your OpenRouter account. `OPENROUTER_MAX_TOKENS` sets the maximum response length and defaults to `8192`. If an OpenRouter request fails, the app can retry that task with the configured Ollama model when Ollama is reachable.

### Local Ollama setup

Install [Ollama](https://ollama.com/download), start it, and download a model. For example:

```powershell
ollama --version
ollama pull qwen3:8b
ollama list
```

The Ollama desktop app normally starts its local server automatically. On systems where it does not, open another terminal and run `ollama serve`, leaving that terminal open.

Set these values in `.env`:

```dotenv
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_CHAT_MODEL=qwen3:8b
```

`OLLAMA_BASE_URL` is the Ollama API root. `OLLAMA_MODEL` selects the chat model; `OLLAMA_CHAT_MODEL` is used when `OLLAMA_MODEL` contains a full `/api/tags` URL or when a fallback model is needed.

### Optional Tavily web search

Set `TAVILY_API_KEY` in `.env` to enable the web search and page extraction tools. Leave it blank to use the coding and Docker features without web search.

## Start the agent

After saving `.env`, start from the repository directory to use it as the workspace:

```powershell
uv run codex-agent
```

Choose a different workspace explicitly:

```powershell
uv run codex-agent --workspace "C:\path\to\your\project"
```

On macOS/Linux, the equivalent is:

```bash
uv run codex-agent --workspace /path/to/your/project
```

To see the available CLI options:

```powershell
uv run codex-agent --help
```

To install a global editable command instead, run `uv tool install --editable .` from this repository. Then change to the project you want the agent to work in and run `codex-agent`.

The selected workspace is where the workspace file tools operate and where `checkpoints.sqlite` is stored. The app also uses this directory as the working directory for Docker commands.

## Ollama on Kaggle (GPU) with a Cloudflare quick tunnel

This setup runs Ollama in a Kaggle notebook and exposes it temporarily to your laptop through a Cloudflare quick tunnel. Kaggle sessions and quick-tunnel URLs are temporary. Repeat the steps and update your laptop's `.env` whenever the URL changes.

### 1. Start Ollama in Kaggle

In a Kaggle notebook, enable a GPU under **Settings → Accelerator → GPU**, then run this cell:

```bash
!sudo apt-get update -qq
!sudo apt-get install -y -qq pciutils lshw zstd
!curl -fsSL https://ollama.com/install.sh | sh
!pip install -q ollama
```

Start the Ollama server in the background in a separate notebook cell:

```bash
%%bash --bg
export OLLAMA_HOST=0.0.0.0
export OLLAMA_ORIGINS=*
ollama serve > ollama.log 2>&1
```

Check the GPU and download a model. `gpt-oss:20b` needs substantial GPU memory; choose a smaller model if the Kaggle accelerator cannot load it.

```bash
!sleep 3
!nvidia-smi --query-gpu=name,memory.total --format=csv
!ollama pull gpt-oss:20b
```

The agent model must be available on this Ollama server. You can download a different model with `ollama pull <model-name>` and use that name in `.env` later.

### 2. Create a temporary Cloudflare tunnel

Download and check `cloudflared`:

```bash
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared --version
```

Start the tunnel in a background cell:

```bash
%%bash --bg
./cloudflared tunnel --url http://localhost:11434 --http-host-header localhost:11434 > cf.log 2>&1
```

Wait for and print the assigned URL in another cell:

```python
import re
import time

public_url = None
for _ in range(30):
    try:
        with open("cf.log", encoding="utf-8") as log_file:
            logs = log_file.read()
        urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", logs)
        if urls:
            public_url = urls[-1]
            break
    except FileNotFoundError:
        pass
    time.sleep(2)

if public_url:
    print("Public URL:", public_url)
    print("Models endpoint:", public_url + "/api/tags")
else:
    print("Tunnel URL not found. Check cf.log:")
    print(logs[-2000:] if "logs" in globals() else "No tunnel log yet.")
```

The `trycloudflare.com` address is a public, unauthenticated tunnel. Anyone who gets the URL may be able to reach the Ollama server while the Kaggle session is running. Share it carefully and stop the notebook session when finished. Do not put the temporary URL in source control.

### 3. Point the laptop agent at Kaggle

In the `.env` next to this project, set the current URL printed by Kaggle. Use the URL root without `/api/tags`:

```dotenv
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=https://your-current-name.trycloudflare.com
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_CHAT_MODEL=gpt-oss:20b
```

Replace the example host with the URL from your active Kaggle session. Set the model names to a model you actually pulled in Kaggle. Save `.env`, then launch or restart the agent. Use `/models` to see model capabilities and other Ollama metadata, and `/model ollama:<model-name>` to switch models during a session.

The app also accepts the legacy configuration where `OLLAMA_MODEL` is the full URL ending in `/api/tags` and `OLLAMA_CHAT_MODEL` is the model ID. Prefer `OLLAMA_BASE_URL` plus a model name as shown above.

### 4. Optional direct Python client from a notebook or laptop

Install the Ollama Python client with `pip install ollama`, then use the tunnel URL:

```python
import ollama

client = ollama.Client(
    host="https://your-current-name.trycloudflare.com"
)

for chunk in client.chat(
    model="gpt-oss:20b",
    messages=[
        {"role": "user", "content": "Explain indexing in DBMS with an example."}
    ],
    stream=True,
):
    print(chunk["message"]["content"], end="", flush=True)

print()
```

Use the same model name as the one pulled in Kaggle. The tunnel URL changes when a new quick tunnel is created.

## Docker sandbox

Docker is required to run project commands, supported language files, and checks through the agent. Start Docker Desktop or Docker Engine and confirm it is running:

```powershell
docker --version
docker info
```

The sandbox has separate images for Python, Node.js/TypeScript, Java, Go, Rust, and native C/C++. Build all of them from the repository folder with:

```powershell
docker build --tag devagent-python --file sandbox/Dockerfile sandbox
docker build --tag devagent-node --file sandbox/Dockerfile.node sandbox
docker build --tag devagent-java --file sandbox/Dockerfile.java sandbox
docker build --tag devagent-go --file sandbox/Dockerfile.go sandbox
docker build --tag devagent-rust --file sandbox/Dockerfile.rust sandbox
docker build --tag devagent-native --file sandbox/Dockerfile.native sandbox
```

You can also build an individual image with the `build_sandbox_image` MCP tool by passing `python`, `node`, `java`, `go`, `rust`, or `native` as `language`. The runner selects an image based on the project command or source file. Containers mount the selected workspace at `/workspace`, limit execution to 512 MB memory, one CPU, and 100 processes, and use Docker bridge networking so project commands can reach the network. Language caches and global tools persist in Docker volumes.

## Commands

| Command | Purpose |
| --- | --- |
| `/help` | Show available commands |
| `/tools` | List connected tools |
| `/model` | Show the active model |
| `/model ollama:qwen3:8b` | Select an Ollama model |
| `/model provider/model-id` | Select an OpenRouter model |
| `/models` | List Ollama models and available metadata |
| `/workspace` | Show workspace information |
| `/docker` | Check Docker availability |
| `/new` | Start a fresh conversation thread |
| `/clear` | Clear and redraw the screen |
| `/exit` | Exit the application |

Enter a normal request at the prompt to ask the agent to work on the project. Press Ctrl+C to cancel a running request and Ctrl+D at an empty prompt to exit. Use `--debug` to show tracebacks for unexpected application errors.

## Project layout

```text
agent/       Main agent, specialist agent definitions, and streaming
mcp/         Workspace, Ollama, web, and project-operation tools
sandbox/    Docker runner and sandbox image definition
ui/          Terminal commands, theme, and rendering
tests/       Unit tests
main.py      CLI entry point
```

## Development

Install the project dependencies with `uv sync`. To run the test suite:

```bash
uv run --with pytest pytest
```

The tests mock Docker and provider calls; they do not require an API key or a running Docker daemon.

## Security notes

- Keep `.env` private. Do not commit API keys, tunnel URLs, or other secrets.
- Workspace file tools are restricted to the selected workspace.
- Project commands run in the Docker sandbox. Review generated changes and commands before using them in important projects.
- The Kaggle quick tunnel is public and temporary. Anyone with its URL may be able to use the Ollama API while it is active.
