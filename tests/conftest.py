import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Load mcp/server.py with its workspace pointed at a temp folder.

    The local `mcp` folder shares a name with the MCP SDK, so the module is
    loaded by file path rather than imported.
    """
    monkeypatch.setenv("MCP_WORKSPACE_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "workspace_server", PROJECT_ROOT / "mcp" / "server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
