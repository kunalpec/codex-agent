"""Connect the agent to its local FastMCP workspace server."""

import os
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from langchain.mcp import MCPAdapter


SERVER_PATH = Path(__file__).resolve().with_name("server.py")


def create_mcp_client(workspace: Path) -> MCPAdapter:
    """Connect to the MCP server with an explicit workspace folder."""
    workspace = workspace.resolve()
    server_env = os.environ.copy()
    server_env["MCP_WORKSPACE_ROOT"] = str(workspace)

    transport = PythonStdioTransport(
        SERVER_PATH,
        env=server_env,
        cwd=str(workspace),
    )
    return MCPAdapter(Client(transport))
