from typing import Any

from deepagents.graph import DeepAgentState


class CodingAgentState(DeepAgentState, total=False):
    """Additional task and workflow context for the coding agent."""

    task_id: str
    task_description: str
    workspace_id: str
    project_root: str

    current_stage: str
    plan: list[str]
    completed_steps: list[str]

    changed_files: list[str]
    diff_summary: str

    test_results: dict[str, Any]
    errors: list[str]
    retry_count: int

    status: str
    final_summary: str
