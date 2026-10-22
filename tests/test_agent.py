import asyncio

from langchain_core.messages import AIMessageChunk, ToolMessage

from agent.agent import CodingAgent


class FakeGraph:
    def __init__(self, items):
        self.items = items

    async def astream(self, *a, **k):
        for item in self.items:
            yield item, {}


def collect(items):
    agent = CodingAgent.__new__(CodingAgent)
    agent.agent = FakeGraph(items)

    async def go():
        return [t async for t in agent.stream("task")]

    return asyncio.run(go())


def test_stream_emits_only_model_text_not_tool_results():
    out = collect([
        AIMessageChunk(content="Hello "),
        ToolMessage(content="SECRET FILE CONTENTS", tool_call_id="1"),
        AIMessageChunk(content=[{"type": "text", "text": "world"},
                                {"type": "tool_use", "name": "x"}]),
        AIMessageChunk(content=""),
    ])
    assert out == ["Hello ", "world"]


def test_agent_builds_with_dummy_key(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    assert CodingAgent(InMemorySaver(), [], tmp_path).agent is not None


def test_stream_events_reassembles_tool_chunks_and_normalizes_result():
    chunks = [
        AIMessageChunk(content="before "),
        AIMessageChunk(content="", tool_call_chunks=[{
            "name": "workspace_read_file", "args": '{"path":',
            "id": "call-1", "index": 0,
        }]),
        AIMessageChunk(content="", tool_call_chunks=[{
            "args": '"src/app.py"', "index": 0,
        }]),
        AIMessageChunk(content="", tool_call_chunks=[{
            "args": "}", "index": 0,
        }]),
        ToolMessage(content=[{"type": "text", "text": "file contents"}],
                    tool_call_id="call-1"),
        AIMessageChunk(content="after"),
    ]
    agent = CodingAgent.__new__(CodingAgent)
    agent.agent = FakeGraph(chunks)

    async def go():
        return [event async for event in agent.stream_events("task")]

    assert asyncio.run(go()) == [
        {"kind": "text", "text": "before "},
        {"kind": "tool_start", "id": "call-1", "name": "workspace_read_file",
         "args": {"path": "src/app.py"}},
        {"kind": "tool_result", "id": "call-1", "name": "workspace_read_file",
         "content": "file contents", "is_error": False},
        {"kind": "text", "text": "after"},
        {"kind": "done"},
    ]
