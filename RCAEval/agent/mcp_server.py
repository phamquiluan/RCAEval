"""MCP server over one case of a blind task view.

    rcaeval-agent mcp --case agent_tasks/re2-ob/cases/re2-ob-0007 --record runs/re2-ob-0007

Serves the CaseToolkit tools (describe_case, ..., submit) over stdio to any MCP
client: Claude Code, Codex, OpenCode or your own agent. Every call is appended
to <record>/calls.jsonl and an accepted submit is written to
<record>/submission.json, a server-side path the agent never needs to see.
"""
import json
import os
from os.path import join

from RCAEval.agent.toolkit import TOOLS, CaseToolkit

INSTRUCTIONS = (
    "Tools for diagnosing one failure case of the RCAEval benchmark. Start with describe_case, "
    "and finish by calling submit with your ranked root-cause candidates."
)


def serve(case_dir, record_dir):
    try:
        import anyio
        import mcp.types as types
        from mcp.server.lowlevel import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        raise SystemExit("the MCP server needs the mcp package: pip install RCAEval[agent]")

    os.makedirs(record_dir, exist_ok=True)
    toolkit = CaseToolkit(case_dir)
    tools = [types.Tool(name=t["name"], description=t["description"], input_schema=t["parameters"]) for t in TOOLS]

    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params):
        # pandas work runs in a worker thread so the server keeps answering pings
        result = await anyio.to_thread.run_sync(toolkit.call, params.name, params.arguments or {})
        with open(join(record_dir, "calls.jsonl"), "a") as f:
            f.write(json.dumps({"type": "tool_call", **toolkit.calls[-1]}, default=str) + "\n")
        if params.name == "submit" and toolkit.submission is not None:
            with open(join(record_dir, "submission.json"), "w") as f:
                json.dump(toolkit.submission, f)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, default=str))],
            is_error="error" in result,
        )

    server = Server("rcaeval", instructions=INSTRUCTIONS, on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(main)
