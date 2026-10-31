"""A stand-in "agent CLI" for tests: an MCP client that ranks anomalies and submits.

    python fake_mcp_agent.py <mcp_config.json>
"""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters, stdio_client


async def main(config_path):
    server = json.load(open(config_path))["mcpServers"]["rcaeval"]
    params = StdioServerParameters(command=server["command"], args=server["args"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            assert "submit" in tools, tools
            await session.call_tool("describe_case", {})
            result = await session.call_tool("rank_anomalies", {"top_k": 5})
            ranked = [r["name"] for r in json.loads(result.content[0].text)["ranked"]]
            await session.call_tool("get_metric", {"name": ranked[0]})
            await session.call_tool("submit", {"ranks": ranked, "reasoning": "largest shift after inject"})


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
