#!/usr/bin/env python3
"""Minimal MCP probe: connect to the server's /mcp endpoint and list tools.

Usage:
    python scripts/mcp_probe.py --server http://localhost:8000 --api-key TOKEN
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def run(server: str, api_key: str) -> int:
    url = server.rstrip("/") + "/mcp/"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(json.dumps({"tools": names}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args(argv)
    return asyncio.run(run(args.server, args.api_key))


if __name__ == "__main__":
    sys.exit(main())
