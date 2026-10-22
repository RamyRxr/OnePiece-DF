"""Minimal websocket client that prints the state stream.

    python -m ofp_vision.tools.probe_client
    python -m ofp_vision.tools.probe_client --watch

This is the reference implementation of the protocol from the consumer side;
the Unity client mirrors it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def run(host: str, port: int, watch: bool) -> int:
    from websockets.asyncio.client import connect

    uri = f"ws://{host}:{port}"
    try:
        async with connect(uri) as connection:
            print(f"connected to {uri}")
            while True:
                raw = await connection.recv()
                message = json.loads(raw)
                if watch:
                    channels = message.get("ch", {})
                    active = {k: v["v"] for k, v in channels.items() if abs(v["v"]) > 0.001}
                    print(f"t={message['t']:>8} q={message['q']} active={active} ev={message['ev']}")
                else:
                    print(json.dumps(message, indent=2))
                    return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"could not connect to {uri}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args.host, args.port, args.watch))


if __name__ == "__main__":
    raise SystemExit(main())
