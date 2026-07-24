from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import traceback
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relay CommunicationMod over a local TCP socket.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51234)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


def append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def receive_line(connection: socket.socket) -> str | None:
    chunks = bytearray()
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            return None
        newline_index = chunk.find(b"\n")
        if newline_index >= 0:
            chunks.extend(chunk[:newline_index])
            return chunks.decode("utf-8")
        chunks.extend(chunk)


def run(args: argparse.Namespace) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print("ready", flush=True)
    pending_state: str | None = None
    connection: socket.socket | None = None

    while True:
        if pending_state is None:
            pending_state = sys.stdin.readline()
            if pending_state == "":
                return
            pending_state = pending_state.rstrip("\r\n\0")
            append_jsonl(
                args.trace,
                {
                    "timestamp": timestamp(),
                    "direction": "game_to_agent",
                    "payload": json.loads(pending_state),
                },
            )

        if connection is None:
            connection, _ = server.accept()

        try:
            connection.sendall(pending_state.encode("utf-8") + b"\n")
            command = receive_line(connection)
            if command is None:
                connection.close()
                connection = None
                continue
            append_jsonl(
                args.trace,
                {
                    "timestamp": timestamp(),
                    "direction": "agent_to_game",
                    "command": command,
                },
            )
            print(command, flush=True)
            pending_state = None
        except (ConnectionError, OSError):
            connection.close()
            connection = None


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except BaseException:
        if args.log is not None:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open("a", encoding="utf-8") as stream:
                stream.write(f"[{timestamp()}]\n")
                traceback.print_exc(file=stream)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
