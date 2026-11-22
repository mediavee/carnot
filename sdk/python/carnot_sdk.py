"""
Zero-dependency Python 3.12+ async SDK & Negotiated Extension Host for Carnot.

Usage (Client):
    async with CarnotClient(cwd=".", provider="anthropic", model="claude-sonnet-5-5") as c:
        res = await c.prompt("Fix the failing tests")
        print(res["final_text"], res["metrics"])

Usage (Out-of-process JSON-RPC 2.0 Extension with I3 negotiated hooks):
    class MyExt(CarnotExtension):
        hooks = ["before_tool"]
        failure_policy = "fail_closed"

        def before_tool(self, tool: str, inp: dict) -> dict:
            if tool == "Bash" and "rm -rf /" in inp.get("command", ""):
                return {"decision": "deny", "reason": "dangerous command"}
            return {"decision": "allow"}
    MyExt().serve_stdio()
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Callable


class CarnotClient:
    """Async JSON-RPC 2.0 client driving `carnot --mode rpc` over stdio."""

    def __init__(
        self,
        binary: str = "carnot",
        cwd: str = ".",
        provider: str = "anthropic",
        model: str = "claude-sonnet-5-5",
        profile: str = "carnot",
        no_session: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.binary = binary
        self.cwd = cwd
        self.provider = provider
        self.model = model
        self.profile = profile
        self.no_session = no_session
        self.on_event = on_event
        self._proc: asyncio.subprocess.Process | None = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "CarnotClient":
        args = [
            "--mode", "rpc",
            "--cwd", self.cwd,
            "--provider", self.provider,
            "--model", self.model,
            "--profile", self.profile,
        ]
        if self.no_session:
            args.append("--no-session")
        self._proc = await asyncio.create_subprocess_exec(
            self.binary,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=16 * 1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            await self._proc.wait()

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while True:
                raw = await self._proc.stdout.readline()
                if not raw:
                    break
                try:
                    msg = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if "id" in msg and msg["id"] in self._pending:
                    fut = self._pending.pop(msg["id"])
                    if not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError(msg["error"]))
                        else:
                            fut.set_result(msg.get("result"))
                elif msg.get("method") == "event" and self.on_event:
                    self.on_event(msg.get("params", {}))
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(EOFError("carnot rpc process closed stdout"))
            self._pending.clear()

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        assert self._proc and self._proc.stdin
        self._req_id += 1
        rid = self._req_id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        frame = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}) + "\n"
        self._proc.stdin.write(frame.encode("utf-8"))
        await self._proc.stdin.drain()
        return await fut

    async def prompt(self, message: str) -> dict[str, Any]:
        return await self.call("prompt", {"message": message})

    async def steer(self, message: str) -> dict[str, Any]:
        return await self.call("steer", {"message": message})

    async def replay_events(self, after_seq: int = 0) -> dict[str, Any]:
        return await self.call("replay_events", {"after_seq": after_seq})

    async def get_state(self) -> dict[str, Any]:
        return await self.call("get_state")

    async def get_metrics(self) -> dict[str, Any]:
        return await self.call("get_metrics")

    async def abort(self) -> dict[str, Any]:
        return await self.call("abort")


class CarnotExtension:
    """Base class for writing out-of-process Carnot extensions in Python over JSON-RPC 2.0."""

    tools: list[dict[str, Any]] = []
    hooks: list[str] = ["session_start", "before_tool", "after_tool", "turn_boundary"]
    timeout_ms: int = 5000
    failure_policy: str = "fail_closed"

    def on_session_start(self, workspace_root: str) -> str | None:
        return None

    def before_tool(self, tool: str, inp: dict[str, Any]) -> dict[str, Any]:
        return {"decision": "allow"}

    def after_tool(self, tool: str, inp: dict[str, Any], is_error: bool, text: str) -> None:
        return None

    def on_turn_boundary(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {"action": "continue"}

    def call_tool(self, name: str, inp: dict[str, Any]) -> dict[str, Any]:
        return {"text": f"unhandled tool: {name}", "is_error": True}

    def serve_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            rid = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            try:
                if method == "ext/init":
                    res: Any = {
                        "protocolVersion": "carnot-ext/1",
                        "tools": self.tools,
                        "hooks": self.hooks,
                        "timeout_ms": self.timeout_ms,
                        "failure_policy": self.failure_policy,
                    }
                elif method == "ext/session_start":
                    res = {"context": self.on_session_start(params.get("workspace_root", "."))}
                elif method == "ext/before_tool":
                    res = self.before_tool(params.get("tool", ""), params.get("input") or {})
                elif method == "ext/after_tool":
                    self.after_tool(
                        params.get("tool", ""),
                        params.get("input") or {},
                        bool(params.get("is_error", False)),
                        str(params.get("text", "")),
                    )
                    res = {}
                elif method == "ext/turn_boundary":
                    res = self.on_turn_boundary(params)
                elif method == "ext/tool_call":
                    res = self.call_tool(params.get("name", ""), params.get("input") or {})
                else:
                    res = {}
                out = {"jsonrpc": "2.0", "id": rid, "result": res}
            except Exception as exc:
                out = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(exc)}}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
