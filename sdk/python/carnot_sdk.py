"""
Zero-dependency Python 3.12+ async SDK for Carnot (`carnot --mode rpc`).

Usage:
    async with CarnotClient(cwd=".", provider="anthropic", model="claude-sonnet-5-5") as c:
        res = await c.prompt("Fix the failing tests")
        print(res["final_text"], res["metrics"])
"""
from __future__ import annotations

import asyncio
import json
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

    async def prompt(
        self,
        message: str,
        images: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"message": message}
        if images:
            params["images"] = images
        return await self.call("prompt", params)

    async def steer(self, message: str) -> dict[str, Any]:
        return await self.call("steer", {"message": message})

    async def follow_up(self, message: str) -> dict[str, Any]:
        return await self.call("follow_up", {"message": message})

    async def compact(self, focus: str | None = None) -> dict[str, Any]:
        params = {"focus": focus} if focus else {}
        return await self.call("compact", params)

    async def undo(self) -> dict[str, Any]:
        return await self.call("undo")

    async def replay_events(self, after_seq: int = 0) -> dict[str, Any]:
        return await self.call("replay_events", {"after_seq": after_seq})

    async def get_state(self) -> dict[str, Any]:
        return await self.call("get_state")

    async def get_metrics(self) -> dict[str, Any]:
        return await self.call("get_metrics")

    async def get_session_stats(self) -> dict[str, Any]:
        return await self.call("get_session_stats")

    async def set_cache_ttl(self, ttl: str) -> dict[str, Any]:
        """Set prompt-cache retention tier (`short`, `hybrid`, or `long`)."""
        return await self.call("set_cache_ttl", {"ttl": ttl})

    async def list_pending(self) -> dict[str, Any]:
        return await self.call("list_pending")

    async def set_require_approval(self, enabled: bool = True) -> dict[str, Any]:
        return await self.call("set_require_approval", {"enabled": enabled})

    async def respond_permission(
        self, request_id: str, decision: str = "allow_once", reason: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"request_id": request_id, "decision": decision}
        if reason is not None:
            params["reason"] = reason
        return await self.call("respond_permission", params)

    async def respond_question(
        self, request_id: str, answers: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self.call("respond_question", {"request_id": request_id, "answers": answers})

    async def abort(self) -> dict[str, Any]:
        return await self.call("abort")


class CarnotExtension:
    """
    Base class for drop-in Carnot file extensions (`carnot-ext/1` over stdio).

    Drop an executable script into `~/.carnot/extensions/` or pass `--ext ./my_ext.py`.
    Only hooks you actually override are advertised in `ext/init` (zero IPC on unused hooks),
    and tools default to `defer_loading: True` (0 static prefix tokens).

    Example:
        #!/usr/bin/env python3
        from carnot_sdk import CarnotExtension

        class GuardExt(CarnotExtension):
            def before_tool(self, tool: str, inp: dict[str, Any]) -> dict[str, Any]:
                if tool == "Bash" and "deploy --prod" in inp.get("command", ""):
                    return {"decision": "deny", "reason": "prod deploy requires manual approval"}
                return {"decision": "allow"}

        if __name__ == "__main__":
            GuardExt().run()
    """

    failure_policy: str = "fail_closed"

    def tools(self) -> list[dict[str, Any]]:
        """Return tool definitions: `[{"name", "description", "input_schema", "defer_loading": True}]`."""
        return []

    def session_start(self, workspace_root: str, model: str) -> str | None:
        return None

    def before_tool(self, tool: str, inp: dict[str, Any]) -> dict[str, Any]:
        return {"decision": "allow"}

    def after_tool(
        self, tool: str, inp: dict[str, Any], is_error: bool, text: str
    ) -> dict[str, Any]:
        return {}

    def turn_boundary(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {"action": "continue"}

    def before_compact(self, ctx: dict[str, Any]) -> dict[str, Any]:
        return {"action": "continue"}

    def call_tool(self, name: str, inp: dict[str, Any]) -> dict[str, Any]:
        return {"is_error": True, "text": f"unknown extension tool: {name}"}

    def _advertised_hooks(self) -> list[str]:
        hooks: list[str] = []
        for method_name, hook_name in (
            ("session_start", "session_start"),
            ("before_tool", "before_tool"),
            ("after_tool", "after_tool"),
            ("turn_boundary", "turn_boundary"),
            ("before_compact", "before_compact"),
        ):
            if getattr(type(self), method_name) is not getattr(CarnotExtension, method_name):
                hooks.append(hook_name)
        return hooks

    def run(self) -> None:
        import sys

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = req.get("id")
            method = req.get("method", "")
            params = req.get("params") or {}
            try:
                if method == "ext/init":
                    normalized_tools = [
                        {"defer_loading": True, **tool} for tool in self.tools()
                    ]
                    result: Any = {
                        "protocol": "carnot-ext/1",
                        "failure_policy": self.failure_policy,
                        "hooks": self._advertised_hooks(),
                        "tools": normalized_tools,
                    }
                elif method == "ext/session_start":
                    ctx = self.session_start(
                        str(params.get("workspace_root", "")),
                        str(params.get("model", "")),
                    )
                    result = {"context": ctx} if ctx else {}
                elif method == "ext/before_tool":
                    result = self.before_tool(
                        str(params.get("tool", "")),
                        dict(params.get("input") or {}),
                    )
                elif method == "ext/after_tool":
                    result = self.after_tool(
                        str(params.get("tool", "")),
                        dict(params.get("input") or {}),
                        bool(params.get("is_error", False)),
                        str(params.get("text", "")),
                    )
                elif method == "ext/turn_boundary":
                    result = self.turn_boundary(dict(params))
                elif method == "ext/before_compact":
                    result = self.before_compact(dict(params))
                elif method == "ext/tool_call":
                    result = self.call_tool(
                        str(params.get("name", "")),
                        dict(params.get("input") or {}),
                    )
                else:
                    result = {}
                resp: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "result": result}
            except Exception as exc:  # noqa: BLE001
                resp = {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {"code": -32000, "message": str(exc)},
                }
            if rid is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
