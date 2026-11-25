"""
Zero-dependency Python 3.12+ async SDK for Carnot (`carnot --mode rpc`) and drop-in extensions (`carnot-ext/1`).

Usage:
    async with CarnotClient(cwd=".") as c:
        res = await c.prompt("Fix the failing tests")
        print(res["final_text"], res["metrics"])
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from typing import Any, Callable


class CarnotClient:
    """Async JSON-RPC 2.0 client driving `carnot --mode rpc` over stdio."""

    def __init__(
        self,
        binary: str = "carnot",
        cwd: str = ".",
        provider: str | None = None,
        model: str | None = None,
        profile: str | None = None,
        effort: str | None = None,
        cache_ttl: str | None = None,
        no_session: bool = False,
        extra_args: list[str] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.binary = binary
        self.cwd = cwd
        self.provider = provider
        self.model = model
        self.profile = profile
        self.effort = effort
        self.cache_ttl = cache_ttl
        self.no_session = no_session
        self.extra_args = list(extra_args or [])
        self.on_event = on_event
        self._proc: asyncio.subprocess.Process | None = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "CarnotClient":
        args = ["--mode", "rpc", "--cwd", self.cwd]
        if self.provider:
            args.extend(["--provider", self.provider])
        if self.model:
            args.extend(["--model", self.model])
        if self.profile:
            args.extend(["--profile", self.profile])
        if self.effort:
            args.extend(["--effort", self.effort])
        if self.cache_ttl:
            args.extend(["--cache-ttl", self.cache_ttl])
        if self.no_session:
            args.append("--no-session")
        args.extend(self.extra_args)

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
        if self._proc and self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.close()
        if self._reader_task:
            self._reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader_task
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except TimeoutError:
                self._proc.kill()
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

    async def ping(self) -> dict[str, Any]:
        return await self.call("ping")

    async def prompt(
        self,
        message: str,
        images: list[dict[str, str]] | None = None,
        *,
        ack: bool = False,
        streaming_behavior: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"message": message}
        if images:
            params["images"] = images
        if ack:
            params["ack"] = True
        if streaming_behavior:
            params["streaming_behavior"] = streaming_behavior
        return await self.call("prompt", params)

    async def continue_turn(self) -> dict[str, Any]:
        return await self.call("continue")

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

    async def get_entries(self, custom_type: str | None = None) -> dict[str, Any]:
        params = {"custom_type": custom_type} if custom_type else {}
        return await self.call("get_entries", params)

    async def append_entry(self, custom_type: str, data: Any) -> dict[str, Any]:
        return await self.call("append_entry", {"custom_type": custom_type, "data": data})

    async def set_model(self, model: str, provider: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"model": model}
        if provider:
            params["provider"] = provider
        return await self.call("set_model", params)

    async def set_effort(self, effort: str) -> dict[str, Any]:
        return await self.call("set_effort", {"effort": effort})

    async def set_cache_ttl(self, ttl: str) -> dict[str, Any]:
        """Set prompt-cache retention tier (`short`, `hybrid`, or `long`)."""
        return await self.call("set_cache_ttl", {"ttl": ttl})

    async def get_available_models(self) -> dict[str, Any]:
        return await self.call("get_available_models")

    async def get_commands(self) -> dict[str, Any]:
        return await self.call("get_commands")

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
    """

    failure_policy: str = "fail_closed"
    timeout_ms: int | None = None

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

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        rid = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}
        try:
            if method == "ext/init":
                normalized_tools = [
                    {"defer_loading": True, **tool} for tool in self.tools()
                ]
                result: Any = {
                    "protocolVersion": "carnot-ext/1",
                    "failure_policy": self.failure_policy,
                    "hooks": self._advertised_hooks(),
                    "tools": normalized_tools,
                }
                if self.timeout_ms is not None:
                    result["timeout_ms"] = self.timeout_ms
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
        return resp if rid is not None else None

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
            resp = self.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
