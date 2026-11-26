#!/usr/bin/env python3
"""
Fair, Deterministic Cross-Harness Benchmark Suite.

Compares:
  - Carnot (Rust)
  - Unharnessed (Rust)
  - Pi (@mariozechner/pi-coding-agent, TypeScript/Node)
  - OpenCode (Bun/TypeScript compiled binary)
  - Claude Code (Anthropic CLI)
  - Codex CLI (OpenAI Responses CLI — clearly marked in its own OpenAI protocol column)

Methodology:
  1. Isolated empty HOME + isolated git scratch workspace (no user CLAUDE.md/AGENTS.md,
     no MCP, no skills, no extensions, telemetry/auto-update disabled).
  2. Local deterministic HTTP/1.1 mock server on 127.0.0.1 supporting:
     - Anthropic Messages SSE (`/v1/messages`)
     - OpenAI Responses SSE (`/v1/responses`) & Chat Completions SSE (`/v1/chat/completions`)
  3. Measures:
     - Full installed footprint (binary + runtime closure)
     - Warm startup-to-first-HTTP-byte latency (warmup=3, runs=15 -> median & p95)
     - Full process-tree peak RSS (parent + children summed every 20ms + rusage)
     - Exact Turn-1 static payload: system prompt bytes, tool count, tool schema bytes,
       total Turn-1 JSON bytes, estimated static tokens, cache_control presence
     - Cross-cwd / cross-timestamp static prefix invariance (byte-identical check!)
     - Extra side-requests count
     - 2-Turn Noisy-Tool Context Discipline: mock triggers a tool call returning 12 KB of
       noisy output and measures how many bytes the harness sends back on Turn 2!
"""

import http.server
import json
import os
import platform
import shutil
import socketserver
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Deterministic Local Mock Server (Anthropic + OpenAI Responses + Completions)
# ---------------------------------------------------------------------------

class MockState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset("single")

    def reset(self, scenario: str = "single", tool_name: str = "Bash", tool_input: Optional[dict] = None):
        with self.lock:
            self.scenario = scenario
            self.tool_name = tool_name
            self.tool_input = tool_input or {"command": "python3 -c \"print('PASS line ' * 1200)\""}
            self.first_byte_ts: Optional[float] = None
            self.requests: List[Dict[str, Any]] = []
            self.turn_counter = 0

    def record_request(self, path: str, headers: Dict[str, str], body_bytes: bytes, arrival_ts: float) -> int:
        with self.lock:
            try:
                parsed = json.loads(body_bytes.decode("utf-8", errors="replace"))
            except Exception:
                parsed = {}
            path_l = path.lower()
            is_main_prompt = (
                "count_tokens" not in path_l
                and (
                    ("/responses" in path_l)
                    or ("/chat/completions" in path_l)
                    or (
                        "/messages" in path_l
                        and isinstance(parsed.get("tools"), list)
                        and len(parsed.get("tools", [])) > 0
                    )
                )
            )
            if is_main_prompt and self.first_byte_ts is None:
                self.first_byte_ts = arrival_ts
            self.turn_counter += 1
            turn_idx = self.turn_counter
            self.requests.append({
                "path": path,
                "headers": headers,
                "body_len": len(body_bytes),
                "body": parsed,
                "arrival_ts": arrival_ts,
                "turn_idx": turn_idx,
                "is_main_prompt": is_main_prompt,
            })
            return turn_idx


MOCK_STATE = MockState()


def anthropic_text_sse(text: str = "OK") -> bytes:
    events = [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_bench_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 120,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 1,
                },
            },
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 4},
        }),
        ("message_stop", {"type": "message_stop"}),
    ]
    out = []
    for ev_name, payload in events:
        out.append(f"event: {ev_name}\ndata: {json.dumps(payload)}\n\n")
    return "".join(out).encode("utf-8")


def anthropic_tool_sse(tool_name: str, tool_input: dict) -> bytes:
    events = [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_bench_t1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 120, "output_tokens": 1},
            },
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_bench_01", "name": tool_name, "input": {}},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 18},
        }),
        ("message_stop", {"type": "message_stop"}),
    ]
    out = []
    for ev_name, payload in events:
        out.append(f"event: {ev_name}\ndata: {json.dumps(payload)}\n\n")
    return "".join(out).encode("utf-8")


def openai_responses_text_sse(text: str = "OK") -> bytes:
    resp_obj = {
        "id": "resp_bench_01",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.4",
        "output": [{
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 124,
        },
    }
    events = [
        ("response.created", {"type": "response.created", "response": {"id": "resp_bench_01", "status": "in_progress"}}),
        ("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_01", "type": "message", "role": "assistant", "status": "in_progress", "content": []},
        }),
        ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_01", "output_index": 0, "content_index": 0, "delta": text}),
        ("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": resp_obj["output"][0],
        }),
        ("response.completed", {"type": "response.completed", "response": resp_obj}),
    ]
    out = []
    for ev_name, payload in events:
        out.append(f"event: {ev_name}\ndata: {json.dumps(payload)}\n\n")
    return "".join(out).encode("utf-8")


def openai_chat_text_sse(text: str = "OK") -> bytes:
    chunks = [
        {"id": "chatcmpl_1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
        {"id": "chatcmpl_1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 120, "completion_tokens": 4, "total_tokens": 124}},
    ]
    out = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MockAPIHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        # TLS/HTTP prewarm probe — do NOT count as a prompt request!
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        body = b'{"data":[{"id":"claude-sonnet-4-5-20250929"},{"id":"claude-sonnet-5-5"},{"id":"gpt-5.4"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        arrival_ts = time.perf_counter()
        clen = int(self.headers.get("Content-Length", "0"))
        if clen > 0:
            body_bytes = self.rfile.read(clen)
        elif self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while True:
                size_line = self.rfile.readline().strip()
                if not size_line:
                    break
                sz = int(size_line.split(b";")[0], 16)
                if sz == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(sz))
                self.rfile.readline()
            body_bytes = b"".join(chunks)
        else:
            body_bytes = b""

        hdrs = {k.lower(): v for k, v in self.headers.items()}
        # Ignore Anthropic count_tokens / event_logging side-calls for first-byte if needed,
        # but STILL record them in requests so we report side-requests transparently!
        turn_idx = MOCK_STATE.record_request(self.path, hdrs, body_bytes, arrival_ts)

        path_l = self.path.lower()
        if "count_tokens" in path_l:
            resp = b'{"input_tokens":100}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        if "/responses" in path_l:
            sse = openai_responses_text_sse("OK")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)
            return

        if "/chat/completions" in path_l:
            sse = openai_chat_text_sse("OK")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)
            return

        # Anthropic `/v1/messages`
        parsed = {}
        try:
            parsed = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except Exception:
            pass
        is_stream = parsed.get("stream", True)
        if not is_stream:
            resp_json = json.dumps({
                "id": "msg_nonstream_1",
                "type": "message",
                "role": "assistant",
                "model": parsed.get("model", "claude-sonnet-4-5-20250929"),
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 2},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_json)))
            self.end_headers()
            self.wfile.write(resp_json)
            return

        # Multi-turn noisy-tool scenario: trigger tool call on the first main request that declares tools!
        declared_tools = [t.get("name", "") for t in parsed.get("tools", []) if isinstance(t, dict)]
        has_prior_tool_result = '"tool_result"' in json.dumps(parsed.get("messages", []))
        if MOCK_STATE.scenario == "noisy_tool" and declared_tools and not has_prior_tool_result:
            chosen_tool, chosen_input = pick_noisy_tool(declared_tools)
            if chosen_tool:
                sse = anthropic_tool_sse(chosen_tool, chosen_input)
            else:
                sse = anthropic_text_sse("OK")
        else:
            sse = anthropic_text_sse("OK")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


def pick_noisy_tool(declared_tools: List[str]) -> Tuple[Optional[str], dict]:
    """Selects a shell/python execution tool from the harness's declared tool schemas."""
    cmd = "python3 -c \"print('ERROR line_idx=' + str(1) + ' ' + 'x'*80); [print(f'INFO noise line {i}: ' + 'a'*90) for i in range(160)]\""
    for name in declared_tools:
        nl = name.lower()
        if nl == "bash":
            return name, {"command": cmd}
        if nl == "shell":
            return name, {"command": cmd}
    for name in declared_tools:
        nl = name.lower()
        if "python" in nl or "repl" in nl or "exec" in nl:
            return name, {"code": "print('ERROR line_idx=1 ' + 'x'*80)\nfor i in range(160): print(f'INFO noise line {i}: ' + 'a'*90)"}
    return None, {}


# ---------------------------------------------------------------------------
# Process-Tree Peak RSS Sampler (macOS accurate: sums parent + all children!)
# ---------------------------------------------------------------------------

def sample_tree_rss_kb(root_pid: int) -> int:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,rss="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return 0
    children: Dict[int, List[int]] = {}
    rss_map: Dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, rss = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        rss_map[pid] = rss
        children.setdefault(ppid, []).append(pid)
    if root_pid not in rss_map:
        return 0
    total = 0
    stack = [root_pid]
    seen = set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        total += rss_map.get(p, 0)
        stack.extend(children.get(p, []))
    return total


def run_with_metrics(cmd: List[str], env: Dict[str, str], cwd: str, timeout_s: float = 25.0) -> Dict[str, Any]:
    """Runs command wrapped in `/usr/bin/time -l` (for exact kernel maxrss on fast binaries)
    AND samples the full process tree every 20ms (for multi-process harnesses)."""
    t0 = time.perf_counter()
    # We run the binary directly (not wrapped in /usr/bin/time) so launch-to-first-HTTP-byte has zero wrapper overhead!
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak_tree_kb = 0
    stop_flag = threading.Event()

    def sampler():
        nonlocal peak_tree_kb
        while not stop_flag.is_set():
            kb = sample_tree_rss_kb(proc.pid)
            if kb > peak_tree_kb:
                peak_tree_kb = kb
            time.sleep(0.015)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        rc = -9
    finally:
        stop_flag.set()
        t.join(timeout=0.5)

    t1 = time.perf_counter()
    first_byte_ms = None
    if MOCK_STATE.first_byte_ts is not None:
        first_byte_ms = (MOCK_STATE.first_byte_ts - t0) * 1000.0
    return {
        "rc": rc,
        "wall_ms": (t1 - t0) * 1000.0,
        "first_http_ms": first_byte_ms,
        "peak_tree_mib": peak_tree_kb / 1024.0,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "requests": list(MOCK_STATE.requests),
    }


def measure_kernel_maxrss_mib(cmd: List[str], env: Dict[str, str], cwd: str, timeout_s: float = 25.0) -> float:
    """Uses macOS `/usr/bin/time -l` (reports bytes!) as a floor for fast processes that exit < 15ms."""
    try:
        res = subprocess.run(
            ["/usr/bin/time", "-l"] + cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            text=True,
        )
        for line in res.stderr.splitlines():
            if "maximum resident set size" in line:
                val = int(line.strip().split()[0])
                # macOS reports bytes, Linux reports KiB
                if platform.system() == "Darwin":
                    return val / (1024.0 * 1024.0)
                return val / 1024.0
    except Exception:
        pass
    return 0.0


def analyze_turn1_payload(requests: List[Dict[str, Any]], protocol: str) -> Dict[str, Any]:
    """Extracts apples-to-apples static prefix metrics from the first main model request."""
    main_reqs = [r for r in requests if r.get("is_main_prompt")]
    if not main_reqs:
        return {"ok": False, "http_requests": len(requests)}

    req = main_reqs[0]
    body = req["body"]
    if protocol == "openai-responses":
        sys_parts = []
        if body.get("instructions"):
            sys_parts.append(body.get("instructions"))
        for item in body.get("input", []):
            if isinstance(item, dict) and item.get("role") in ("system", "developer"):
                sys_parts.append(item)
        sys_bytes = len(json.dumps(sys_parts, ensure_ascii=False).encode("utf-8"))
        tools = body.get("tools", [])
        sys_canon_obj = sys_parts
    elif protocol == "openai-completions":
        msgs = body.get("messages", [])
        sys_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "developer")]
        sys_bytes = len(json.dumps(sys_msgs, ensure_ascii=False).encode("utf-8"))
        tools = body.get("tools", [])
        sys_canon_obj = sys_msgs
    else:
        sys_raw = body.get("system", "")
        sys_bytes = len(json.dumps(sys_raw, ensure_ascii=False).encode("utf-8"))
        tools = body.get("tools", [])
        sys_canon_obj = sys_raw

    tools_bytes = len(json.dumps(tools, ensure_ascii=False).encode("utf-8"))
    static_bytes = sys_bytes + tools_bytes
    raw_json_str = json.dumps(body, sort_keys=True, ensure_ascii=False)
    has_cache_control = '"cache_control"' in raw_json_str
    has_prompt_cache_key = '"prompt_cache_key"' in raw_json_str

    return {
        "ok": True,
        "http_requests": len(requests),
        "main_requests": len(main_reqs),
        "turn1_total_bytes": req["body_len"],
        "system_bytes": sys_bytes,
        "tools_count": len(tools) if isinstance(tools, list) else 0,
        "tools_bytes": tools_bytes,
        "static_prefix_bytes": static_bytes,
        "static_prefix_est_tokens": round(static_bytes / 4.0),
        "has_cache_control": has_cache_control or has_prompt_cache_key,
        "system_canonical": json.dumps(sys_canon_obj, sort_keys=True),
        "tools_canonical": json.dumps(tools, sort_keys=True),
    }


def dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024.0 * 1024.0)
    if not path.exists():
        return 0.0
    out = subprocess.check_output(["du", "-sk", str(path)], text=True)
    kb = int(out.split()[0])
    return kb / 1024.0


def resolve_real_file_mb(bin_path: str) -> float:
    p = Path(bin_path).resolve()
    return p.stat().st_size / (1024.0 * 1024.0) if p.is_file() else 0.0


# ---------------------------------------------------------------------------
# Harness Definitions
# ---------------------------------------------------------------------------

def build_harnesses(port: int) -> List[Dict[str, Any]]:
    base_url = f"http://127.0.0.1:{port}"
    repo_root = Path(__file__).resolve().parent.parent
    carnot_bin = os.environ.get("CARNOT_BIN") or str(repo_root / "target/release/carnot")
    uh_bin = os.environ.get("UNHARNESSED_BIN") or (shutil.which("unharnessed") or "")
    pi_bin = os.environ.get("PI_BIN") or (shutil.which("pi") or "")
    opencode_bin = shutil.which("opencode") or ""
    claude_bin = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    codex_bin = shutil.which("codex") or ""
    node_bin = shutil.which("node") or ""

    # Compute exact installed closures
    pi_node_modules = Path(pi_bin).resolve().parent.parent if pi_bin and Path(pi_bin).exists() else Path("/nonexistent")
    pi_closure_mb = (dir_size_mb(pi_node_modules) if pi_node_modules.exists() else 0.0) + (resolve_real_file_mb(node_bin) if node_bin else 0.0)
    opencode_real = Path(opencode_bin).resolve()
    # OpenCode Homebrew binary is often a standalone compiled executable or inside a package dir
    opencode_bin_mb = resolve_real_file_mb(opencode_bin)
    if opencode_bin_mb < 1.0:
        opencode_closure_mb = dir_size_mb(opencode_real.parent.parent)
    else:
        opencode_closure_mb = opencode_bin_mb

    claude_real = Path(claude_bin).resolve()
    claude_bin_mb = resolve_real_file_mb(claude_bin)
    claude_closure_mb = claude_bin_mb if claude_bin_mb >= 1.0 else dir_size_mb(claude_real.parent.parent) + resolve_real_file_mb(node_bin)

    codex_bin_mb = resolve_real_file_mb(codex_bin)
    if codex_bin_mb < 1.0:
        codex_closure_mb = dir_size_mb(Path(codex_bin).resolve().parent.parent)
    else:
        codex_closure_mb = codex_bin_mb

    def base_env(home_dir: str) -> Dict[str, str]:
        e = {
            "PATH": os.environ.get("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"),
            "HOME": home_dir,
            "TMPDIR": home_dir,
            "XDG_CONFIG_HOME": os.path.join(home_dir, ".config"),
            "XDG_DATA_HOME": os.path.join(home_dir, ".local", "share"),
            "XDG_CACHE_HOME": os.path.join(home_dir, ".cache"),
            "XDG_STATE_HOME": os.path.join(home_dir, ".local", "state"),
            "TERM": "dumb",
            "NO_COLOR": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "PI_OFFLINE": "1",
            "ANTHROPIC_API_KEY": "sk-ant-bench-dummy",
            "ANTHROPIC_BASE_URL": base_url,
            "OPENAI_API_KEY": "sk-openai-bench-dummy",
            "OPENAI_BASE_URL": f"{base_url}/v1",
        }
        return e

    def setup_pi_home(home_dir: str):
        agent_dir = Path(home_dir) / ".pi" / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        models_cfg = {
            "providers": {
                "anthropic": {
                    "baseUrl": base_url,
                }
            }
        }
        (agent_dir / "models.json").write_text(json.dumps(models_cfg))

    def setup_uh_home(home_dir: str):
        agent_dir = Path(home_dir) / ".unharnessed"
        agent_dir.mkdir(parents=True, exist_ok=True)
        models_cfg = {
            "providers": {
                "anthropic": {
                    "baseUrl": base_url,
                }
            }
        }
        (agent_dir / "models.json").write_text(json.dumps(models_cfg))

    def setup_opencode_home(home_dir: str):
        cfg_dir = Path(home_dir) / ".config" / "opencode"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "$schema": "https://opencode.ai/config.json",
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "autoupdate": False,
            "provider": {
                "anthropic": {
                    "options": {
                        "baseURL": f"{base_url}/v1",
                        "apiKey": "sk-ant-bench-dummy",
                    },
                    "models": {
                        "claude-sonnet-4-5-20250929": {
                            "name": "Claude Sonnet 4.5 (Bench)",
                            "tool_call": True,
                            "limit": {"context": 200000, "output": 8192},
                        }
                    },
                }
            },
        }
        (cfg_dir / "opencode.json").write_text(json.dumps(cfg))

    def setup_codex_home(home_dir: str):
        codex_dir = Path(home_dir) / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        (codex_dir / "config.toml").write_text(
            f'model = "gpt-5.4"\nmodel_provider = "bench"\n\n'
            f'[model_providers.bench]\nname = "bench"\nbase_url = "{base_url}/v1"\n'
            f'env_key = "OPENAI_API_KEY"\nwire_api = "responses"\n'
        )

    harnesses = [
        {
            "id": "carnot",
            "label": "Carnot (Rust, standalone)",
            "protocol": "anthropic-messages",
            "binary_mb": round(resolve_real_file_mb(carnot_bin), 1),
            "closure_mb": round(resolve_real_file_mb(carnot_bin), 1),
            "cmd": lambda cwd: [
                carnot_bin, "--provider", "anthropic", "--model", "claude-sonnet-4-5-20250929",
                "--no-session", "-C", cwd, "-p", "Reply OK",
            ],
            "env": base_env,
            "setup_home": None,
        },
        {
            "id": "carnot-core",
            "label": "Carnot (Rust, 5-tool core)",
            "protocol": "anthropic-messages",
            "binary_mb": round(resolve_real_file_mb(carnot_bin), 1),
            "closure_mb": round(resolve_real_file_mb(carnot_bin), 1),
            "cmd": lambda cwd: [
                carnot_bin, "--provider", "anthropic", "--model", "claude-sonnet-4-5-20250929",
                "--no-session", "--no-web", "--no-lsp", "--no-memory", "--no-subagent", "--no-codemode",
                "-C", cwd, "-p", "Reply OK",
            ],
            "env": base_env,
            "setup_home": None,
        },
        {
            "id": "unharnessed",
            "label": "Unharnessed (Rust)",
            "protocol": "anthropic-messages",
            "binary_mb": round(resolve_real_file_mb(uh_bin), 1),
            "closure_mb": round(resolve_real_file_mb(uh_bin), 1),
            "cmd": lambda cwd: [
                uh_bin, "-p", "--provider", "anthropic", "--model", "claude-sonnet-4-5-20250929",
                "--no-session", "--no-skills", "--no-context-files", "--offline", "--cwd", cwd, "Reply OK",
            ],
            "env": lambda h: {**base_env(h), "UNHARNESSED_CODING_AGENT_DIR": os.path.join(h, ".unharnessed")},
            "setup_home": setup_uh_home,
            "warmup_timeout_s": 90.0,
        },
        {
            "id": "pi",
            "label": "Pi (TS/Node)",
            "protocol": "anthropic-messages",
            "binary_mb": round(dir_size_mb(pi_node_modules) if pi_node_modules.exists() else 0.0, 1),
            "closure_mb": round(pi_closure_mb, 1),
            "cmd": lambda cwd: [
                pi_bin, "-p", "--provider", "anthropic", "--model", "claude-sonnet-4-5-20250929",
                "--no-session", "--no-extensions", "--no-skills", "--no-prompt-templates",
                "--no-themes", "--no-context-files", "--offline", "Reply OK",
            ],
            "env": base_env,
            "setup_home": setup_pi_home,
        },
        {
            "id": "opencode",
            "label": "OpenCode (Bun/TS)",
            "protocol": "anthropic-messages",
            "binary_mb": round(opencode_bin_mb, 1),
            "closure_mb": round(opencode_closure_mb, 1),
            "cmd": lambda cwd: [
                opencode_bin, "run", "--standalone", "--auto",
                "--model", "anthropic/claude-sonnet-4-5-20250929", "Reply OK",
            ],
            "env": base_env,
            "setup_home": setup_opencode_home,
        },
        {
            "id": "claude",
            "label": "Claude Code",
            "protocol": "anthropic-messages",
            "binary_mb": round(claude_bin_mb, 1),
            "closure_mb": round(claude_closure_mb, 1),
            "cmd": lambda cwd: [
                claude_bin, "-p", "--model", "claude-sonnet-4-5-20250929",
                "--no-session-persistence", "--dangerously-skip-permissions", "Reply OK",
            ],
            "env": base_env,
            "setup_home": None,
        },
        {
            "id": "codex",
            "label": "Codex CLI (Responses)",
            "protocol": "openai-responses",
            "binary_mb": round(codex_bin_mb, 1),
            "closure_mb": round(codex_closure_mb, 1),
            "cmd": lambda cwd: [
                codex_bin, "exec", "--skip-git-repo-check", "--ephemeral",
                "-C", cwd, "Reply OK",
            ],
            "env": base_env,
            "setup_home": setup_codex_home,
        },
    ]
    return harnesses


def make_git_workspace(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(path), check=True)
    (path / "hello.rs").write_text("pub fn answer() -> i32 { 42 }\n")


def main():
    runs = int(os.environ.get("BENCH_RUNS", "15"))
    warmups = int(os.environ.get("BENCH_WARMUPS", "3"))

    server = ThreadedHTTPServer(("127.0.0.1", 0), MockAPIHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    harnesses = build_harnesses(port)
    results = []

    with tempfile.TemporaryDirectory(prefix="carnot-bench-ws-") as tmp_root:
        ws_a = Path(tmp_root) / "repo-alpha"
        ws_b = Path(tmp_root) / "repo-beta"
        make_git_workspace(ws_a)
        make_git_workspace(ws_b)

        for h in harnesses:
            hid = h["id"]
            cmd_a = h["cmd"](str(ws_a))
            cmd_b = h["cmd"](str(ws_b))
            exe = cmd_a[0] if cmd_a else ""
            if not exe or not (os.path.isfile(exe) or shutil.which(exe)):
                print(f"\n>>> Skipping {h['label']} ({hid}): binary not installed.", file=sys.stderr, flush=True)
                continue
            print(f"\n>>> Benchmarking {h['label']} ({hid}) ...", file=sys.stderr, flush=True)
            home_dir = Path(tmp_root) / f"home-{hid}"
            home_dir.mkdir(parents=True, exist_ok=True)
            if h["setup_home"]:
                h["setup_home"](str(home_dir))
            env = h["env"](str(home_dir))

            # 1) Probe + Warmups (also filters out Gatekeeper/first-run cold cost & kernel download)
            probe_ok = False
            last_probe = None
            w_timeout = float(h.get("warmup_timeout_s", 30.0))
            for w in range(warmups):
                MOCK_STATE.reset("single")
                res = run_with_metrics(cmd_a, env, str(ws_a), timeout_s=w_timeout)
                last_probe = res
                if res["first_http_ms"] is not None:
                    probe_ok = True
            if not probe_ok:
                print(
                    f"  [SKIP/WARN] {hid} did not hit mock server (rc={last_probe['rc'] if last_probe else '?'}). "
                    f"stderr={(last_probe['stderr'][:240] if last_probe else '')!r}",
                    file=sys.stderr,
                    flush=True,
                )
                results.append({
                    "id": hid,
                    "label": h["label"],
                    "protocol": h["protocol"],
                    "binary_mb": h["binary_mb"],
                    "closure_mb": h["closure_mb"],
                    "status": "probe_failed",
                    "rc": last_probe["rc"] if last_probe else -1,
                    "stderr_snippet": (last_probe["stderr"][:300] if last_probe else ""),
                })
                continue

            # 2) Timed Warm Runs (N=runs) -> Median & p95 startup-to-first-HTTP-byte + RSS
            first_http_samples: List[float] = []
            wall_samples: List[float] = []
            rss_samples: List[float] = []
            turn1_a = None

            for i in range(runs):
                MOCK_STATE.reset("single")
                r = run_with_metrics(cmd_a, env, str(ws_a), timeout_s=25.0)
                if r["rc"] == 0 and r["first_http_ms"] is not None:
                    first_http_samples.append(r["first_http_ms"])
                    wall_samples.append(r["wall_ms"])
                    if r["peak_tree_mib"] > 0:
                        rss_samples.append(r["peak_tree_mib"])
                    if turn1_a is None:
                        turn1_a = analyze_turn1_payload(r["requests"], h["protocol"])

            # Kernel maxrss check (catches fast Rust binaries that exit before 15ms poll!)
            MOCK_STATE.reset("single")
            kernel_rss_mib = measure_kernel_maxrss_mib(cmd_a, env, str(ws_a), timeout_s=25.0)
            peak_rss_mib = max(max(rss_samples) if rss_samples else 0.0, kernel_rss_mib)

            # 3) Cross-CWD Static Prefix Invariance Check (run in ws_b 1s later!)
            time.sleep(1.05)
            MOCK_STATE.reset("single")
            r_b = run_with_metrics(cmd_b, env, str(ws_b), timeout_s=25.0)
            turn1_b = analyze_turn1_payload(r_b["requests"], h["protocol"])
            prefix_invariant = (
                bool(turn1_a and turn1_b and turn1_a.get("ok") and turn1_b.get("ok"))
                and turn1_a["system_canonical"] == turn1_b["system_canonical"]
                and turn1_a["tools_canonical"] == turn1_b["tools_canonical"]
            )

            # 4) 2-Turn Noisy Tool Output Discipline (Anthropic protocol harnesses)
            # Also verifies quality: critical `ERROR line_idx=1` MUST be preserved in Turn 2!
            noisy_turn2_delta_bytes = None
            noisy_error_preserved = None
            if h["protocol"] == "anthropic-messages":
                MOCK_STATE.reset("noisy_tool")
                r_noisy = run_with_metrics(cmd_a, env, str(ws_a), timeout_s=30.0)
                main_reqs = [req for req in r_noisy["requests"] if req.get("is_main_prompt")]
                t2_req = next(
                    (req for req in main_reqs[1:] if '"tool_result"' in json.dumps(req["body"])),
                    None,
                )
                if main_reqs and t2_req is not None:
                    noisy_turn2_delta_bytes = t2_req["body_len"] - main_reqs[0]["body_len"]
                    noisy_error_preserved = "ERROR line_idx=1" in json.dumps(t2_req["body"])

            first_http_samples.sort()
            wall_samples.sort()
            p95_idx = min(len(first_http_samples) - 1, int(0.95 * len(first_http_samples)))

            entry = {
                "id": hid,
                "label": h["label"],
                "protocol": h["protocol"],
                "status": "ok",
                "runs": len(first_http_samples),
                "binary_mb": h["binary_mb"],
                "closure_mb": h["closure_mb"],
                "startup_first_http_median_ms": round(statistics.median(first_http_samples), 1),
                "startup_first_http_p95_ms": round(first_http_samples[p95_idx], 1),
                "wall_median_ms": round(statistics.median(wall_samples), 1),
                "peak_rss_mib": round(peak_rss_mib, 1),
                "http_requests_per_prompt": turn1_a["http_requests"] if turn1_a else 0,
                "system_bytes": turn1_a["system_bytes"] if turn1_a else 0,
                "tools_count": turn1_a["tools_count"] if turn1_a else 0,
                "tools_bytes": turn1_a["tools_bytes"] if turn1_a else 0,
                "static_prefix_bytes": turn1_a["static_prefix_bytes"] if turn1_a else 0,
                "static_prefix_est_tokens": turn1_a["static_prefix_est_tokens"] if turn1_a else 0,
                "turn1_total_bytes": turn1_a["turn1_total_bytes"] if turn1_a else 0,
                "has_cache_control": turn1_a["has_cache_control"] if turn1_a else False,
                "cross_cwd_prefix_invariant": prefix_invariant,
                "noisy_16kb_tool_turn2_delta_bytes": noisy_turn2_delta_bytes,
                "noisy_error_signal_preserved": noisy_error_preserved,
            }
            print(
                f"  OK {hid}: startup={entry['startup_first_http_median_ms']}ms (p95/max={entry['startup_first_http_p95_ms']}ms), "
                f"rss={entry['peak_rss_mib']}MiB, turn1={entry['turn1_total_bytes']}B ({entry['static_prefix_est_tokens']} tok, {entry['tools_count']} tools), "
                f"invariant={entry['cross_cwd_prefix_invariant']}, noisy_t2_delta={entry['noisy_16kb_tool_turn2_delta_bytes']}B (err_ok={entry['noisy_error_signal_preserved']})",
                file=sys.stderr,
                flush=True,
            )
            results.append(entry)

    server.shutdown()
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": f"{platform.system()} {platform.machine()} ({platform.release()})",
        "runs_per_harness": runs,
        "warmups_per_harness": warmups,
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
