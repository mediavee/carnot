#!/usr/bin/env python3
"""
Carnot On-Demand Live Smoke Suite (opt-in only — NEVER run automatically in CI or `cargo test`).

Runs 5 small, cost-capped, real end-to-end provider checks against a built `carnot` binary in an
isolated temporary git workspace:

  1. basic_stream : Streaming text, usage telemetry, cost accounting, prefix_audit stable=true.
  2. core_tools   : Real Read + line-range Edit + sandboxed Bash verification on a tiny Python bug.
  3. codemode     : Real CodeMode JS script batching `Promise.all([tools.Read(...), ...])`.
  4. cache_hit    : 2-turn resumed session verifying `cache_read_tokens > 0` on Turn 2.
  5. goal_loop    : `--goal-cmd` deterministic verification loop driving repair until exit 0.

Examples:
  # Preview commands without spending $0.00
  python3 scripts/live_smoke.py --provider anthropic --dry-run

  # Smoke test one provider with strict $0.25/case and $1.00 total cap
  python3 scripts/live_smoke.py --provider deepseek --max-cost-usd 0.25 --max-total-cost-usd 1.00

  # Run only specific cases on a specific model
  python3 scripts/live_smoke.py --provider gemini --model gemini-2.5-flash --only basic_stream,goal_loop

  # Smoke test every currently configured provider discovered via `carnot auth status`
  python3 scripts/live_smoke.py --all-configured --only basic_stream,core_tools
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALL_CASES = ("basic_stream", "core_tools", "codemode", "cache_hit", "goal_loop")

# Providers whose local/fake backends should be skipped during `--all-configured` unless named explicitly.
SKIP_IN_ALL_CONFIGURED = {"faux", "ollama"}


@dataclass
class CaseResult:
    provider: str
    model: str | None
    case: str
    status: str  # PASS | FAIL | WARN | SKIP | DRY_RUN
    duration_s: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    tools_called: list[str] = field(default_factory=list)
    detail: str = ""


def find_carnot_binary(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"carnot binary not found at: {p}")
        return p

    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (
        repo_root / "target" / "release" / "carnot",
        repo_root / "target" / "debug" / "carnot",
    ):
        if candidate.is_file():
            return candidate

    which = shutil.which("carnot")
    if which:
        return Path(which)

    raise SystemExit(
        "No `carnot` binary found. Build one first with `cargo build --release -p carnot-cli` "
        "(or `cargo build -p carnot-cli`), or pass `--binary /path/to/carnot`."
    )


def discover_configured_providers(binary: Path) -> list[str]:
    proc = subprocess.run(
        [str(binary), "auth", "status"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"`carnot auth status` failed ({proc.returncode}): {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    out: list[str] = []
    for row in data.get("providers", []):
        pid = row.get("provider")
        api = row.get("api", "")
        if row.get("configured") and pid and pid not in SKIP_IN_ALL_CONFIGURED and api != "faux":
            out.append(str(pid))
    return out


def init_temp_git_workspace() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="carnot-live-smoke-"))
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
    subprocess.run(["git", "config", "user.email", "smoke@carnot.local"], cwd=tmp, check=True)
    subprocess.run(["git", "config", "user.name", "Carnot Smoke"], cwd=tmp, check=True)
    return tmp


def parse_jsonl_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "event" in obj:
            events.append(obj)
    return events


def extract_telemetry(events: list[dict[str, Any]]) -> dict[str, Any]:
    tools_called: list[str] = []
    errors: list[str] = []
    final_text = ""
    cost_usd = 0.0
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    prefix_stable = True
    cache_breaks = 0
    resolved_model: str | None = None
    session_id: str | None = None

    for ev in events:
        kind = ev.get("event")
        if kind == "session_started":
            resolved_model = ev.get("model") or resolved_model
            session_id = ev.get("session_id") or session_id
        elif kind == "prefix_audit":
            if ev.get("stable") is False:
                prefix_stable = False
        elif kind == "tool_start":
            name = ev.get("tool") or ev.get("name")
            if name:
                tools_called.append(str(name))
        elif kind == "error":
            errors.append(str(ev.get("message") or ev))
        elif kind == "run_settled":
            final_text = str(ev.get("final_text") or "")
            metrics = ev.get("metrics") or {}
            usage = metrics.get("total_usage") or {}
            cost_usd = float(usage.get("cost_usd") or 0.0)
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            cache_read_tokens = int(usage.get("cache_read_tokens") or 0)
            cache_breaks = int(metrics.get("cache_breaks") or 0)

    return {
        "tools_called": tools_called,
        "errors": errors,
        "final_text": final_text,
        "cost_usd": cost_usd,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "prefix_stable": prefix_stable,
        "cache_breaks": cache_breaks,
        "resolved_model": resolved_model,
        "session_id": session_id,
    }


def build_base_cmd(
    binary: Path,
    provider: str,
    model: str | None,
    max_cost_usd: float,
    no_sandbox: bool,
    extra_flags: list[str],
    prompt: str,
) -> list[str]:
    cmd = [
        str(binary),
        "--provider",
        provider,
        "--mode",
        "json",
        "--no-mcp",
        "--no-web",
        "--no-lsp",
        "--no-memory",
        "--no-subagent",
        "--no-context-files",
        "--no-skills",
        "--max-turns",
        "12",
        "--max-cost-usd",
        f"{max_cost_usd:.4f}",
    ]
    if model:
        cmd.extend(["--model", model])
    if no_sandbox:
        cmd.append("--no-sandbox")
    cmd.extend(extra_flags)
    cmd.extend(["-p", prompt])
    return cmd


def invoke_carnot(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, list[dict[str, Any]], str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    events = parse_jsonl_events(proc.stdout)
    return proc.returncode, events, proc.stderr.strip()


def run_case(
    binary: Path,
    provider: str,
    model: str | None,
    case: str,
    max_cost_usd: float,
    timeout_s: int,
    no_sandbox: bool,
    dry_run: bool,
) -> CaseResult:
    ws = init_temp_git_workspace()
    t0 = time.monotonic()
    try:
        if case == "basic_stream":
            prompt = "Reply with exactly the single token: CARNOT_SMOKE_OK"
            cmd = build_base_cmd(
                binary, provider, model, max_cost_usd, no_sandbox,
                ["--no-session", "--no-codemode"], prompt,
            )
            if dry_run:
                return CaseResult(provider, model, case, "DRY_RUN", detail=" ".join(cmd))
            rc, events, stderr = invoke_carnot(cmd, ws, timeout_s)
            tel = extract_telemetry(events)
            ok = (
                rc == 0
                and not tel["errors"]
                and tel["prefix_stable"]
                and tel["cache_breaks"] == 0
                and "CARNOT_SMOKE_OK" in tel["final_text"]
                and tel["output_tokens"] > 0
            )
            detail = (
                "ok"
                if ok
                else f"rc={rc} text={tel['final_text'][:120]!r} errors={tel['errors']} stderr={stderr[:160]!r}"
            )
            return CaseResult(
                provider=provider,
                model=tel["resolved_model"] or model,
                case=case,
                status="PASS" if ok else "FAIL",
                duration_s=time.monotonic() - t0,
                cost_usd=tel["cost_usd"],
                input_tokens=tel["input_tokens"],
                output_tokens=tel["output_tokens"],
                cache_read_tokens=tel["cache_read_tokens"],
                tools_called=tel["tools_called"],
                detail=detail,
            )

        if case == "core_tools":
            (ws / "math_bug.py").write_text(
                "def multiply(a: int, b: int) -> int:\n"
                "    # BUG: adds instead of multiplies\n"
                "    return a + b\n",
                encoding="utf-8",
            )
            (ws / "check.py").write_text(
                "from math_bug import multiply\n"
                "assert multiply(6, 7) == 42, f'got {multiply(6, 7)}'\n"
                "print('CHECK_OK')\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "."], cwd=ws, check=True)
            subprocess.run(["git", "commit", "-qm", "seed"], cwd=ws, check=True)
            prompt = (
                "Read `math_bug.py`, fix `multiply` using Edit so it multiplies `a * b`, "
                "then run `python3 check.py` with Bash and report the result."
            )
            cmd = build_base_cmd(
                binary, provider, model, max_cost_usd, no_sandbox,
                ["--no-session", "--no-codemode"], prompt,
            )
            if dry_run:
                return CaseResult(provider, model, case, "DRY_RUN", detail=" ".join(cmd))
            rc, events, stderr = invoke_carnot(cmd, ws, timeout_s)
            tel = extract_telemetry(events)
            verify = subprocess.run(
                [sys.executable, "check.py"], cwd=ws, capture_output=True, text=True, check=False
            )
            used_edit_or_write = any(t in ("Edit", "Write") for t in tel["tools_called"])
            ok = rc == 0 and verify.returncode == 0 and "CHECK_OK" in verify.stdout and used_edit_or_write
            detail = (
                f"tools={tel['tools_called']}"
                if ok
                else f"rc={rc} verify_rc={verify.returncode} tools={tel['tools_called']} stderr={stderr[:160]!r}"
            )
            return CaseResult(
                provider=provider,
                model=tel["resolved_model"] or model,
                case=case,
                status="PASS" if ok else "FAIL",
                duration_s=time.monotonic() - t0,
                cost_usd=tel["cost_usd"],
                input_tokens=tel["input_tokens"],
                output_tokens=tel["output_tokens"],
                cache_read_tokens=tel["cache_read_tokens"],
                tools_called=tel["tools_called"],
                detail=detail,
            )

        if case == "codemode":
            (ws / "n1.txt").write_text("11\n", encoding="utf-8")
            (ws / "n2.txt").write_text("17\n", encoding="utf-8")
            (ws / "n3.txt").write_text("14\n", encoding="utf-8")
            prompt = (
                "Use the `CodeMode` tool (not individual Read calls) to read `n1.txt`, `n2.txt`, "
                "and `n3.txt` in parallel with `Promise.all`, sum the 3 integers in JS, and reply "
                "with `SUM=<total>`."
            )
            cmd = build_base_cmd(
                binary, provider, model, max_cost_usd, no_sandbox,
                ["--no-session", "--codemode"], prompt,
            )
            if dry_run:
                return CaseResult(provider, model, case, "DRY_RUN", detail=" ".join(cmd))
            rc, events, stderr = invoke_carnot(cmd, ws, timeout_s)
            tel = extract_telemetry(events)
            used_cm = "CodeMode" in tel["tools_called"]
            has_answer = "42" in tel["final_text"]
            ok = rc == 0 and used_cm and has_answer
            detail = (
                f"tools={tel['tools_called']}"
                if ok
                else f"rc={rc} used_codemode={used_cm} text={tel['final_text'][:120]!r} stderr={stderr[:160]!r}"
            )
            return CaseResult(
                provider=provider,
                model=tel["resolved_model"] or model,
                case=case,
                status="PASS" if ok else "FAIL",
                duration_s=time.monotonic() - t0,
                cost_usd=tel["cost_usd"],
                input_tokens=tel["input_tokens"],
                output_tokens=tel["output_tokens"],
                cache_read_tokens=tel["cache_read_tokens"],
                tools_called=tel["tools_called"],
                detail=detail,
            )

        if case == "cache_hit":
            # Seed a ~1,600-token deterministic context note in turn 1 so providers with a
            # 1,024-token prompt-cache threshold (OpenAI / Anthropic / Gemini / DeepSeek) warm up.
            session_dir = ws / ".smoke-sessions"
            session_dir.mkdir(exist_ok=True)
            filler = "\n".join(
                f"Spec rule {i:03d}: invariant token ALPHA_{i:03d} must remain unchanged across edits."
                for i in range(1, 95)
            )
            p1 = f"Acknowledge this specification with exactly `ACK_1` (do not call tools):\n\n{filler}"
            p2 = "Now reply with exactly `ACK_2` and quote the invariant token from Spec rule 042."
            cmd1 = build_base_cmd(
                binary, provider, model, max_cost_usd / 2.0, no_sandbox,
                ["--session-dir", str(session_dir), "--no-codemode"], p1,
            )
            if dry_run:
                return CaseResult(provider, model, case, "DRY_RUN", detail=f"turn1: {' '.join(cmd1[:18])} ... -> --resume")
            rc1, ev1, err1 = invoke_carnot(cmd1, ws, timeout_s)
            tel1 = extract_telemetry(ev1)
            session_files = sorted(session_dir.glob("*.jsonl"))
            if rc1 != 0 or not session_files:
                return CaseResult(
                    provider=provider,
                    model=tel1["resolved_model"] or model,
                    case=case,
                    status="FAIL",
                    duration_s=time.monotonic() - t0,
                    cost_usd=tel1["cost_usd"],
                    detail=f"turn1 failed rc={rc1} sessions={len(session_files)} stderr={err1[:160]!r}",
                )
            cmd2 = build_base_cmd(
                binary, provider, model, max_cost_usd / 2.0, no_sandbox,
                ["--session-dir", str(session_dir), "--resume", str(session_files[-1]), "--no-codemode"],
                p2,
            )
            rc2, ev2, err2 = invoke_carnot(cmd2, ws, timeout_s)
            tel2 = extract_telemetry(ev2)
            total_cost = tel1["cost_usd"] + tel2["cost_usd"]
            answered = rc2 == 0 and "ALPHA_042" in tel2["final_text"]
            cached = tel2["cache_read_tokens"] > 0
            if answered and cached and tel2["cache_breaks"] == 0:
                status = "PASS"
                detail = f"turn2 cache_read_tokens={tel2['cache_read_tokens']}"
            elif answered and not cached:
                # Some models have 2048-4096 token cache thresholds or omit cache telemetry.
                status = "WARN"
                detail = "resumed & recalled ALPHA_042, but provider reported cache_read_tokens=0"
            else:
                status = "FAIL"
                detail = f"rc2={rc2} text={tel2['final_text'][:120]!r} stderr={err2[:160]!r}"
            return CaseResult(
                provider=provider,
                model=tel2["resolved_model"] or tel1["resolved_model"] or model,
                case=case,
                status=status,
                duration_s=time.monotonic() - t0,
                cost_usd=total_cost,
                input_tokens=tel1["input_tokens"] + tel2["input_tokens"],
                output_tokens=tel1["output_tokens"] + tel2["output_tokens"],
                cache_read_tokens=tel2["cache_read_tokens"],
                tools_called=tel1["tools_called"] + tel2["tools_called"],
                detail=detail,
            )

        if case == "goal_loop":
            (ws / "calc.py").write_text(
                "def fib(n: int) -> int:\n"
                "    return 0  # TODO\n",
                encoding="utf-8",
            )
            (ws / "test_calc.py").write_text(
                "from calc import fib\n"
                "assert [fib(i) for i in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]\n"
                "print('FIB_OK')\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "."], cwd=ws, check=True)
            subprocess.run(["git", "commit", "-qm", "seed fib"], cwd=ws, check=True)
            prompt = "Implement `fib(n)` in `calc.py` so `test_calc.py` passes."
            py_cmd = "python3" if shutil.which("python3") else sys.executable
            cmd = build_base_cmd(
                binary, provider, model, max_cost_usd, no_sandbox,
                [
                    "--no-session",
                    "--no-codemode",
                    "--goal-cmd",
                    f"{py_cmd} test_calc.py",
                    "--goal-max-iter",
                    "4",
                ],
                prompt,
            )
            if dry_run:
                return CaseResult(provider, model, case, "DRY_RUN", detail=" ".join(cmd))
            rc, events, stderr = invoke_carnot(cmd, ws, timeout_s)
            tel = extract_telemetry(events)
            verify = subprocess.run(
                [sys.executable, "test_calc.py"], cwd=ws, capture_output=True, text=True, check=False
            )
            ok = rc == 0 and verify.returncode == 0 and "FIB_OK" in verify.stdout
            detail = (
                f"verified FIB_OK (tools={len(tel['tools_called'])})"
                if ok
                else f"rc={rc} verify_rc={verify.returncode} stderr={stderr[:160]!r}"
            )
            return CaseResult(
                provider=provider,
                model=tel["resolved_model"] or model,
                case=case,
                status="PASS" if ok else "FAIL",
                duration_s=time.monotonic() - t0,
                cost_usd=tel["cost_usd"],
                input_tokens=tel["input_tokens"],
                output_tokens=tel["output_tokens"],
                cache_read_tokens=tel["cache_read_tokens"],
                tools_called=tel["tools_called"],
                detail=detail,
            )

        raise ValueError(f"unknown smoke case: {case}")
    except subprocess.TimeoutExpired:
        return CaseResult(
            provider=provider,
            model=model,
            case=case,
            status="FAIL",
            duration_s=time.monotonic() - t0,
            detail=f"timed out after {timeout_s}s",
        )
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def print_summary(results: list[CaseResult]) -> None:
    print()
    header = f"{'PROVIDER':<12} {'MODEL':<26} {'CASE':<13} {'STATUS':<8} {'TIME':>6} {'COST':>8} {'CACHE_RD':>9}  DETAIL"
    print(header)
    print("-" * len(header))
    total_cost = 0.0
    for r in results:
        total_cost += r.cost_usd
        m = (r.model or "-")[:26]
        print(
            f"{r.provider:<12} {m:<26} {r.case:<13} {r.status:<8} "
            f"{r.duration_s:>5.1f}s ${r.cost_usd:>7.4f} {r.cache_read_tokens:>9d}  {r.detail}"
        )
    print("-" * len(header))
    passed = sum(1 for r in results if r.status == "PASS")
    warned = sum(1 for r in results if r.status == "WARN")
    failed = sum(1 for r in results if r.status == "FAIL")
    print(f"Total: {len(results)} cases | PASS={passed} WARN={warned} FAIL={failed} | Spend=${total_cost:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="On-demand live smoke suite for Carnot (never run automatically in CI)."
    )
    parser.add_argument("--provider", help="Single provider id to test (e.g. anthropic, openai, gemini, deepseek)")
    parser.add_argument(
        "--all-configured",
        action="store_true",
        help="Auto-discover all configured providers via `carnot auth status`",
    )
    parser.add_argument("--model", help="Optional model override for the chosen provider")
    parser.add_argument("--binary", help="Path to `carnot` binary (default: auto-detect target/release or target/debug)")
    parser.add_argument(
        "--only",
        help=f"Comma-separated subset of cases to run (choices: {','.join(ALL_CASES)})",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.25,
        help="Hard per-case USD cap passed to `carnot --max-cost-usd` (default: 0.25)",
    )
    parser.add_argument(
        "--max-total-cost-usd",
        type=float,
        default=1.50,
        help="Stop remaining cases once cumulative suite spend reaches this USD cap (default: 1.50)",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Per-case timeout in seconds (default: 120)")
    parser.add_argument("--no-sandbox", action="store_true", help="Pass `--no-sandbox` (useful in restricted containers)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without calling any provider ($0.00)")
    parser.add_argument("--json-out", help="Optional path to write full JSON results")
    args = parser.parse_args()

    if not args.provider and not args.all_configured:
        parser.error("specify `--provider <id>` or `--all-configured` (use `--dry-run` to preview without spending)")

    binary = find_carnot_binary(args.binary)
    providers = [args.provider] if args.provider else discover_configured_providers(binary)
    if not providers:
        raise SystemExit("No configured live providers found via `carnot auth status`.")

    if args.only:
        cases = [c.strip() for c in args.only.split(",") if c.strip()]
        invalid = [c for c in cases if c not in ALL_CASES]
        if invalid:
            raise SystemExit(f"Unknown case(s): {invalid}. Valid: {', '.join(ALL_CASES)}")
    else:
        cases = list(ALL_CASES)

    results: list[CaseResult] = []
    spent_total = 0.0

    for prov in providers:
        for case in cases:
            if not args.dry_run and spent_total >= args.max_total_cost_usd:
                results.append(
                    CaseResult(
                        provider=prov,
                        model=args.model,
                        case=case,
                        status="SKIP",
                        detail=f"cumulative budget cap reached (${spent_total:.4f} >= ${args.max_total_cost_usd:.2f})",
                    )
                )
                continue
            res = run_case(
                binary=binary,
                provider=prov,
                model=args.model if len(providers) == 1 else None,
                case=case,
                max_cost_usd=args.max_cost_usd,
                timeout_s=args.timeout,
                no_sandbox=args.no_sandbox,
                dry_run=args.dry_run,
            )
            spent_total += res.cost_usd
            results.append(res)

    print_summary(results)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "binary": str(binary),
                    "total_cost_usd": round(spent_total, 6),
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
