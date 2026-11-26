#!/usr/bin/env python3
"""
Summarizes one or more Harbor job directories (Carnot vs Claude Code / competitors)
into a single auditable JSON report with per-task pass/fail, reward, cost, tokens,
cache hit rate, turns, rewinds, and wall time.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_iso_sec(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return round((t1 - t0).total_seconds(), 2)
    except Exception:
        return None


def summarize_job(job_dir: Path) -> dict[str, Any]:
    # Harbor writes <job_dir>/<timestamp>/result.json or <job_dir>/result.json
    candidates = sorted(job_dir.glob("*/result.json"))
    if (job_dir / "result.json").is_file():
        run_root = job_dir
    elif candidates:
        run_root = candidates[-1].parent
    else:
        raise FileNotFoundError(f"No Harbor result.json found under {job_dir}")

    trials: list[dict[str, Any]] = []
    for trial_res in sorted(run_root.glob("*/result.json")):
        if trial_res == run_root / "result.json":
            continue
        data = json.loads(trial_res.read_text(encoding="utf-8"))
        task_name = data.get("task_name") or trial_res.parent.name.split("__")[0]
        agent_info = data.get("agent_info") or {}
        agent_res = data.get("agent_result") or {}
        ver_res = data.get("verifier_result") or {}
        rewards = ver_res.get("rewards") or {}
        reward = float(rewards.get("reward", 0.0) or 0.0) if isinstance(rewards, dict) else 0.0
        ae = data.get("agent_execution") or {}
        aset = data.get("agent_setup") or {}
        agent_setup_s = parse_iso_sec(aset.get("started_at"), aset.get("finished_at"))
        agent_wall_s = parse_iso_sec(ae.get("started_at"), ae.get("finished_at"))
        total_wall_s = parse_iso_sec(data.get("started_at"), data.get("finished_at"))
        exc = data.get("exception_info")

        in_t = int(agent_res.get("n_input_tokens") or 0)
        cache_t = int(agent_res.get("n_cache_tokens") or 0)
        out_t = int(agent_res.get("n_output_tokens") or 0)
        cost_usd = agent_res.get("cost_usd")
        meta = dict(agent_res.get("metadata") or {})

        # Fallback: parse agent/carnot.jsonl directly if Harbor had not synced logs before populate_context_post_run
        jsonl_path = trial_res.parent / "agent" / "carnot.jsonl"
        if in_t == 0 and out_t == 0 and jsonl_path.is_file():
            u_in = u_cr = u_cw5 = u_cw1 = u_out = 0
            u_cost = 0.0
            u_turns = 0
            u_rewinds = 0
            u_prefix_hits = 0
            u_prefix_misses = 0
            u_sem_saved = 0
            for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "usage_recorded":
                    u_turns += 1
                    u = ev.get("usage") or {}
                    u_in += int(u.get("input_tokens") or 0)
                    u_cr += int(u.get("cache_read_tokens") or 0)
                    u_cw5 += int(u.get("cache_write_5m_tokens") or 0)
                    u_cw1 += int(u.get("cache_write_1h_tokens") or 0)
                    u_out += int(u.get("output_tokens") or 0)
                    u_cost += float(u.get("cost_usd") or 0.0)
                elif ev.get("event") in ("compaction_applied", "rewind_applied"):
                    u_rewinds += 1
                elif ev.get("event") == "run_settled" and isinstance(ev.get("metrics"), dict):
                    m = ev["metrics"]
                    tu = m.get("total_usage") or {}
                    u_in = int(tu.get("input_tokens", u_in) or 0)
                    u_cr = int(tu.get("cache_read_tokens", u_cr) or 0)
                    u_cw5 = int(tu.get("cache_write_5m_tokens", u_cw5) or 0)
                    u_cw1 = int(tu.get("cache_write_1h_tokens", u_cw1) or 0)
                    u_out = int(tu.get("output_tokens", u_out) or 0)
                    u_cost = float(tu.get("cost_usd", u_cost) or 0.0)
                    u_turns = int(m.get("turns", u_turns) or u_turns)
                    u_rewinds = int(
                        m.get("compactions_applied", m.get("rewinds_applied", u_rewinds)) or u_rewinds
                    )
                    u_prefix_hits = int(m.get("prefix_cache_hits") or 0)
                    u_prefix_misses = int(m.get("prefix_cache_misses") or 0)
                    u_sem_saved = int(m.get("semantic_bytes_saved") or 0)
            tot = u_in + u_cr + u_cw5 + u_cw1
            if tot > 0 or u_out > 0:
                in_t = tot
                cache_t = u_cr
                out_t = u_out
                cost_usd = round(u_cost, 6)
                meta.setdefault("turns", u_turns)
                meta.setdefault("compactions_applied", u_rewinds)
                meta.setdefault("prefix_cache_hits", u_prefix_hits)
                meta.setdefault("prefix_cache_misses", u_prefix_misses)
                meta.setdefault("semantic_bytes_saved", u_sem_saved)

        trials.append(
            {
                "task_name": task_name,
                "agent": agent_info.get("name"),
                "model": (agent_info.get("model_info") or {}).get("name"),
                "passed": reward >= 1.0,
                "reward": reward,
                "agent_setup_s": agent_setup_s,
                "agent_wall_s": agent_wall_s,
                "total_wall_s": total_wall_s,
                "n_input_tokens": in_t,
                "n_cache_tokens": cache_t,
                "n_output_tokens": out_t,
                "cache_hit_rate": round(cache_t / in_t, 4) if in_t > 0 else 0.0,
                "cost_usd": round(float(cost_usd), 6) if cost_usd is not None else None,
                "turns": meta.get("turns"),
                "compactions_applied": meta.get("compactions_applied", meta.get("rewinds_applied")),
                "codemode_calls": meta.get("codemode_calls"),
                "exception": (exc or {}).get("exception_type") if isinstance(exc, dict) else None,
            }
        )

    passed_n = sum(1 for t in trials if t["passed"])
    tot_n = len(trials)
    tot_in = sum(t["n_input_tokens"] for t in trials)
    tot_cache = sum(t["n_cache_tokens"] for t in trials)
    tot_out = sum(t["n_output_tokens"] for t in trials)
    costs = [t["cost_usd"] for t in trials if t["cost_usd"] is not None]
    walls = [t["agent_wall_s"] for t in trials if t["agent_wall_s"] is not None]

    return {
        "job_dir": str(run_root),
        "agent": trials[0]["agent"] if trials else None,
        "model": trials[0]["model"] if trials else None,
        "passed": passed_n,
        "total_tasks": tot_n,
        "pass_rate_pct": round(passed_n / tot_n * 100.0, 1) if tot_n else 0.0,
        "total_cost_usd": round(sum(costs), 4) if costs else None,
        "mean_cost_usd": round(sum(costs) / len(costs), 4) if costs else None,
        "total_agent_wall_s": round(sum(walls), 1) if walls else None,
        "mean_agent_wall_s": round(sum(walls) / len(walls), 1) if walls else None,
        "total_input_tokens": tot_in,
        "total_cache_tokens": tot_cache,
        "aggregate_cache_hit_pct": round(tot_cache / tot_in * 100.0, 1) if tot_in > 0 else 0.0,
        "total_output_tokens": tot_out,
        "trials": trials,
    }


def print_markdown_table(report: list[dict[str, Any]]) -> None:
    print("| Agent | Model | Pass | Total Cost | Mean Cost | Cache Hit | Mean Wall |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    for r in report:
        cost = f"${r['total_cost_usd']:.4f}" if r["total_cost_usd"] is not None else "-"
        mcost = f"${r['mean_cost_usd']:.4f}" if r["mean_cost_usd"] is not None else "-"
        wall = f"{r['mean_agent_wall_s']:.1f}s" if r["mean_agent_wall_s"] is not None else "-"
        print(
            f"| {r['agent'] or '-'} | {r['model'] or '-'} | "
            f"{r['passed']}/{r['total_tasks']} ({r['pass_rate_pct']}%) | "
            f"{cost} | {mcost} | {r['aggregate_cache_hit_pct']}% | {wall} |"
        )


def main():
    args = sys.argv[1:]
    table_mode = False
    if "--table" in args:
        table_mode = True
        args = [a for a in args if a != "--table"]
    if not args:
        print("usage: summarize_harbor_run.py [--table] <job_dir> [<job_dir2> ...]", file=sys.stderr)
        sys.exit(2)
    report = [summarize_job(Path(p)) for p in args]
    if table_mode:
        print_markdown_table(report)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
