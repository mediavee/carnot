# Local Benchmarks & Evals

Scripts to measure Carnot locally without committing stale scorecards to git.

## 1. Harness Footprint ($0, ~30s)

Measures cold startup latency, peak RSS, installed footprint, static prefix size, cross-directory prefix invariance, and noisy-tool context discipline against locally installed CLIs (`codex`, `claude`, `opencode`, `pi`) using a local mock API:

```bash
cargo build --release --bin carnot
python3 benchmarks/compare_harnesses.py > /tmp/carnot_footprint.json
python3 benchmarks/render_footprint_svg.py /tmp/carnot_footprint.json assets
```

Quick zero-cost prefix check:

```bash
cargo run --release --bin carnot -- audit-prefix
```

## 2. Live Provider Smoke Suite (manual, cost-capped)

Tests real provider wire protocols (`basic_stream`, `core_tools`, `codemode`, `cache_hit`, `goal_loop`) with hard USD caps. Never run in CI.

```bash
python3 scripts/live_smoke.py --provider deepseek --dry-run
python3 scripts/live_smoke.py --provider deepseek --max-cost-usd 0.20 --max-total-cost-usd 0.75
python3 scripts/live_smoke.py --provider gemini --model gemini-2.5-flash --only codemode,goal_loop
```

## 3. Harbor Task Slice (optional A/B)

Run a small Terminal-Bench slice with per-task cost caps (`--ak max_cost_usd=0.50`). Optional extensions default to `false` in `harbor_carnot_agent.py`.

```bash
# Build static Linux binary (example on Apple Silicon for linux/aarch64 Docker):
cross build --release -p carnot-cli --target aarch64-unknown-linux-musl
mkdir -p /tmp/carnot-harbor-bin
cp target/aarch64-unknown-linux-musl/release/carnot /tmp/carnot-harbor-bin/carnot-linux-aarch64

# Run a 5-task baseline slice:
PYTHONPATH=$PWD/benchmarks \
  harbor run \
    -a harbor_carnot_agent:CarnotHarborAgent \
    -d cohere/terminal_bench_aai \
    -m deepseek/deepseek-chat \
    -l 5 -n 2 \
    --ak reasoning_effort=medium \
    --ak cache_ttl=hybrid \
    --ak max_cost_usd=0.50 \
    --jobs-dir /tmp/harbor-carnot-base

# Summarize runs:
python3 benchmarks/summarize_harbor_run.py --table /tmp/harbor-carnot-base
```

`CarnotHarborAgent` `--ak` options: `reasoning_effort`, `cache_ttl`, `profile`, `max_turns`, `max_cost_usd`, `codemode`, `subagent`, `web`, `lsp`, `memory`, `goal`, `goal_cmd`, `goal_max_iter`, `goal_model`.
