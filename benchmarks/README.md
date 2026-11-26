# Carnot Benchmarks

Carnot uses a **3-tier, cost-aware benchmarking workflow** so you can verify performance and regressions without burning money on full 89-task benchmark suites on every change.

## Tier 0 — Local Harness Overhead ($0.00, ~30s)

Deterministic local mock-API benchmark comparing Carnot against installed competitor CLIs (`pi`, `opencode`, `claude`, `codex`) on:
- Cold startup latency (`ms`) and peak RSS (`MB`)
- Turn-1 static prompt/tool payload size (`bytes` / estimated tokens)
- Cross-workspace static prefix invariance (`BLAKE3` match)
- 2-turn noisy-tool context discipline (CAS spillover + semantic shaping)

```bash
cargo build --release -p carnot-cli
python3 benchmarks/compare_harnesses.py
```

And instant zero-cost prefix invariance check:

```bash
cargo run --release -p carnot-cli -- audit-prefix
```

## Tier 1 — On-Demand Live Smoke Suite (~$0.05–$0.25, manual only)

Validates real provider wire protocols (`basic_stream`, `core_tools`, `codemode`, `cache_hit`, `goal_loop`) with hard per-case and total USD caps. **Never run automatically in CI or `cargo test`.**

```bash
# Preview exact commands ($0.00)
python3 scripts/live_smoke.py --provider deepseek --dry-run

# Run on one provider with strict caps
python3 scripts/live_smoke.py --provider deepseek --max-cost-usd 0.20 --max-total-cost-usd 0.75

# Run a single targeted check on a specific model
python3 scripts/live_smoke.py --provider gemini --model gemini-2.5-flash --only codemode,goal_loop

# Run basic + tools checks across all configured providers
python3 scripts/live_smoke.py --all-configured --only basic_stream,core_tools
```

## Tier 2 — Budget-Friendly Harbor Slice (~$1–$3 for 5–10 tasks)

Instead of launching an expensive full Terminal-Bench run, run a **5–10 task slice** with hard per-task cost caps (`--ak max_cost_usd=0.50`) and explicit A/B feature toggles.

### 1. Build static Linux musl binary once

```bash
# On Apple Silicon Docker containers (linux/aarch64):
cross build --release -p carnot-cli --target aarch64-unknown-linux-musl
mkdir -p /tmp/carnot-harbor-bin
cp target/aarch64-unknown-linux-musl/release/carnot /tmp/carnot-harbor-bin/carnot-linux-aarch64
```

### 2. Run a controlled A/B Harbor slice

All standalone batteries (`subagent`, `codemode`, `web`, `lsp`, `memory`) default to `false` in `harbor_carnot_agent.py` (passing `--no-*` explicitly) so A/B comparisons stay apples-to-apples.

```bash
# Baseline Carnot (5 tasks, max $0.50/task -> hard ceiling $2.50)
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

# Variant B: same 5 tasks with CodeMode enabled
PYTHONPATH=$PWD/benchmarks \
  harbor run \
    -a harbor_carnot_agent:CarnotHarborAgent \
    -d cohere/terminal_bench_aai \
    -m deepseek/deepseek-chat \
    -l 5 -n 2 \
    --ak reasoning_effort=medium \
    --ak cache_ttl=hybrid \
    --ak codemode=true \
    --ak max_cost_usd=0.50 \
    --jobs-dir /tmp/harbor-carnot-codemode
```

### 3. Summarize & compare results

```bash
# Markdown comparison table
python3 benchmarks/summarize_harbor_run.py --table \
  /tmp/harbor-carnot-base /tmp/harbor-carnot-codemode

# Full auditable JSON report
python3 benchmarks/summarize_harbor_run.py \
  /tmp/harbor-carnot-base /tmp/harbor-carnot-codemode > /tmp/compare.json
```

### Available `CarnotHarborAgent` `--ak` options

| Option | Default | Description |
| :--- | :--- | :--- |
| `reasoning_effort` | `medium` | `low \| medium \| high \| xhigh \| max` |
| `cache_ttl` | `hybrid` | `short \| hybrid \| long \| 5m \| 1h \| 24h` |
| `profile` | `carnot` | `minimal \| standard \| carnot` |
| `max_turns` | `500` | Maximum agent loop turns |
| `max_cost_usd` | `None` | Hard per-task USD circuit-breaker |
| `codemode` | `false` | Enable `QuickJS` `CodeMode` tool |
| `subagent` | `false` | Enable disposable-context `Subagent` tool |
| `web` / `lsp` / `memory` | `false` | Toggle optional standalone extensions |
| `goal` / `goal_cmd` | `None` | Enable `carnot-ext-goal` autonomous verification loop |
| `goal_max_iter` / `goal_model` | `None` | Goal loop iteration cap and independent verifier model |
