<p align="center">
  <img src="assets/logo.svg" width="76" height="76" alt="Carnot" />
</p>

<h1 align="center">Carnot</h1>

<p align="center">
  A small, fast Rust coding-agent harness built to be embedded.
</p>

<p align="center">
  <a href="#footprint">Footprint</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#interfaces--extensions">Interfaces</a> ·
  <a href="#providers">Providers</a> ·
  <a href="#platforms">Platforms</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="SECURITY.md">Security</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2024%20(MSRV%201.90)-e54d2e?style=flat-square&logo=rust&logoColor=white" alt="Rust 2024" />
  <img src="https://img.shields.io/badge/carnot%20crates-forbid(unsafe__code)-10b981?style=flat-square" alt="forbid unsafe_code" />
  <img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-6366f1?style=flat-square" alt="MIT / Apache-2.0" />
</p>

<br />

Carnot is a Rust coding-agent runtime and CLI designed to run inside IDEs (ACP), desktop or web apps (HTTP/SSE, Rust/Python/TS SDKs), CI jobs, and multi-agent fleets. It ships as a single 16.5 MB native binary with a 986-char static system prompt, prompt-cache invariance across directories, and fail-closed OS sandboxing.

> All Carnot crates set `unsafe_code = "forbid"`. Optional C bindings exist in `tree-sitter` syntax parsers (`carnot-tools` default feature) and the opt-in QuickJS `ext-codemode` extension (excluded from default `carnot-cli` builds).

## Footprint

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="assets/benchmarks-light.svg" />
  <img alt="Carnot cold start, peak RSS, installed size, and static prefix comparison" src="assets/benchmarks-light.svg" width="100%" />
</picture>

<details>
<summary>Exact numbers &amp; reproduction command (15 runs + 3 warmups, Darwin arm64, $0 local mock API)</summary>

| Harness | Startup (med) | Peak RSS | Installed Size | Static Prefix | Cross-CWD Invariant |
| :--- | ---: | ---: | ---: | ---: | :---: |
| **Carnot (core, 5 tools)** | **14.8 ms** | **12.5 MiB** | **16.5 MB** | **960 tok** | **Yes** |
| **Carnot (full, 13 tools)** | **23.3 ms** | **13.2 MiB** | **16.5 MB** | **2 627 tok** | **Yes** |
| Codex CLI | 78.1 ms | 132.4 MiB | 284.4 MB | 9 730 tok | No |
| Claude Code | 456.5 ms | 216.9 MiB | 347.0 MB | 19 210 tok | No |
| OpenCode | 652.4 ms | 737.2 MiB | 215.3 MB | 9 644 tok | No |

Reproduce locally:

```bash
cargo build --release --bin carnot
python3 benchmarks/compare_harnesses.py > /tmp/carnot_footprint.json
python3 benchmarks/render_footprint_svg.py /tmp/carnot_footprint.json assets
```

</details>

## Quick Start

### CLI & headless automation

```bash
cargo build --release --bin carnot

export ANTHROPIC_API_KEY=...
carnot -p "Fix the failing tests and summarize the root cause" --goal-cmd "cargo test --workspace"
git diff | carnot -p "Review this diff" --image ui.png

# Interactive TUI or resume latest session
carnot
carnot -c
```

### IDE backend (ACP) & local HTTP/SSE server

```bash
# Multi-session Agent Client Protocol over stdio
carnot --mode acp

# Authenticated localhost HTTP/1.1 + SSE server (powered by hyper)
CARNOT_HTTP_TOKEN=secret carnot serve --addr 127.0.0.1:7744
```

### Embed from Rust or Python

```rust
use carnot::CarnotBuilder;

let agent = CarnotBuilder::new("./my-repo")
    .provider_name("anthropic")
    .sandbox(true)
    .build()
    .await?;

let outcome = agent.prompt("Audit Cargo.toml for unused dependencies").await?;
```

```python
from carnot_sdk import CarnotClient

async with CarnotClient(cwd="./my-repo") as client:
    res = await client.prompt("Summarize recent git changes")
```

Run `carnot doctor` to inspect your environment and active sandbox backend, or `carnot audit-prefix` to verify prefix invariance.

## Core Design

- **Static prompt-cache prefix.** The 986-char system prompt and canonical tool schemas stay byte-identical across working directories, dates, and sessions (`carnot audit-prefix`). Dates are injected once in Turn 1 instead of the system prompt. Supports configurable cache TTLs (`short`, `hybrid`, `long`) and cache-warm compaction.
- **5 core tools.** `Read` (multimodal + optional tree-sitter outlines), `Edit` (atomic fuzzy & `ReadTracker`-anchored line-range patching with undo), `Write` (unread/stale guard + `.git` protection), `.gitignore`-aware `Grep`, and sandboxed `Bash` (PTY mode + BLAKE3 output spilling with secret redaction).
- **Fail-closed OS sandbox.** `Bash` runs under macOS Seatbelt or Linux `bwrap` (with Landlock ABI v1-v6 fallback inside containers). Outbound network is off by default, credentials are blocked, and `.git` stays read-only under Seatbelt and `bwrap`.
- **Opt-in workspace trust.** Workspace-local `.mcp.json`, hooks, and drop-in extensions never execute unless explicitly enabled (`--trust-workspace-mcp`, `--trust-workspace-ext`, `--trust-workspace-hooks`).

## Interfaces & Extensions

| Interface | Entry point | Notes |
| :--- | :--- | :--- |
| **Rust SDK** | `carnot::CarnotBuilder` | Direct embedding with custom `Arc<dyn Module>` and `Arc<dyn Tool>` |
| **ACP (IDE)** | `carnot --mode acp` | Multi-session multimodal Agent Client Protocol over stdio |
| **RPC & Python SDK** | `carnot --mode rpc` ([`sdk/python`](sdk/python/carnot_sdk.py)) | Bidirectional JSON-RPC 2.0 + zero-dep async Python client |
| **HTTP/SSE & TS SDK** | `carnot serve` ([`sdk/ts`](sdk/ts/src/index.ts)) | `hyper` HTTP/1.1 + SSE with Bearer auth, CSRF checks, and `Last-Event-ID` replay |
| **Terminal TUI / REPL** | `carnot` / `carnot --plain` | Ratatui TUI with mid-turn steer, `/context`, `/cache`, `/goal`, `/fleet`, `/fork`, `/undo` |

Built-in extensions (8 crates, all optional on the `carnot` library crate):
- Included in default `carnot-cli` builds: `web`, `lsp`, `subagent`, `memory`, `goal`, `hooks`, `claude-plugins`.
- Opt-in (`--features ext-codemode` or `--features all-ext`): `codemode` (QuickJS async sandbox via `rquickjs`).
- Drop-in file extensions: place executable scripts in `~/.carnot/extensions/` or pass `--ext ./my_ext.py` (`carnot-ext/1` JSON-RPC).

## Providers

Wire protocols are compiled Rust adapters; concrete providers live as data in [`crates/carnot-ai/catalog/providers.json`](crates/carnot-ai/catalog/providers.json) (overlayable with `~/.carnot/providers.json` or `--provider-config`):

- **Wire protocols:** `anthropic-messages`, `openai-responses` (stateless with encrypted reasoning), `openai-completions`, `google-generative`, `google-cloudcode`, `subprocess-jsonl`, and offline `faux`.
- **21 built-in presets (+ `faux`):** Anthropic, OpenAI (`openai`, `openai-responses`), Gemini, Cloud Code (`antigravity`), Grok / xAI, Meta Muse, Mistral, OpenCode Zen & Go, OpenRouter, DeepSeek, Kimi, Qwen, Groq, Fireworks, Together, Cerebras, Ollama, LM Studio, and vLLM.

User defaults live in `~/.carnot/config.toml` (override with `CARNOT_CONFIG`):

```toml
provider = "anthropic"
model = "claude-sonnet-5-5"
effort = "high"
cache_ttl = "hybrid" # short | hybrid | long
profile = "carnot"

[extensions]
web = true
lsp = true
subagent = true
memory = true
```

CLI helpers: `carnot providers`, `carnot models --all`, `carnot auth login|status|refresh|logout`, `carnot mcp list|inspect`, `carnot modules`, `carnot audit-prefix`, `carnot doctor`.

## Platforms

| Platform | CI Status | OS Sandbox (`Bash`) | Notes |
| :--- | :--- | :--- | :--- |
| **macOS** | Full test suite | Seatbelt (`sandbox-exec`) | Fail-closed, offline by default, `.git` read-only |
| **Linux** | Full test suite | `bwrap`, fallback to Landlock (ABI v1-v6) | See [`SECURITY.md`](SECURITY.md) for Landlock container caveats |
| **Windows** | Compile check only | None (`--no-sandbox` required) | In-process path confinement & `.git` write guard remain active |

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="assets/architecture-light.svg" />
  <img alt="Carnot 3-layer architecture" src="assets/architecture-light.svg" width="100%" />
</picture>

<details>
<summary>Workspace crates (13 core crates + 8 extensions)</summary>

| Crate | Role |
| :--- | :--- |
| `carnot-cli` | CLI binary, plain REPL, subcommands (`doctor`, `audit-prefix`, `models`, `providers`, `auth`, `mcp`, `modules`) |
| `carnot-tui` | Optional Ratatui TUI |
| `carnot` | Rust SDK facade (`CarnotBuilder`) |
| `carnot-protocol` | ACP stdio, JSON-RPC stdio, `hyper` HTTP/SSE server, interaction brokers |
| `carnot-agent` | Async turn loop, parallel tool scheduler, event log, compaction |
| `carnot-ext` | Core modules (`ExecMicroSkills`, `PrefixGuard`, `PolicyGuard`, `AskUser`) and drop-in host |
| `carnot-mcp` | MCP client (`2026-07-28`, `2025-11-25`, legacy SSE) with SSRF checks |
| `carnot-tools` | `Read`, `Edit`, `Write`, `Grep`, `Bash`, optional tree-sitter outlines, `ReadTracker` |
| `carnot-ai` | `ProviderRegistry`, wire adapters, `model_routes`, OAuth, retries |
| `carnot-session` | Branching JSONL v2 session store and prefix hashing |
| `carnot-sandbox` | macOS Seatbelt, Linux `bwrap`, and Linux Landlock fallback |
| `carnot-cas` | BLAKE3 spill store with secret redaction and output compressors |
| `carnot-core` | Leaf domain types (`Message`, `ToolEffects`, `Usage`, `SessionMetrics`, `CarnotConfig`) |
| `extensions/carnot-ext-*` | `web`, `lsp`, `subagent`, `memory`, `goal`, `hooks`, `claude-plugins`, `codemode` |

</details>

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`benchmarks/README.md`](benchmarks/README.md).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --lib
cargo test --workspace --all-features
python3 scripts/validate_catalog.py
```

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
