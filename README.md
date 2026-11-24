<p align="center">
  <img src="assets/logo.svg" width="76" height="76" alt="Carnot" />
</p>

<h1 align="center">Carnot</h1>

<p align="center">
  A fast, safe Rust coding-agent harness — maximum work, minimum wasted tokens.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#surfaces--extensions">Surfaces</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="SECURITY.md">Security</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2024-e54d2e?style=flat-square&logo=rust&logoColor=white" alt="Rust 2024" />
  <img src="https://img.shields.io/badge/unsafe-forbid-10b981?style=flat-square" alt="unsafe forbid" />
  <img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-6366f1?style=flat-square" alt="MIT / Apache-2.0" />
</p>

<br />

Carnot compiles to a single **13 MB** native binary (`#![forbid(unsafe_code)]`) with a **7.4 ms** cold start, **10.3 MiB** peak RAM, a **922-token** cross-directory invariant prompt prefix, and fail-closed OS sandboxing.

## Quick Start

```bash
# Build from source (all built-in extensions included)
cargo build --release --bin carnot --features all-ext

# Launch the interactive TUI (auto-detects your configured provider)
export ANTHROPIC_API_KEY=...
carnot

# Run a one-shot task
carnot -p "Fix the failing tests and summarize the root cause"

# Resume a session, or run as an IDE / Host / HTTP server
carnot --resume <session-id> -p "Continue"
carnot --mode acp
CARNOT_HTTP_TOKEN=secret carnot serve --addr 127.0.0.1:7744
```

> Standalone `carnot` enables `web`, `lsp`, `subagent`, and `memory` out of the box with zero static prefix-cache cost. Pass `--no-web` / `--no-lsp` / `--no-subagent` / `--no-memory` or `--profile minimal` to trim them, or run `carnot doctor` to inspect your setup.

## Highlights

- **Instant startup & tiny footprint** — `7.4 ms` median to first HTTP request and `10.3 MiB` peak RSS.
- **97.6% prompt cache hit rate** — byte-identical static prefix across workspaces (`carnot audit-prefix`), hybrid `1h`/`5m` cache breakpoints, and cache-warm semantic compaction.
- **5 surgical core tools** — `Read` (multimodal + tree-sitter outlines), `Edit` (atomic + fuzzy patch + CAS undo), `Write`, `.gitignore`-aware `Grep`, and `Bash` with BLAKE3 output shaping.
- **Fail-closed OS sandbox** — macOS Seatbelt and Linux `bwrap` with automatic Landlock (ABI v1–v6) fallback inside containers.
- **Zero-hardcode providers & full MCP** — Anthropic, OpenAI, Gemini, OpenRouter, DeepSeek, Ollama, and custom gateways driven purely by [`providers.json`](crates/carnot-ai/catalog/providers.json), plus MCP `2026-07-28`, `2025-11-25`, and legacy SSE.

## Benchmarks

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="assets/benchmarks-light.svg" />
  <img alt="Carnot cold-start latency and peak RSS memory benchmarks" src="assets/benchmarks-light.svg" width="100%" />
</picture>

<br />

| Benchmark | Result | Source |
| :--- | :--- | :--- |
| **Cold start / RAM / Binary** | `7.4 ms` · `10.3 MiB` RSS · `13.0 MB` · `922 tok` static prefix | [`harness_overhead.json`](benchmarks/harness_overhead.json) |
| **Terminal-Bench 4** (`Opus 5.5`, `max`) | `63.2%` (`24/38` valid tasks), `97.6%` cache hit, `100%` first-try `Edit`, `-18.5%` cost vs Claude Code | [`terminal_bench_4.json`](benchmarks/terminal_bench_4.json) |
| **Cache-Warm Compaction A/B** | `14/14` recall (0 amnesia), `-66.7%` post-compaction cost, `22.5×` context compression | [`compaction_ab.json`](benchmarks/compaction_ab.json) |

<details>
<summary>Full harness comparison &amp; compaction breakdown</summary>

#### Harness Overhead (15 runs + 3 warmups · Darwin arm64)

| Harness | Startup (med) | Peak RSS | Binary | Static Prefix | Cross-CWD Inv. |
| :--- | ---: | ---: | ---: | ---: | :---: |
| **Carnot (Rust)** | **7.4 ms** | **10.3 MiB** | **13.0 MB** | **922 tok** | **Yes** |
| Unharnessed (Rust) | 7.6 ms | 81.7 MiB | 26.8 MB | 1 035 tok | No |
| Codex CLI | 41.4 ms | 133.7 MiB | 284.4 MB | 9 730 tok | No |
| Claude Code | 249.9 ms | 216.0 MiB | 343.6 MB | 19 210 tok | No |
| Pi (Node) | 462.5 ms | 159.4 MiB | 238.1 MB | 1 399 tok | No |
| OpenCode (Bun) | 545.9 ms | 528.7 MiB | 215.3 MB | 9 644 tok | No |

#### Two-Tier Cache-Warm Compaction A/B (`claude-sonnet-5-5`, 4-phase resumed session)

| Metric | No Compaction | Carnot Compaction | Delta |
| :--- | ---: | ---: | ---: |
| Verification score | `14 / 14` | **`14 / 14`** | `0 amnesia` |
| Total net cost | `$0.7475` | **`$0.5593`** | **`-25.2%`** |
| Post-compaction cost (Phases 2–4) | `$0.2196` | **`$0.0732`** | **`-66.7%`** |
| Context size | `74,112` peak | **`68,775 → 3,059`** | **`-95.6%`** |

</details>

## Surfaces & Extensions

| Surface | How to use | Notes |
| :--- | :--- | :--- |
| **Terminal TUI / REPL** | `carnot` (or `carnot --plain`) | Streaming Markdown, compact diffs, `^P` model picker, interactive approvals & `AskUser` |
| **IDE (ACP stdio)** | `carnot --mode acp` | Multi-session multimodal Agent Client Protocol with live model/effort switching |
| **Host RPC** | `carnot --mode rpc --host-extension` | Bidirectional JSON-RPC for embedding hosts (dynamic tools, gates, hooks, context breakdown) |
| **HTTP / SSE API** | `carnot serve --addr 127.0.0.1:7744` | Bearer-authenticated REST + Server-Sent Events |
| **Rust SDK** | `carnot::CarnotBuilder` | Embed directly in Rust apps with custom `Arc<dyn Module>` and `Arc<dyn Tool>` |

**Extensions** come in two zero-prefix-bloat tiers:

- **Built-in Rust extensions** (`--features all-ext`): `web` (native + local search/fetch), `lsp` (warm diagnostics + `LspQuery`), `subagent` (isolated worktree delegation), `memory` (BM25 cross-session memory), `hooks` (Claude Code compatible), and `claude-plugins`.
- **Drop-in file extensions**: place any executable script or binary in `~/.carnot/extensions/` or pass `--ext ./my_ext.py` (speaks `carnot-ext/1` JSON-RPC; Python helper in [`sdk/python/carnot_sdk.py`](sdk/python/carnot_sdk.py)).

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="assets/architecture-light.svg" />
  <img alt="Carnot 3-layer architecture" src="assets/architecture-light.svg" width="100%" />
</picture>

<details>
<summary>Workspace crate map (12 core crates + 6 extensions)</summary>

| Crate | Role |
| :--- | :--- |
| `carnot-cli` | CLI, Ratatui TUI, plain REPL, `doctor`, `audit-prefix`, `models`, `providers` |
| `carnot` | Ergonomic Rust SDK facade (`CarnotBuilder`) |
| `carnot-protocol` | ACP stdio, Host RPC stdio, HTTP/SSE server, interaction brokers |
| `carnot-agent` | Async turn loop, effect-parallel scheduler, lossless `EventLog`, compaction |
| `carnot-ext` | Core modules (`ExecMicroSkills`, `PrefixGuard`, `PolicyGuard`, `AskUser`) & drop-in host |
| `carnot-mcp` | MCP `2026-07-28` / `2025-11-25` / legacy SSE client with lazy schema caching |
| `carnot-tools` | `Read`, `Edit`, `Write`, `Grep`, `Bash`, optional tree-sitter outlines & `ReadTracker` |
| `carnot-ai` | Data-driven `ProviderRegistry`, wire adapters, OAuth, retry & `webpki` TLS |
| `carnot-session` | Crash-safe branching JSONL v2 journal & canonical prefix hashing |
| `carnot-sandbox` | Fail-closed macOS Seatbelt, Linux `bwrap`, and safe Linux Landlock fallback |
| `carnot-cas` | Zero-I/O BLAKE3 content store with pre-spill secret redaction & semantic compressors |
| `carnot-core` | Leaf types (`Message`, `ToolEffects`, `Usage`, `ContextBreakdown`, `CarnotConfig`) |

</details>

## Security

See [`SECURITY.md`](SECURITY.md) for the full threat model: fail-closed OS sandboxing, filesystem confinement, pre-spill secret redaction, SSRF protection, and workspace trust boundaries.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
