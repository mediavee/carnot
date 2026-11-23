<p align="center">
  <img src="assets/logo.svg" width="88" height="88" alt="Carnot logo" />
</p>

<h1 align="center">Carnot</h1>

<p align="center">
  <strong>Thermodynamic Coding Agent Micro-Harness in Pure Safe Rust.</strong><br/>
  7.4 ms cold start · 10.3 MiB RSS · 97.0% prompt cache hit · 100% first-try edits · <code>#![forbid(unsafe_code)]</code>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-2024%20edition-orange?style=flat-square&logo=rust" alt="Rust 2024" />
  <img src="https://img.shields.io/badge/unsafe-forbid-10b981?style=flat-square" alt="unsafe forbid" />
  <img src="https://img.shields.io/badge/MCP-2026--07--28%20%7C%202025--11--25%20%7C%20SSE-0ea5e9?style=flat-square" alt="MCP" />
  <img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-6366f1?style=flat-square" alt="License" />
</p>

---

**Carnot** is a high-efficiency coding-agent runtime designed around one principle: **every wasted token, millisecond, or syscall is thermodynamic heat.** Instead of shipping a 300 MB JS bundle with a 20K-token prompt prefix that invalidates prompt caches across directories, Carnot compiles to a single **13 MB** native binary with a **922-token cross-CWD invariant prefix**, **5 canonical tools**, lossless BLAKE3 output shaping, and a fail-closed OS sandbox.

## Why Carnot

- **Fastest wire-to-wire startup & lowest footprint** — `7.4 ms` median to first HTTP request (`9.5 ms` p95), `10.3 MiB` peak RSS, `13.0 MB` stripped binary.
- **97.6%+ prompt cache hit & two-tier cache-warm compaction** — byte-identical static prefix (`carnot audit-prefix`), Hybrid `1h`/`5m` cache breakpoints, Turn-1-only workspace/skills injection (`AGENTS.md`, `CLAUDE.md`, `SKILL.md`), and high-recall semantic compaction that preserves the verbatim user objective, recent tail, CAS handles, and verified workspace state while reusing the warm prompt cache.
- **5 surgical tools, zero context bloat** — `Read` (multimodal images, tree-sitter symbols/outlines, diff-on-reread), `Edit` (atomic line-range + 4-tier resilient patch + CAS undo + tree-sitter syntax guard), `Write`, `Grep`, and `Bash` (confined `cwd`, adaptive-yield job registry, lazy zero-I/O CAS spill, semantic compressors with 100% diagnostic recall).
- **Zero-hardcode modular providers** — protocol adapters (`anthropic-messages`, `openai-completions`, `openai-responses`, `google-generative`, `google-cloudcode`, `subprocess-jsonl`, `faux`) driven purely by [`providers.json`](crates/carnot-ai/catalog/providers.json) + `webpki-root-certs` TLS fallback.
- **Interactive TUI, full MCP & production surfaces** — full-screen Ratatui TUI (streaming Markdown with syntax-highlighted code blocks & tables, compact `Edit` diffs, slash autocomplete, `^P` model picker, interactive `AskUser` & `/approve` modals, live context meter; `--plain` line-REPL fallback), MCP `2026-07-28` / `2025-11-25` / Legacy HTTP+SSE `2024-11-05` (`defer_loading: true` = 0 static tokens), fail-closed macOS Seatbelt / Linux `bwrap`, ACP stdio, Host RPC, Bearer-auth HTTP/SSE server, `carnot doctor`, `SIGINT`/`SIGTERM` cleanup, and `--max-turns` / `--max-cost-usd` / `--compact-at` / `--timeout` guardrails.

## Benchmarks

### Harness Overhead (15 runs + 3 warmups · Darwin arm64 · [`results_8harnesses_v7.json`](benchmarks/results_8harnesses_v7.json))

| Harness | Startup (med) | Startup (p95) | Wall (1-turn) | Peak RSS | Binary | Static Prefix | Cross-CWD Inv. | 16 KB Noise Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **Carnot (Rust)** | **7.4 ms** | **9.5 ms** | **31.5 ms** | **10.3 MiB** | **13.0 MB** | **922 tok** | **Yes** | **2.97 KB** |
| Unharnessed (Rust) | 7.6 ms | 20.5 ms | 36.3 ms | 81.7 MiB | 26.8 MB | 1 035 tok | No | 18.39 KB |
| lythe / rho (Rust) | 41.4 ms | 87.5 ms | 73.2 ms | 46.6 MiB | 32.4 MB | 1 090 tok | Yes | 17.47 KB |
| Codex CLI | 41.4 ms | 48.7 ms | 165.5 ms | 133.7 MiB | 284.4 MB | 9 730 tok | No | — |
| prime-agent (Rust) | 113.8 ms | 154.0 ms | 167.1 ms | 92.6 MiB | 59.2 MB | 9 200 tok | No | 18.40 KB |
| Claude Code | 249.9 ms | 403.4 ms | 293.2 ms | 216.0 MiB | 343.6 MB | 19 210 tok | No | 18.51 KB |
| Pi (TS/Node) | 462.5 ms | 530.2 ms | 488.9 ms | 159.4 MiB | 238.1 MB | 1 399 tok | No | 18.42 KB |
| OpenCode (Bun/TS) | 545.9 ms | 586.0 ms | 652.7 ms | 528.7 MiB | 215.3 MB | 9 644 tok | No | 18.40 KB |

### Terminal-Bench 4 (`claude-opus-5-5` · official `1.0x` timeout · `effort=max` · [`results_tb4_carnot_v8_opus55_max.json`](benchmarks/results_tb4_carnot_v8_opus55_max.json))

Measured on `41 / 62` CPU tasks (native Linux `x86_64`, dedicated 8 vCPU / 16 GB RAM per task, identical `effort=max` & `1.0x` timeout vs public [Unharnessed](https://unharnessed.dev/run) & Claude Code baselines on the same tasks):

| Harness (`Opus 5.5` · `effort=max`) | Raw (41 tasks) | Ex-gateway Turn-1 refusals (38 tasks) | Mean Cost | Prompt Cache Hit | First-Try `Edit` |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Carnot** | **24 / 41 (58.5%)** | **24 / 38 (63.2%)** | **$9.56** | **97.63%** | **258 / 258 (100%)** |
| Unharnessed (public) | 27 / 41 (65.9%) | 25 / 38 (65.8%) | $9.64 | — | — |
| Claude Code (public) | — | — | $11.73 | 96.91% | — |

- Ties Unharnessed head-to-head on **Software (`7/11`)**, **Hardware (`3/4`)**, **Media (`3/3`)**, and **ML (`1/3`)** while costing **18.5% less than Claude Code** with **0 harness crashes**.
- Full `62 / 62` CPU sweep at `effort=medium` (80m cap): `30 / 62` (`48.4%`), `97.00%` cache hit, `122 / 122` (`100%`) `Edit`, `$2.33` mean cost ([`results_tb4_carnot_v6_opus55.json`](benchmarks/results_tb4_carnot_v6_opus55.json)).

### Two-Tier Cache-Warm Semantic Compaction A/B ([`results_compaction_ab_v9.json`](benchmarks/results_compaction_ab_v9.json))

Controlled 4-phase resumed session (`--resume`, `claude-sonnet-5-5`, 240 KB of raw log shards with buried overrides & Phase-1 schema constraints, 14 strict assertions):

| Metric | No Compaction (`--compact-at 0`) | Carnot Cache-Warm Compaction | Delta |
| --- | ---: | ---: | ---: |
| **Verification Score** | `14 / 14` | **`14 / 14`** | **0 amnesia (100% recall)** |
| **Total Net Cost (USD)** | `$0.7475` | **`$0.5593`** | **`-25.2%` (incl. compaction call)** |
| **Post-Compaction Cost (Phases 2–4)** | `$0.2196` | **`$0.0732`** | **`-66.7%` (3.0× cheaper)** |
| **Total Context AUC (tokens)** | `2,660,304` | **`593,369`** | **`-77.7%`** |
| **Context Compression** | `74,112` peak | **`68,775 → 3,059`** | **`-95.6%` (22.5×)** |
| **Post-Phase1 Redundant `Read`s** | `1` | **`0`** | **0 re-reads** |

## Quick Start

```bash
# Build & verify
cargo build --release --bin carnot
./target/release/carnot doctor
./target/release/carnot audit-prefix

# Interactive terminal TUI (`^P` model picker, `^T` thinking, `^O` diffs, `/help`; pass `--plain` for minimalist line REPL)
ANTHROPIC_API_KEY=... carnot --model claude-opus-5-5 --effort high

# One-shot with production guardrails
ANTHROPIC_API_KEY=... carnot --model claude-opus-5-5 --effort medium \
  --max-turns 80 --max-cost-usd 5.0 --compact-at 220000 --timeout 1800 \
  -p "Fix the failing integration tests"

# Prompt from file (safe from /proc/<pid>/cmdline) or stdin
carnot --prompt-file task.txt --mode json
cat task.txt | carnot -p -

# Resume, IDE / Agent Client Protocol, or Bearer-auth HTTP/SSE API
carnot --resume <session-id> -p "Continue"
carnot --mode acp
CARNOT_HTTP_TOKEN=secret carnot serve --addr 127.0.0.1:7744
```

## Architecture (12 Core Crates + Modular Extensions)

```text
Core crates (crates/*)
  carnot-cli         CLI · doctor · audit-prefix · SIGINT/SIGTERM · interactive REPL · single-thread runtime
  carnot             Ergonomic Rust SDK facade (`CarnotBuilder`, opt-in `ext-*` / `all-ext` features)
  carnot-protocol    Bidirectional multi-session multimodal ACP stdio · Host RPC stdio · Bearer-auth HTTP/1.1 + SSE · `ChannelInteractionBroker`
  carnot-agent       Async turn loop · I2 effect-parallel scheduler · lossless EventLog · abort-safe tool pairing
  carnot-ext         Core modules (`ExecMicroSkills`, `PrefixGuard`, `PolicyGuard`, `AskUserModule`, `IsolatedForkWorktree`) + Pi-style drop-in host
  carnot-mcp         MCP 2026-07-28 · 2025-11-25 · Legacy SSE 2024-11-05 · lazy schema cache · SSRF/MCPoison guards
  carnot-tools       5 canonical LID tools (`Read`, `Edit`, `Write`, `.gitignore`-aware `Grep`, `Bash`) · optional `syntax` (tree-sitter) · BLAKE3 ReadTracker
  carnot-ai          ProviderRegistry · protocol adapters (incl. native web search/fetch) · declarative `providers.json` · retry & webpki TLS fallback
  carnot-session     Crash-safe branching JSONL v2 · canonical prefix hash · multimodal compaction
  carnot-sandbox     Fail-closed macOS Seatbelt & Linux `bwrap` · catalog-driven env scrubbing
  carnot-cas         Lazy zero-I/O 256-bit BLAKE3 store (pre-spill redaction + 7d GC + RO `$TMPDIR` fallback) · semantic compressors
  carnot-core        Zero-dependency leaf types (`Message`, `ContentOrigin`, `ToolEffects`, `Usage`, `CarnotConfig`, `UserQuestion`)

Official in-process Rust extensions (extensions/* — opt-in, 0 bloat in default build)
  carnot-ext-web             Hybrid Native + Local Web (`auto|native|local`): native Anthropic `web_search`/`web_fetch`, OpenAI Responses `web_search`, Gemini 3 `googleSearch`/`urlContext`, + SSRF-safe local `WebFetch` (`text/markdown` negotiation, HTML→MD/Text, CAS spill) & API-backed `WebSearch` (SearXNG/Exa/Tavily/Brave)
  carnot-ext-lsp             Non-blocking background LSP warm-up, post-edit diagnostics & deferred `LspQuery`
  carnot-ext-subagent        Disposable-context sync & non-blocking background `SubagentModule` (worktree isolation + cost rollup)
  carnot-ext-hooks           Declarative lifecycle hooks 100% compatible with Claude Code (`command` & `http` webhooks for `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `SessionStart`, `PreCompact`, `Stop`)
  carnot-ext-claude-plugins  Native Claude Code plugin loader (`.claude-plugin/plugin.json`, `hooks.json`, `skills/`, `commands/`, `agents/`, `.mcp.json`, `${CLAUDE_PLUGIN_ROOT}`)
  carnot-ext-memory          Two-tier cache-safe cross-session memory (Turn-1 curated digest + BM25 `MemorySearch` + secret-redacted `MemorySave` + synced `MEMORY.md`)
```

### Interactive Approvals & `AskUser`

Carnot ships first-class interactive user flows across every embedding surface (**0 static prefix tokens**, `defer_loading: true`):

- **Tool approvals (`PermissionBroker`)** : ACP `session/request_permission` (with `allow_once`, session-cached `allow_always`, `reject_once`), Host RPC (`set_require_approval`, `respond_permission`, `list_pending`), HTTP/SSE (`GET /v1/pending`, `POST /v1/permissions`), and Rust SDK (`Agent::set_permission_broker`).
- **Clarifying questions (`AskUser` / `UserQuestionBroker`)** : enabled automatically on interactive surfaces (ACP, RPC, HTTP, REPL; opt-in via `CarnotBuilder::ask_user(true)`). Supports structured multiple-choice options, `multi_select`, and free text — with `carnot/ask_user` on ACP (and automatic fallback to standard `session/request_permission` option buttons on generic ACP clients!), `respond_question` on RPC, `POST /v1/questions` on HTTP, and interactive terminal prompts in the REPL.

### Extensions (Two Tiers)

1. **In-process Rust extensions (`extensions/*`, zero IPC overhead):**
   - Enable at compile time via Cargo features (`ext-web`, `ext-lsp`, `ext-subagent`, `ext-hooks`, `ext-claude-plugins`, `ext-memory`, or `all-ext`), or wire any custom `Arc<dyn Module>` via `CarnotBuilder::with_module(...)`.
   - CLI flags (when compiled in): `--web`, `--web-mode`, `--lsp`, `--subagent`, `--hooks-file`, `--claude-hooks`, `--claude-plugin`, `--load-installed-claude-plugins`, `--memory`.
2. **Pi-style drop-in file extensions (zero recompilation):**
   - Drop any executable script (Python, Bun/TS, Bash) or native binary into `~/.carnot/extensions/`, declare entries in `~/.carnot/extensions.json`, or pass `--ext ./my_ext.py` (workspace `.carnot/extensions/` requires explicit `--trust-workspace-ext`).
   - Speaks `carnot-ext/1` JSON-RPC 2.0 over multiplexed stdio with negotiated hooks (0 IPC on unused hooks) and `defer_loading: true` by default (0 static prefix tokens). Python helper: [`sdk/python/carnot_sdk.py`](sdk/python/carnot_sdk.py).

## Security

See [`SECURITY.md`](SECURITY.md) for the full threat model (fail-closed OS sandbox, filesystem confinement, pre-spill secret redaction, SSRF protection, anti-RCE workspace trust boundaries, and `PolicyGuard` ordering).

## Development

```bash
cargo fmt --all -- --check
cargo check -p carnot-tools --no-default-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
