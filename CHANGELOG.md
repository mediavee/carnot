# Changelog

All notable changes to Carnot are documented in this file.

## [0.1.0] - 2026-11-24

### Added
- **Core Rust Agent Engine (`#![forbid(unsafe_code)]`)**: Effect-parallel async turn loop, lossless `EventLog`, crash-safe branching JSONL v2 session store, and BLAKE3 content-addressed spill store with pre-spill secret redaction.
- **Prefix-Invariant Prompt Caching**:
  - Cross-directory byte-identical static prefix verified by `carnot audit-prefix`.
  - Provider-agnostic `CacheTtl` retention tiers (`short`, `hybrid`, `long`, with `5m`/`1h`/`24h` aliases) configurable via CLI (`--cache-ttl`), `~/.carnot/config.toml`, TUI (`/cache`), RPC (`set_cache_ttl`), HTTP (`POST /v1/cache-ttl`), and ACP (`cache_ttl`).
  - `SessionMetrics::prefix_retention_rate()`, runtime `CacheBreakDetected` monitoring, and smart subagent cache retention policies.
  - Two-tier cache-warm semantic compaction.
- **5 Surgical Core Tools**: Multimodal `Read` with tree-sitter outlines, `ReadTracker` (absolute-line re-read diffs + shift notices, symbol/outline tracking), atomic fuzzy & harness-anchored line-range `Edit` (implicit line verification + unique relocation with zero extra prompt/output tokens, CAS undo), `Write` with `.git` protection and unread/stale-file overwrite guard, `.gitignore`-aware `Grep`, and sandboxed `Bash`.
- **Sparse Calendar Date Grounding**: Persisted `[date: YYYY-MM-DD]` note injected once on Turn 1, preserved across compaction, and re-emitted only on calendar day change — keeping the static system prompt 100% byte-invariant.
- **Fail-Closed OS Sandboxing**: macOS Seatbelt and Linux `bwrap` (protecting credentials such as `~/.ssh`, `~/.aws`, `~/.carnot/auth.json`, `~/.carnot/server.token` and locking `.git` read-only), plus Linux Landlock (ABI v1–v6) self-exec fallback for containers where `bwrap` is unavailable (note: Landlock is additive allow-only and cannot carve `.git` out of a writable workspace; in-process `Edit`/`Write` always block `.git` writes).
- **21 Built-in Providers (+ offline `faux`) & Multi-Protocol `model_routes`**: Anthropic, OpenAI Completions & Stateless Responses, Google Gemini & Cloud Code, Grok / xAI (`grok-4.7` Responses + sticky `x-grok-conv-id`), Muse / Meta (`api.meta.ai/v1`), Mistral (`alnum9` tool IDs + structured thinking), OpenCode Zen & OpenCode Go multi-protocol gateways, OpenRouter, DeepSeek, Kimi, Qwen, Groq, Fireworks, Together, Cerebras, Ollama, LM Studio, vLLM, and offline `faux`, plus OAuth 2.0 (`carnot auth`).
- **Built-in & Drop-in Extensions**: `web`, `lsp`, `subagent`, `memory`, `codemode` (isolated QuickJS async sandbox with `Promise.all`, `tools.*`, `searchTools`/`describeTool`, transactional `store`/`load`, gated via `Agent::invoke_tool_gated`, and BLAKE3 CAS spill), `goal` (`carnot-ext-goal` — autonomous verification loop combining deterministic sandboxed `--goal-cmd` checks, an independent virgin-context Worker ≠ Verifier LLM judge with cross-model routing via `--goal-model`, BLAKE3 stagnation detection, and 0 bytes added to the static cached prefix), `hooks`, `claude-plugins`, full MCP (`2026-07-28`, `2025-11-25`, legacy SSE), and `carnot-ext/1` drop-in file extensions.
- **Multi-Surface UX, SDKs & Benchmarks**: Dedicated optional `carnot-tui` crate (live mid-turn steer & follow-up, `/context`, `/cache`, `/fleet`, `/fork`, `/undo`, `^P` model picker), plain REPL, ACP stdio, Host RPC stdio, Bearer-authenticated HTTP/SSE server (`carnot serve`), zero-dependency Python & TypeScript SDKs, `~/.carnot/config.toml`, opt-in cost-capped live smoke suite (`scripts/live_smoke.py`), and 3-tier budget-friendly benchmarking (`benchmarks/README.md`, `benchmarks/harbor_carnot_agent.py`, `benchmarks/summarize_harbor_run.py`).
