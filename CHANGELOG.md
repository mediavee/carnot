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
- **5 Surgical Core Tools**: Multimodal `Read` with tree-sitter outlines and `ReadTracker`, atomic fuzzy `Edit` with CAS undo, `Write` with `.git` write protection, `.gitignore`-aware `Grep`, and sandboxed `Bash`.
- **Fail-Closed OS Sandboxing**: macOS Seatbelt and Linux `bwrap` (protecting credentials such as `~/.ssh`, `~/.aws`, `~/.carnot/auth.json`, `~/.carnot/server.token` and locking `.git` read-only), plus Linux Landlock (ABI v1–v6) self-exec fallback for containers where `bwrap` is unavailable (note: Landlock is additive allow-only and cannot carve `.git` out of a writable workspace; in-process `Edit`/`Write` always block `.git` writes).
- **21 Built-in Providers (+ offline `faux`) & Multi-Protocol `model_routes`**: Anthropic, OpenAI Completions & Stateless Responses, Google Gemini & Cloud Code, Grok / xAI (`grok-4.7` Responses + sticky `x-grok-conv-id`), Muse / Meta (`api.meta.ai/v1`), Mistral (`alnum9` tool IDs + structured thinking), OpenCode Zen & OpenCode Go multi-protocol gateways, OpenRouter, DeepSeek, Kimi, Qwen, Groq, Fireworks, Together, Cerebras, Ollama, LM Studio, vLLM, and offline `faux`, plus OAuth 2.0 (`carnot auth`).
- **Built-in & Drop-in Extensions**: `web`, `lsp`, `subagent`, `memory`, `hooks`, `claude-plugins`, full MCP (`2026-07-28`, `2025-11-25`, legacy SSE), and `carnot-ext/1` drop-in file extensions.
- **Multi-Surface UX & SDKs**: Ratatui TUI (live mid-turn steer & follow-up, `/context`, `/cache`, `/fleet`, `/fork`, `/undo`, `^P` model picker), plain REPL, ACP stdio, Host RPC stdio, Bearer-authenticated HTTP/SSE server (`carnot serve`), zero-dependency Python & TypeScript SDKs, and `~/.carnot/config.toml`.
