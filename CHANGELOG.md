# Changelog

## [0.1.0] - 2026-11-26

Initial release.

### Core runtime
- Async turn loop with effect-based parallel tool scheduling, live tool progress events, and a branching JSONL session log (`carnot-agent`, `carnot-session`).
- Static 757-char system prompt and 5 core tool schemas (`Read`, `Edit`, `Write`, `Grep`, `Bash`) kept byte-identical across directories, dates, and sessions (`carnot audit-prefix`).
- Configurable prompt-cache TTL (`short`, `hybrid`, `long`) and two-tier compaction that reuses the warm prefix.
- Content-addressed BLAKE3 spill store (`carnot-cas`) with secret redaction before writing to disk and deterministic output compressors for common test/build tools.

### Sandboxing & security
- OS sandbox for `Bash` (`carnot-sandbox`): macOS Seatbelt, Linux `bwrap`, and Linux Landlock (ABI v1–v6) fallback inside containers. Fails closed when no backend is available unless `--no-sandbox` is passed.
- Path confinement and `.git` write protection in `Read`, `Edit`, `Write`, and `Grep`.
- Workspace-local MCP configs, hooks, and drop-in extensions require explicit opt-in flags.

### Providers, protocols & extensions
- Data-driven provider catalog (`crates/carnot-ai/catalog/providers.json`) with 21 providers, multi-protocol `model_routes`, OAuth 2.0 (`carnot auth`), and offline `faux` provider for tests.
- Transports (`carnot-protocol`): ACP stdio, JSON-RPC stdio, and `hyper`-based HTTP/SSE server (`carnot serve`) with Bearer auth and localhost Host/Origin checks.
- SDKs: Rust (`carnot::CarnotBuilder`), Python (`sdk/python`), and TypeScript (`sdk/ts`).
- Built-in extensions: `web`, `lsp`, `subagent`, `memory`, `goal`, `hooks`, `claude-plugins`, and opt-in `codemode` (`--features ext-codemode`), plus MCP client and `carnot-ext/1` drop-in file extensions.
- Optional Ratatui TUI (`carnot-tui`) and plain REPL (`carnot --plain`).
