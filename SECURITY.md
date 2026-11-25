# Security Policy & Threat Model

Carnot is built with `#![forbid(unsafe_code)]` across every core crate and official extension, and enforces defense-in-depth at every boundary.

## Security Boundaries

1. **OS Kernel Sandbox (`carnot-sandbox` — fail-closed by default)**
   - Every `Bash` command runs inside macOS Seatbelt (`sandbox-exec`), Linux `bwrap` (`bubblewrap`), or — when `bwrap` is unavailable or blocked by container seccomp (`CLONE_NEWUSER` denied) — safe Linux Landlock (ABI v1–v6, including in-kernel TCP restrictions on ABI v4+ and signal/socket scoping on ABI v6+).
   - Network is disabled by default (`--allow-network` required for outbound access), writes are confined to the workspace, and credential paths (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.carnot/auth.json`, `~/.carnot/server.token`, `$CARGO_HOME/credentials*`, `$CARNOT_AUTH_FILE`, etc.) are blocked.
   - macOS Seatbelt and Linux `bwrap` mount/lock `<workspace>/.git` read-only. Because Linux Landlock is strictly additive (allow-only) and cannot carve out read-only subtrees beneath a writable root, the Landlock fallback cannot make `<workspace>/.git` read-only for `Bash` commands (and binds `/proc/self` to the trampoline PID); prefer `bwrap` on Linux when strict `.git` isolation is required (`carnot doctor` warns whenever Landlock fallback is active). In-process `Edit`/`Write` tools always reject `.git` writes regardless of sandbox backend.
   - Provider API keys and sensitive env vars (discovered dynamically from `providers.json`) are scrubbed from child environments.

2. **Filesystem Confinement (`carnot-tools`)**
   - `Read`, `Edit`, `Write`, and `Grep` strictly confine paths inside `workspace_root` (blocking `..` traversal, absolute escapes, and dangling symlinks).

3. **Secret Redaction Before Disk Persistence (`carnot-cas`, `carnot-ext-memory`, `carnot-ext/policy`)**
   - Well-known API tokens, keys, and credentials (`sk-ant-*`, `sk-*`, `ghp_*`, `AKIA*`, Bearer tokens, private key headers) are redacted **before** spilling large tool outputs to `.carnot/h/` or saving entries in `carnot-ext-memory`.

4. **SSRF & Network Hardening (`carnot-mcp`, `carnot-ext-web`)**
   - `WebFetch` and workspace MCP HTTP clients validate public IPs at DNS resolution time, reject loopback/private/link-local/metadata ranges (e.g. `169.254.169.254`), re-validate every redirect hop (max 5), and cap decompressed responses at 2 MiB.

5. **Anti-RCE Workspace Trust Boundaries (Opt-In Only)**
   - Untrusted repositories may contain `.mcp.json`, `.carnot/extensions/`, `.carnot/hooks.json`, or `.claude/settings.json`.
   - Carnot **NEVER** executes workspace-local MCP servers, drop-in extensions, or shell hooks unless the user explicitly opts in (`--trust-workspace-mcp`, `--trust-workspace-ext`, `--trust-workspace-hooks`).
   - Note: Just like Claude Code and Pi, explicitly trusted hooks, drop-in extensions, and stdio MCP servers run with the user's local privileges outside the `Bash` sandbox. Only enable workspace trust flags on repositories you trust.

6. **PolicyGuard Invariant**
   - When enabled, `PolicyGuardModule` is always ordered **last** in the `before_tool_call` chain so no earlier extension or hook can bypass it via `RewriteInput`.

7. **Embedded HTTP/SSE Server (`carnot serve`)**
   - Requires constant-time Bearer token auth (`CARNOT_HTTP_TOKEN` or auto-generated high-entropy random `0600` token), enforces strict localhost `Host`/`Origin` CSRF checks and `Content-Type: application/json`, rejects unauthenticated requests before body allocation, and bounds concurrent connections, headers, body size, and read timeouts.

## Reporting a Vulnerability

Please open a private security advisory on the repository (or contact the maintainers directly) with reproduction steps and affected crate/extension. Do not post exploit details in public issues before a fix is available.
