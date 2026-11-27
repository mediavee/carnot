# Security

All Carnot workspace crates and built-in extensions set `unsafe_code = "forbid"`. Optional C bindings exist in `tree-sitter` parsers (`carnot-tools` default feature) and the opt-in `carnot-ext-codemode` QuickJS extension.

## Boundaries

1. **OS sandbox (`carnot-sandbox`, fail-closed)**
   - `Bash` runs inside macOS Seatbelt (`sandbox-exec`), Linux `bwrap`, or Linux Landlock (ABI v1–v6) when `bwrap` cannot create user namespaces inside a container.
   - Outbound network is off by default (`--allow-network` to enable). Writes are limited to the workspace, and common credential paths (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.carnot/auth.json`, `~/.carnot/server.token`, `$CARGO_HOME/credentials*`, etc.) are blocked.
   - Seatbelt and `bwrap` keep `<workspace>/.git` read-only. Landlock is allow-only and cannot carve a read-only `.git` out of a writable workspace (`carnot doctor` warns when Landlock is active). In-process `Edit` and `Write` always reject `.git` writes.
   - Provider API keys discovered from `providers.json` are scrubbed from child environments.
   - **Windows:** no OS sandbox backend is implemented yet. `carnot-sandbox` fails closed unless `--no-sandbox` is passed. In-process path confinement and `.git` write protection remain active.

2. **Filesystem confinement (`carnot-tools`)**
   - `Read`, `Edit`, `Write`, and `Grep` reject `..` traversal, out-of-workspace absolute paths, and dangling symlinks escaping `workspace_root`.

3. **Secret redaction (`carnot-cas`, `carnot-ext-memory`, `carnot-ext/policy`)**
   - Common tokens, API keys, Bearer headers, and PEM private keys are redacted before spilling outputs to `.carnot/h/` or saving memory entries.

4. **SSRF checks (`carnot-mcp`, `carnot-ext-web`)**
   - `WebFetch` and workspace MCP HTTP clients resolve DNS and reject loopback, private, link-local, and metadata IPs (such as `169.254.169.254`), re-check redirects (max 5), and cap response bodies at 2 MiB.

5. **Workspace trust (opt-in)**
   - Carnot never executes workspace-local `.mcp.json`, `.carnot/extensions/`, `.carnot/hooks.json`, or `.claude/settings.json` unless you pass `--trust-workspace-mcp`, `--trust-workspace-ext`, or `--trust-workspace-hooks`.
   - Explicitly trusted hooks, drop-in extensions, and stdio MCP servers run with your user privileges outside the `Bash` sandbox.

6. **PolicyGuard ordering**
   - When enabled, `PolicyGuardModule` runs last in `before_tool_call` so earlier extensions cannot bypass it via `RewriteInput`.

7. **HTTP/SSE server (`carnot serve`)**
   - Uses `hyper` over `127.0.0.1` with constant-time Bearer token auth (`CARNOT_HTTP_TOKEN` or an auto-generated `0600` token at `~/.carnot/server.token`), strict loopback `Host` and `Origin` checks, `Content-Type: application/json` enforcement on POST bodies, and bounded connections, headers, body size (8 MiB), and read timeouts. Unauthenticated requests are rejected before reading the body.

## Reporting a vulnerability

Open a private security advisory on GitHub or contact the maintainers directly with reproduction steps. Do not post exploit details in public issues before a fix is available.
