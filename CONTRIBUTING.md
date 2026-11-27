# Contributing

## Rules

1. **No `unsafe` Rust.** Every workspace crate inherits `unsafe_code = "forbid"` from `Cargo.toml`.
2. **No hardcoded providers.** Provider URLs, headers, auth env vars, and `model_routes` live in [`crates/carnot-ai/catalog/providers.json`](crates/carnot-ai/catalog/providers.json) (enforced by `ci_gate_zero_hardcoded_providers_outside_data_catalog`).
3. **Invariant static prefix.** Do not put dates, working directories, or dynamic tool lists in the static system prompt. Verify with `carnot audit-prefix`.
4. **Fail closed.** Sandbox denials, path confinement, `.git` write guards, SSRF checks, and HTTP auth/CSRF checks must fail closed.

## Checks

Requires Rust 1.90+ (edition 2024):

```bash
cargo fmt --all -- --check
cargo check -p carnot-tools --no-default-features
cargo check -p carnot-cli --no-default-features
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --lib
cargo test --workspace --all-features
python3 scripts/validate_catalog.py
```

Python SDK test against a local build:

```bash
cargo build --bin carnot
CARNOT_BIN="$PWD/target/debug/carnot" python3 -m unittest discover -s sdk/python -v
```

## License

Contributions are dual-licensed under MIT or Apache-2.0 ([`LICENSE-MIT`](LICENSE-MIT), [`LICENSE-APACHE`](LICENSE-APACHE)).
