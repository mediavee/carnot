# Contributing to Carnot

Thank you for your interest in contributing to Carnot!

## Architectural Principles

1. **Zero `unsafe` Rust**: All crates inherit `unsafe_code = "forbid"` from the workspace `Cargo.toml`.
2. **Zero hardcoded providers**: Concrete provider URLs, headers, auth env vars, and `model_routes` belong strictly in [`crates/carnot-ai/catalog/providers.json`](crates/carnot-ai/catalog/providers.json) (enforced by `ci_gate_zero_hardcoded_providers_outside_data_catalog`).
3. **Byte-identical static prefix**: Never inject volatile timestamps, working directories, or dynamic tool lists into the static prefix. Always verify with `carnot audit-prefix`.
4. **Fail-closed security**: OS sandbox denials, path confinement, `.git` write protection, SSRF validation, and HTTP auth/CSRF checks must fail closed.

## Local Verification

Requires Rust **1.90+** (edition 2024):

```bash
cargo fmt --all -- --check
cargo check -p carnot-tools --no-default-features
cargo check -p carnot-cli --no-default-features
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --lib
cargo test --workspace --all-features
```

To verify the Python SDK against a local build:

```bash
cargo build --bin carnot
CARNOT_BIN="$PWD/target/debug/carnot" python3 -m unittest discover -s sdk/python -v
```

## Licensing

By submitting a contribution, you agree that your contribution will be dual-licensed under **MIT OR Apache-2.0**, as specified in [`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE).
