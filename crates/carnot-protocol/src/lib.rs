//! Multi-protocol transport layer exposing one Carnot runtime over:
//!   1. ACP (Agent Client Protocol) JSON-RPC 2.0 over stdio (Zed, T3 Code, VS Code)
//!   2. Host RPC JSON-RPC 2.0 over stdio (Pi/Lythe-compatible host bridge)
//!   3. Embedded zero-bloat HTTP/1.1 + SSE API Server (`/health`, `/v1/metrics`, `/v1/prompt`)

pub mod acp;
pub mod http;
pub mod rpc;
