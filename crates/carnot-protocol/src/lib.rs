//! Agent integration protocols:
//!   - `acp`: full bidirectional Agent Client Protocol (ACP) over stdio (`session/request_permission` + `carnot/ask_user`)
//!   - `rpc`: Host JSON-RPC 2.0 over stdio (concurrent `steer`/`abort`/`respond_question`/`respond_permission`)
//!   - `http`: Bearer-authenticated HTTP/1.1 + SSE API server
//!   - `interaction`: shared async broker for interactive approvals & `AskUser` questions

pub mod acp;
pub mod http;
pub mod interaction;
pub mod rpc;
