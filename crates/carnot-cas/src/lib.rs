//! Atomic 256-bit BLAKE3 Content-Addressed File Handles, Lossless Output Shaper,
//! and Shared Secret Redaction.

pub mod redact;
pub mod shape;
pub mod store;

pub use redact::{redact_secrets, redact_secrets_cow};
pub use shape::{
    ShapedOutput, clean_terminal_noise, shape_command_output, shape_command_output_with_redaction,
    shape_text_output,
};
pub use store::{CasHandle, CasStore};
