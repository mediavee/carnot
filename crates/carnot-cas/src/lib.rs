//! Atomic 256-bit BLAKE3 Content-Addressed File Handles, Lossless Output Shaper,
//! and Incremental Path-Inclusive Worktree Hasher (S1).

pub mod shape;
pub mod store;
pub mod worktree;

pub use shape::{ShapedOutput, clean_terminal_noise, shape_command_output, shape_text_output};
pub use store::{CasHandle, CasStore};
pub use worktree::{WorktreeHasher, WorktreeSnapshot};
