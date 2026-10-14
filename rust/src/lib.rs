//! Exact CPU reference and range-readable archive core. GPU kernels live separately.
pub mod archive;
pub mod codec;
pub mod dense6;
pub mod encode;
pub mod fiber;
pub mod pawnless;
pub mod range_cache;
pub mod retrograde;

pub type Result<T> = std::result::Result<T, String>;
