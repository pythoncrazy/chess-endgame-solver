//! Pawnless endgame tables: the CPU solver, the table files and the packed format.
//! The GPU solver is in `gpu/`. The independent verifier is in `verify/`.
pub mod cm;
pub mod packed;
pub mod pawnless;

pub type Result<T> = std::result::Result<T, String>;
