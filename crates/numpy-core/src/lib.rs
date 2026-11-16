//! Core array creation, manipulation, indexing, Einstein summation, memory overlap/memmap,
//! and pretty formatting for `NdArray<T>` and `DynArray`.

pub use numpy_types::*;

pub mod creation;
pub mod dynamic;
pub mod einsum;
pub mod ext;
pub mod format;
pub mod indexing;
pub mod manipulation;
pub mod memory;
pub mod util;

pub use creation::*;
pub use einsum::*;
pub use ext::{DynArrayExt, NdArrayExt};
pub use format::*;
pub use indexing::*;
pub use manipulation::*;
pub use memory::*;
pub use util::{DynScalar, RAYON_THRESHOLD};
