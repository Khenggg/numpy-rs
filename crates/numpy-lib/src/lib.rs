//! Sorting, searching, reductions, statistics, histograms and set operations.
//!
//! Functions accept typed arrays; [`dynamic`] supplies dtype-dispatched reductions.
//! Integer sums/products widen to i64/u64. Statistics retain floating/complex
//! output dtypes and promote integers to f64. Indices use i64.
pub use numpy_types::*;

pub mod dynamic;
pub mod histogram;
pub mod io;
mod lanes;
pub mod polynomial;
pub mod reductions;
pub mod sets;
pub mod signal;
pub mod sorting;
pub mod statistics;

pub use histogram::*;
pub use io::*;
pub use polynomial::*;
pub use reductions::*;
pub use sets::*;
pub use signal::*;
pub use sorting::*;
pub use statistics::*;
