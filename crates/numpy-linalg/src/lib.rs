//! High-performance linear algebra (`numpy.linalg` & BLAS/LAPACK ops) in pure Rust
//! backed by `faer` 0.21, `matrixmultiply`, and `rayon`.

pub use numpy_types::*;

pub mod batch;
pub mod decomp;
pub mod norms;
pub mod products;
pub mod solve;
pub mod traits;

pub use decomp::{
    cholesky, cholesky_dyn, eig, eig_dyn, eigh, eigh_dyn, eigvals, eigvals_dyn, eigvalsh,
    eigvalsh_dyn, qr, qr_dyn, svd, svd_dyn, svdvals, svdvals_dyn,
};
pub use norms::{
    cond, cond_dyn, det, det_dyn, matrix_norm, matrix_norm_dyn, matrix_rank, matrix_rank_dyn,
    norm, norm_dyn, slogdet, slogdet_dyn, vector_norm, vector_norm_dyn,
};
pub use products::{
    cross, cross_dyn, dot, dot_dyn, inner, inner_dyn, kron, kron_dyn, linalg_outer,
    linalg_outer_dyn, matmul, matmul_dyn, matvec, matvec_dyn, multi_dot, multi_dot_dyn, outer,
    outer_dyn, tensordot, tensordot_dyn, trace, trace_axes, trace_axes_dyn, trace_dyn, vdot,
    vdot_dyn, vecdot, vecdot_dyn, vecmat, vecmat_dyn,
};
pub use solve::{
    inv, inv_dyn, lstsq, lstsq_dyn, matrix_power, matrix_power_dyn, pinv, pinv_dyn, solve,
    solve_dyn, tensorinv, tensorinv_dyn, tensorsolve, tensorsolve_dyn,
};
pub use traits::{
    EigOutput, EigResult, EighOutput, EighResult, LinalgReal, LinalgScalar, LstsqOutput,
    LstsqResult, NormOrd, ProductScalar, QrMode, QrOutput, QrResult, SlogdetOutput, SlogdetResult,
    SvdOutput, SvdResult, TensordotAxes, Uplo, linalg_common_dtype, linalg_common_dtype_pair,
    trace_default_dtype,
};
