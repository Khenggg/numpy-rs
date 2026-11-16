//! Core scalar traits, enum options, result structs, and NumPy dtype promotion helpers.

use faer::Conj;
use faer::diag::{Diag, DiagMut};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::cholesky::llt;
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::linalg::householder;
use faer::linalg::lu::partial_pivoting::{factor as lu_factor, inverse as lu_inv, solve as lu_solve};
use faer::linalg::qr::no_pivoting::factor as qr_factor;
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::perm::PermRef;
use faer::{Mat, MatMut, MatRef, Par};
use num_complex::{Complex, Complex32, Complex64};
use numpy_types::{DType, DynArray, NdArray, NumPyError, Result, Scalar, f16, promote_types};

/// Matrix triangle selection for Hermitian/symmetric eigenvalue routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Uplo {
    #[default]
    Lower,
    Upper,
}

impl Uplo {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "L" | "l" => Ok(Self::Lower),
            "U" | "u" => Ok(Self::Upper),
            _ => Err(NumPyError::InvalidValue(format!(
                "UPLO argument must be 'L' or 'U', got '{s}'"
            ))),
        }
    }
}

/// Mode for QR decomposition, matching `numpy.linalg.qr(..., mode=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QrMode {
    #[default]
    Reduced,
    Complete,
    R,
    Raw,
}

impl QrMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "reduced" => Ok(Self::Reduced),
            "complete" => Ok(Self::Complete),
            "r" => Ok(Self::R),
            "raw" => Ok(Self::Raw),
            _ => Err(NumPyError::InvalidValue(format!(
                "Unrecognized QR mode '{s}'; expected 'reduced', 'complete', 'r', or 'raw'"
            ))),
        }
    }
}

/// Norm order parameter matching `numpy.linalg.norm` and `cond`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NormOrd {
    /// `None` in NumPy: Frobenius norm for matrices, 2-norm for vectors/flattened arrays.
    #[default]
    Default,
    /// `'fro'`: Frobenius norm (matrices only).
    Fro,
    /// `'nuc'`: Nuclear norm (sum of singular values, matrices only).
    Nuc,
    /// `+inf`: max row sum (matrix) or max |x| (vector).
    PosInf,
    /// `-inf`: min row sum (matrix) or min |x| (vector).
    NegInf,
    /// `1`: max col sum (matrix) or L1 (vector).
    L1,
    /// `-1`: min col sum (matrix) or harmonic min-like (vector).
    LNeg1,
    /// `2`: spectral norm / largest singular value (matrix) or L2 (vector).
    L2,
    /// `-2`: smallest singular value (matrix) or vector (-2)-norm.
    LNeg2,
    /// `0`: count of nonzero elements (vectors only).
    Zero,
    /// Arbitrary finite real `p` (vectors only).
    P(f64),
}

impl NormOrd {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "fro" => Ok(Self::Fro),
            "nuc" => Ok(Self::Nuc),
            "inf" | "+inf" => Ok(Self::PosInf),
            "-inf" => Ok(Self::NegInf),
            _ => Err(NumPyError::InvalidValue(format!(
                "Invalid norm order string '{s}'"
            ))),
        }
    }

    pub fn from_f64(ord: f64) -> Self {
        if ord == f64::INFINITY {
            Self::PosInf
        } else if ord == f64::NEG_INFINITY {
            Self::NegInf
        } else if ord == 0.0 {
            Self::Zero
        } else if ord == 1.0 {
            Self::L1
        } else if ord == -1.0 {
            Self::LNeg1
        } else if ord == 2.0 {
            Self::L2
        } else if ord == -2.0 {
            Self::LNeg2
        } else {
            Self::P(ord)
        }
    }
}

/// Axis specification for `tensordot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensordotAxes {
    /// Contract the last `n` axes of `a` with the first `n` axes of `b`.
    Num(usize),
    /// Contract explicit axis lists `(axes_a, axes_b)`.
    Axes(Vec<isize>, Vec<isize>),
}

impl From<usize> for TensordotAxes {
    fn from(n: usize) -> Self {
        Self::Num(n)
    }
}

impl From<(Vec<isize>, Vec<isize>)> for TensordotAxes {
    fn from((a, b): (Vec<isize>, Vec<isize>)) -> Self {
        Self::Axes(a, b)
    }
}

impl From<(&[isize], &[isize])> for TensordotAxes {
    fn from((a, b): (&[isize], &[isize])) -> Self {
        Self::Axes(a.to_vec(), b.to_vec())
    }
}

/// Typed output of `qr`.
///
/// For `QrMode::Raw`, reflectors `(h, tau)` follow `faer`'s Hermitian Householder convention
/// (`H = I - tau * v * v^H` with real `tau`), which reconstructs unitary `Q` and upper-triangular
/// `R = triu(h.T)` cleanly for both real and complex matrices (differing in diagonal phase from
/// LAPACK's non-Hermitian complex reflector convention, `F-B2-06`).
#[derive(Debug, Clone)]
pub enum QrOutput<T: Scalar> {
    QR { q: NdArray<T>, r: NdArray<T> },
    R { r: NdArray<T> },
    Raw { h: NdArray<T>, tau: NdArray<T> },
}

/// Dynamic output of `qr_dyn`.
#[derive(Debug, Clone)]
pub enum QrResult {
    QR { q: DynArray, r: DynArray },
    R { r: DynArray },
    Raw { h: DynArray, tau: DynArray },
}

/// Typed output of `svd`.
#[derive(Debug, Clone)]
pub struct SvdOutput<T: LinalgScalar> {
    pub u: Option<NdArray<T>>,
    pub s: NdArray<T::Real>,
    pub vh: Option<NdArray<T>>,
}

/// Dynamic output of `svd_dyn`.
#[derive(Debug, Clone)]
pub struct SvdResult {
    pub u: Option<DynArray>,
    pub s: DynArray,
    pub vh: Option<DynArray>,
}

/// Typed output of `eig`.
#[derive(Debug, Clone)]
pub struct EigOutput<T: LinalgScalar> {
    pub eigenvalues: NdArray<T::Complex>,
    pub eigenvectors: NdArray<T::Complex>,
}

/// Dynamic output of `eig_dyn`.
#[derive(Debug, Clone)]
pub struct EigResult {
    pub eigenvalues: DynArray,
    pub eigenvectors: DynArray,
}

/// Typed output of `eigh`.
#[derive(Debug, Clone)]
pub struct EighOutput<T: LinalgScalar> {
    pub eigenvalues: NdArray<T::Real>,
    pub eigenvectors: NdArray<T>,
}

/// Dynamic output of `eigh_dyn`.
#[derive(Debug, Clone)]
pub struct EighResult {
    pub eigenvalues: DynArray,
    pub eigenvectors: DynArray,
}

/// Typed output of `slogdet`.
#[derive(Debug, Clone)]
pub struct SlogdetOutput<T: LinalgScalar> {
    pub sign: NdArray<T>,
    pub logabsdet: NdArray<T::Real>,
}

/// Dynamic output of `slogdet_dyn`.
#[derive(Debug, Clone)]
pub struct SlogdetResult {
    pub sign: DynArray,
    pub logabsdet: DynArray,
}

/// Typed output of `lstsq` (`rank` uses `i32` matching NumPy `np.intc`, `FIX(F-B2-04)`).
#[derive(Debug, Clone)]
pub struct LstsqOutput<T: LinalgScalar> {
    pub x: NdArray<T>,
    pub residuals: NdArray<T::Real>,
    pub rank: NdArray<i32>,
    pub s: NdArray<T::Real>,
}

/// Dynamic output of `lstsq_dyn`.
#[derive(Debug, Clone)]
pub struct LstsqResult {
    pub x: DynArray,
    pub residuals: DynArray,
    pub rank: DynArray,
    pub s: DynArray,
}

/// Real floating-point scalar supported by `numpy-linalg` (`f32`, `f64`).
pub trait LinalgReal: Scalar + ProductScalar + PartialOrd + Copy + Send + Sync + 'static {
    fn eps() -> Self;
    fn is_finite_val(self) -> bool;
    fn is_nan_val(self) -> bool;
    fn abs_val(self) -> Self;
    fn sqrt_val(self) -> Self;
    fn ln_val(self) -> Self;
    fn exp_val(self) -> Self;
    fn powf_val(self, p: Self) -> Self;
    fn pos_inf() -> Self;
    fn neg_inf() -> Self;
    fn nan_val() -> Self;
    fn max_val(self, other: Self) -> Self;
    fn min_val(self, other: Self) -> Self;
    fn real_into_dyn(arr: NdArray<Self>) -> DynArray;
}

impl LinalgReal for f32 {
    #[inline]
    fn eps() -> Self {
        f32::EPSILON
    }
    #[inline]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn is_nan_val(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn abs_val(self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt_val(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn ln_val(self) -> Self {
        self.ln()
    }
    #[inline]
    fn exp_val(self) -> Self {
        self.exp()
    }
    #[inline]
    fn powf_val(self, p: Self) -> Self {
        self.powf(p)
    }
    #[inline]
    fn pos_inf() -> Self {
        f32::INFINITY
    }
    #[inline]
    fn neg_inf() -> Self {
        f32::NEG_INFINITY
    }
    #[inline]
    fn nan_val() -> Self {
        f32::NAN
    }
    #[inline]
    fn max_val(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            f32::NAN
        } else {
            self.max(other)
        }
    }
    #[inline]
    fn min_val(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            f32::NAN
        } else {
            self.min(other)
        }
    }
    #[inline]
    fn real_into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::F32(arr)
    }
}

impl LinalgReal for f64 {
    #[inline]
    fn eps() -> Self {
        f64::EPSILON
    }
    #[inline]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn is_nan_val(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn abs_val(self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt_val(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn ln_val(self) -> Self {
        self.ln()
    }
    #[inline]
    fn exp_val(self) -> Self {
        self.exp()
    }
    #[inline]
    fn powf_val(self, p: Self) -> Self {
        self.powf(p)
    }
    #[inline]
    fn pos_inf() -> Self {
        f64::INFINITY
    }
    #[inline]
    fn neg_inf() -> Self {
        f64::NEG_INFINITY
    }
    #[inline]
    fn nan_val() -> Self {
        f64::NAN
    }
    #[inline]
    fn max_val(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            f64::NAN
        } else {
            self.max(other)
        }
    }
    #[inline]
    fn min_val(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            f64::NAN
        } else {
            self.min(other)
        }
    }
    #[inline]
    fn real_into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::F64(arr)
    }
}

/// Real output scalar supported by `norm` / `vector_norm` / `matrix_norm` (`f16`, `f32`, `f64`).
pub trait NormRealScalar: Scalar + PartialEq + Copy + Send + Sync + 'static {
    const ZERO_R: Self;
    const POS_INF_R: Self;
    const NAN_R: Self;
    fn is_nan_r(self) -> bool;
    fn add_r(self, rhs: Self) -> Self;
    fn mul_r(self, rhs: Self) -> Self;
    fn sqrt_r(self) -> Self;
    fn powf_r(self, p: f64) -> Self;
    fn gt_r(self, rhs: Self) -> bool;
    fn lt_r(self, rhs: Self) -> bool;
    fn from_f64_r(v: f64) -> Self;
}

impl NormRealScalar for f16 {
    const ZERO_R: Self = f16::from_bits(0);
    const POS_INF_R: Self = f16::INFINITY;
    const NAN_R: Self = f16::NAN;
    #[inline]
    fn is_nan_r(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn add_r(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32() + rhs.to_f32())
    }
    #[inline]
    fn mul_r(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32() * rhs.to_f32())
    }
    #[inline]
    fn sqrt_r(self) -> Self {
        f16::from_f32(self.to_f32().sqrt())
    }
    #[inline]
    fn powf_r(self, p: f64) -> Self {
        f16::from_f32(self.to_f32().powf(p as f32))
    }
    #[inline]
    fn gt_r(self, rhs: Self) -> bool {
        self.to_f32() > rhs.to_f32()
    }
    #[inline]
    fn lt_r(self, rhs: Self) -> bool {
        self.to_f32() < rhs.to_f32()
    }
    #[inline]
    fn from_f64_r(v: f64) -> Self {
        f16::from_f32(v as f32)
    }
}

impl NormRealScalar for f32 {
    const ZERO_R: Self = 0.0;
    const POS_INF_R: Self = f32::INFINITY;
    const NAN_R: Self = f32::NAN;
    #[inline]
    fn is_nan_r(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn add_r(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn mul_r(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn sqrt_r(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn powf_r(self, p: f64) -> Self {
        self.powf(p as f32)
    }
    #[inline]
    fn gt_r(self, rhs: Self) -> bool {
        self > rhs
    }
    #[inline]
    fn lt_r(self, rhs: Self) -> bool {
        self < rhs
    }
    #[inline]
    fn from_f64_r(v: f64) -> Self {
        v as f32
    }
}

impl NormRealScalar for f64 {
    const ZERO_R: Self = 0.0;
    const POS_INF_R: Self = f64::INFINITY;
    const NAN_R: Self = f64::NAN;
    #[inline]
    fn is_nan_r(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn add_r(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn mul_r(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn sqrt_r(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn powf_r(self, p: f64) -> Self {
        self.powf(p)
    }
    #[inline]
    fn gt_r(self, rhs: Self) -> bool {
        self > rhs
    }
    #[inline]
    fn lt_r(self, rhs: Self) -> bool {
        self < rhs
    }
    #[inline]
    fn from_f64_r(v: f64) -> Self {
        v
    }
}

/// Element scalar supported by `norm` / `vector_norm` / `matrix_norm` (`f16`, `f32`, `f64`, `Complex32`, `Complex64`).
pub trait NormScalar: Scalar + PartialEq + Copy + Send + Sync + 'static {
    type NormReal: NormRealScalar;
    const ZERO_N: Self;
    fn is_nan_n(self) -> bool;
    fn is_zero_n(self) -> bool;
    fn abs_n(self) -> Self::NormReal;
    fn supports_svd_n() -> bool;
    fn svdvals_for_norm(moved: &NdArray<Self>) -> Result<NdArray<Self::NormReal>>;
}

impl NormScalar for f16 {
    type NormReal = f16;
    const ZERO_N: Self = f16::from_bits(0);
    #[inline]
    fn is_nan_n(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn is_zero_n(self) -> bool {
        self.to_f32() == 0.0
    }
    #[inline]
    fn abs_n(self) -> f16 {
        f16::from_f32(self.to_f32().abs())
    }
    #[inline]
    fn supports_svd_n() -> bool {
        false
    }
    fn svdvals_for_norm(_moved: &NdArray<Self>) -> Result<NdArray<f16>> {
        Err(NumPyError::DTypeError(
            "array type float16 is unsupported in linalg".into(),
        ))
    }
}

macro_rules! impl_norm_scalar_linalg {
    ($($t:ty => $r:ty),+) => {$(
        impl NormScalar for $t {
            type NormReal = $r;
            const ZERO_N: Self = <$t as ProductScalar>::ZERO;
            #[inline]
            fn is_nan_n(self) -> bool {
                self.is_nan_elem()
            }
            #[inline]
            fn is_zero_n(self) -> bool {
                self == <$t as ProductScalar>::ZERO
            }
            #[inline]
            fn abs_n(self) -> $r {
                self.abs_real()
            }
            #[inline]
            fn supports_svd_n() -> bool {
                true
            }
            fn svdvals_for_norm(moved: &NdArray<Self>) -> Result<NdArray<$r>> {
                crate::decomp::svdvals(moved)
            }
        }
    )+};
}

impl_norm_scalar_linalg!(
    f32 => f32,
    f64 => f64,
    Complex32 => f32,
    Complex64 => f64
);

/// Linear algebra scalar supported by decompositions and solvers (`f32`, `f64`, `Complex32`, `Complex64`).
pub trait LinalgScalar:
    Scalar + ProductScalar + NormScalar<NormReal = <Self as LinalgScalar>::Real> + Copy + Send + Sync + 'static
{
    type Real: LinalgReal;
    type Complex: Scalar + ProductScalar + Copy + Send + Sync + 'static;

    fn is_complex_type() -> bool;
    fn from_real_val(r: Self::Real) -> Self;
    fn real_part(self) -> Self::Real;
    fn imag_part(self) -> Self::Real;
    fn abs_real(self) -> Self::Real;
    fn conj_val(self) -> Self;
    fn negate_val(self) -> Self;
    fn mul_val(self, rhs: Self) -> Self;
    fn div_val(self, rhs: Self) -> Self;
    fn div_real(self, r: Self::Real) -> Self;
    fn is_finite_elem(self) -> bool;
    fn is_nan_elem(self) -> bool;
    fn nan_scalar() -> Self;
    fn to_complex(self) -> Self::Complex;
    fn into_dyn(arr: NdArray<Self>) -> DynArray;
    fn real_into_dyn(arr: NdArray<Self::Real>) -> DynArray;
    fn complex_into_dyn(arr: NdArray<Self::Complex>) -> DynArray;

    // Low-level concrete `faer` 0.21 kernel wrappers:
    fn faer_cholesky_scratch(dim: usize, par: Par) -> StackReq;
    fn faer_cholesky_in_place(a: MatMut<'_, Self>, par: Par, stack: &mut MemStack) -> Result<()>;

    fn faer_qr_bs(m: usize, n: usize) -> usize;
    fn faer_qr_scratch(m: usize, n: usize, bs: usize, par: Par) -> StackReq;
    fn faer_qr_in_place(a: MatMut<'_, Self>, h: MatMut<'_, Self>, par: Par, stack: &mut MemStack);
    fn faer_householder_left_scratch(m: usize, bs: usize, cols: usize) -> StackReq;
    fn faer_apply_householder_left(
        basis: MatRef<'_, Self>,
        coeff: MatRef<'_, Self>,
        q: MatMut<'_, Self>,
        par: Par,
        stack: &mut MemStack,
    );

    fn faer_svd_scratch(
        m: usize,
        n: usize,
        comp_u: ComputeSvdVectors,
        comp_v: ComputeSvdVectors,
        par: Par,
    ) -> StackReq;
    fn faer_svd(
        a: MatRef<'_, Self>,
        s: DiagMut<'_, Self>,
        u: Option<MatMut<'_, Self>>,
        v: Option<MatMut<'_, Self>>,
        par: Par,
        stack: &mut MemStack,
    ) -> Result<()>;

    fn faer_self_adjoint_evd_scratch(dim: usize, comp: ComputeEigenvectors, par: Par) -> StackReq;
    fn faer_self_adjoint_evd(
        a: MatRef<'_, Self>,
        s: DiagMut<'_, Self>,
        u: Option<MatMut<'_, Self>>,
        par: Par,
        stack: &mut MemStack,
    ) -> Result<()>;

    fn faer_lu_scratch(m: usize, n: usize, par: Par) -> StackReq;
    fn faer_lu_in_place<'a>(
        a: MatMut<'_, Self>,
        perm_fwd: &'a mut [usize],
        perm_bwd: &'a mut [usize],
        par: Par,
        stack: &mut MemStack,
    ) -> (usize, PermRef<'a, usize>);
    fn faer_solve_scratch(dim: usize, rhs_ncols: usize, par: Par) -> StackReq;
    fn faer_solve_in_place(
        lu: MatRef<'_, Self>,
        row_perm: PermRef<'_, usize>,
        rhs: MatMut<'_, Self>,
        par: Par,
        stack: &mut MemStack,
    );
    fn faer_inverse_scratch(dim: usize, par: Par) -> StackReq;
    fn faer_inverse(
        out: MatMut<'_, Self>,
        lu: MatRef<'_, Self>,
        row_perm: PermRef<'_, usize>,
        par: Par,
        stack: &mut MemStack,
    );

    fn evd_scratch_req(dim: usize, compute_vecs: bool, par: Par) -> StackReq;
    fn evd_compute(
        a: MatRef<'_, Self>,
        s_out: &mut [Self::Complex],
        u_out: Option<&mut [Self::Complex]>,
        par: Par,
        stack: &mut MemStack,
    ) -> Result<()>;
}

#[inline]
fn normalize_complex_col<R: LinalgReal>(u: &mut [Complex<R>], n: usize, col: usize) {
    let mut sum_sq = 0.0f64;
    for r in 0..n {
        let z = u[r * n + col];
        let re = z.re.to_f64_lossy();
        let im = z.im.to_f64_lossy();
        sum_sq += re * re + im * im;
    }
    let norm = sum_sq.sqrt();
    if norm > 0.0 && norm.is_finite() {
        let inv = R::from_f64_lossy(1.0 / norm);
        for r in 0..n {
            let z = &mut u[r * n + col];
            z.re = z.re.mul_elem(inv);
            z.im = z.im.mul_elem(inv);
        }
    }
}

macro_rules! impl_faer_common_ops {
    ($t:ty) => {
        fn faer_cholesky_scratch(dim: usize, par: Par) -> StackReq {
            llt::factor::cholesky_in_place_scratch::<$t>(dim, par, Default::default())
        }
        fn faer_cholesky_in_place(
            a: MatMut<'_, Self>,
            par: Par,
            stack: &mut MemStack,
        ) -> Result<()> {
            llt::factor::cholesky_in_place(a, Default::default(), par, stack, Default::default())
                .map_err(|_| NumPyError::LinAlgError("Matrix is not positive definite".into()))?;
            Ok(())
        }
        fn faer_qr_bs(m: usize, n: usize) -> usize {
            qr_factor::recommended_blocksize::<$t>(m, n)
        }
        fn faer_qr_scratch(m: usize, n: usize, bs: usize, par: Par) -> StackReq {
            qr_factor::qr_in_place_scratch::<$t>(m, n, bs, par, Default::default())
        }
        fn faer_qr_in_place(
            a: MatMut<'_, Self>,
            h: MatMut<'_, Self>,
            par: Par,
            stack: &mut MemStack,
        ) {
            qr_factor::qr_in_place(a, h, par, stack, Default::default());
        }
        fn faer_householder_left_scratch(m: usize, bs: usize, cols: usize) -> StackReq {
            householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<$t>(
                m, bs, cols,
            )
        }
        fn faer_apply_householder_left(
            basis: MatRef<'_, Self>,
            coeff: MatRef<'_, Self>,
            q: MatMut<'_, Self>,
            par: Par,
            stack: &mut MemStack,
        ) {
            householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                basis,
                coeff,
                Conj::No,
                q,
                par,
                stack,
            );
        }
        fn faer_svd_scratch(
            m: usize,
            n: usize,
            comp_u: ComputeSvdVectors,
            comp_v: ComputeSvdVectors,
            par: Par,
        ) -> StackReq {
            svd::svd_scratch::<$t>(m, n, comp_u, comp_v, par, Default::default())
        }
        fn faer_svd(
            a: MatRef<'_, Self>,
            s: DiagMut<'_, Self>,
            u: Option<MatMut<'_, Self>>,
            v: Option<MatMut<'_, Self>>,
            par: Par,
            stack: &mut MemStack,
        ) -> Result<()> {
            svd::svd(a, s, u, v, par, stack, Default::default())
                .map_err(|_| NumPyError::LinAlgError("SVD did not converge".into()))
        }
        fn faer_self_adjoint_evd_scratch(
            dim: usize,
            comp: ComputeEigenvectors,
            par: Par,
        ) -> StackReq {
            evd::self_adjoint_evd_scratch::<$t>(dim, comp, par, Default::default())
        }
        fn faer_self_adjoint_evd(
            a: MatRef<'_, Self>,
            s: DiagMut<'_, Self>,
            u: Option<MatMut<'_, Self>>,
            par: Par,
            stack: &mut MemStack,
        ) -> Result<()> {
            evd::self_adjoint_evd(a, s, u, par, stack, Default::default())
                .map_err(|_| NumPyError::LinAlgError("Eigenvalues did not converge".into()))
        }
        fn faer_lu_scratch(m: usize, n: usize, par: Par) -> StackReq {
            lu_factor::lu_in_place_scratch::<usize, $t>(m, n, par, Default::default())
        }
        fn faer_lu_in_place<'a>(
            a: MatMut<'_, Self>,
            perm_fwd: &'a mut [usize],
            perm_bwd: &'a mut [usize],
            par: Par,
            stack: &mut MemStack,
        ) -> (usize, PermRef<'a, usize>) {
            let (info, perm) =
                lu_factor::lu_in_place(a, perm_fwd, perm_bwd, par, stack, Default::default());
            (info.transposition_count, perm)
        }
        fn faer_solve_scratch(dim: usize, rhs_ncols: usize, par: Par) -> StackReq {
            lu_solve::solve_in_place_scratch::<usize, $t>(dim, rhs_ncols, par)
        }
        fn faer_solve_in_place(
            lu: MatRef<'_, Self>,
            row_perm: PermRef<'_, usize>,
            rhs: MatMut<'_, Self>,
            par: Par,
            stack: &mut MemStack,
        ) {
            lu_solve::solve_in_place(lu, lu, row_perm, rhs, par, stack);
        }
        fn faer_inverse_scratch(dim: usize, par: Par) -> StackReq {
            lu_inv::inverse_scratch::<usize, $t>(dim, par)
        }
        fn faer_inverse(
            out: MatMut<'_, Self>,
            lu: MatRef<'_, Self>,
            row_perm: PermRef<'_, usize>,
            par: Par,
            stack: &mut MemStack,
        ) {
            lu_inv::inverse(out, lu, lu, row_perm, par, stack);
        }
    };
}

macro_rules! impl_evd_real_body {
    ($r:ty) => {
        fn evd_scratch_req(dim: usize, compute_vecs: bool, par: Par) -> StackReq {
            evd::evd_scratch::<$r>(
                dim,
                ComputeEigenvectors::No,
                if compute_vecs {
                    ComputeEigenvectors::Yes
                } else {
                    ComputeEigenvectors::No
                },
                par,
                Default::default(),
            )
        }
        fn evd_compute(
            a: MatRef<'_, Self>,
            s_out: &mut [Complex<$r>],
            mut u_out: Option<&mut [Complex<$r>]>,
            par: Par,
            stack: &mut MemStack,
        ) -> Result<()> {
            let n = a.nrows();
            if n == 0 {
                return Ok(());
            }
            let mut s_re = Diag::<$r>::full(n, 0.0);
            let mut s_im = Diag::<$r>::full(n, 0.0);
            let mut ur = u_out.as_ref().map(|_| Mat::<$r>::full(n, n, 0.0));
            evd::evd_real(
                a,
                s_re.as_mut(),
                s_im.as_mut(),
                None,
                ur.as_mut().map(|m| m.as_mut()),
                par,
                stack,
                Default::default(),
            )
            .map_err(|_| NumPyError::LinAlgError("Eigenvalues did not converge".into()))?;

            let s_re_col = s_re.column_vector();
            let s_im_col = s_im.column_vector();
            if let Some(u_buf) = u_out.as_deref_mut() {
                let ur = ur.as_ref().expect("allocated ur");
                let mut i = 0;
                while i < n {
                    if s_im_col[i] == 0.0 {
                        s_out[i] = Complex::new(s_re_col[i], 0.0);
                        for r in 0..n {
                            u_buf[r * n + i] = Complex::new(ur[(r, i)], 0.0);
                        }
                        normalize_complex_col(u_buf, n, i);
                        i += 1;
                    } else if i + 1 < n {
                        s_out[i] = Complex::new(s_re_col[i], s_im_col[i]);
                        s_out[i + 1] = Complex::new(s_re_col[i + 1], s_im_col[i + 1]);
                        for r in 0..n {
                            let re = ur[(r, i)];
                            let im = ur[(r, i + 1)];
                            u_buf[r * n + i] = Complex::new(re, im);
                            u_buf[r * n + (i + 1)] = Complex::new(re, -im);
                        }
                        normalize_complex_col(u_buf, n, i);
                        normalize_complex_col(u_buf, n, i + 1);
                        i += 2;
                    } else {
                        s_out[i] = Complex::new(s_re_col[i], s_im_col[i]);
                        for r in 0..n {
                            u_buf[r * n + i] = Complex::new(ur[(r, i)], 0.0);
                        }
                        normalize_complex_col(u_buf, n, i);
                        i += 1;
                    }
                }
            } else {
                for i in 0..n {
                    s_out[i] = Complex::new(s_re_col[i], s_im_col[i]);
                }
            }
            Ok(())
        }
    };
}

macro_rules! impl_evd_cplx_body {
    ($c:ty, $r:ty) => {
        fn evd_scratch_req(dim: usize, compute_vecs: bool, par: Par) -> StackReq {
            evd::evd_scratch::<$c>(
                dim,
                ComputeEigenvectors::No,
                if compute_vecs {
                    ComputeEigenvectors::Yes
                } else {
                    ComputeEigenvectors::No
                },
                par,
                Default::default(),
            )
        }
        fn evd_compute(
            a: MatRef<'_, Self>,
            s_out: &mut [Complex<$r>],
            mut u_out: Option<&mut [Complex<$r>]>,
            par: Par,
            stack: &mut MemStack,
        ) -> Result<()> {
            let n = a.nrows();
            if n == 0 {
                return Ok(());
            }
            let mut s = Diag::<$c>::full(n, <$c as ProductScalar>::ZERO);
            let mut ur = u_out
                .as_ref()
                .map(|_| Mat::<$c>::full(n, n, <$c as ProductScalar>::ZERO));
            evd::evd_cplx(
                a,
                s.as_mut(),
                None,
                ur.as_mut().map(|m| m.as_mut()),
                par,
                stack,
                Default::default(),
            )
            .map_err(|_| NumPyError::LinAlgError("Eigenvalues did not converge".into()))?;

            let s_col = s.column_vector();
            for i in 0..n {
                s_out[i] = s_col[i];
            }
            if let Some(u_buf) = u_out.as_deref_mut() {
                let ur = ur.as_ref().expect("allocated ur");
                for i in 0..n {
                    for r in 0..n {
                        u_buf[r * n + i] = ur[(r, i)];
                    }
                    normalize_complex_col(u_buf, n, i);
                }
            }
            Ok(())
        }
    };
}

impl LinalgScalar for f32 {
    type Real = f32;
    type Complex = Complex32;
    #[inline]
    fn is_complex_type() -> bool {
        false
    }
    #[inline]
    fn from_real_val(r: f32) -> Self {
        r
    }
    #[inline]
    fn real_part(self) -> f32 {
        self
    }
    #[inline]
    fn imag_part(self) -> f32 {
        0.0
    }
    #[inline]
    fn abs_real(self) -> f32 {
        self.abs()
    }
    #[inline]
    fn conj_val(self) -> Self {
        self
    }
    #[inline]
    fn negate_val(self) -> Self {
        -self
    }
    #[inline]
    fn mul_val(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn div_val(self, rhs: Self) -> Self {
        self / rhs
    }
    #[inline]
    fn div_real(self, r: f32) -> Self {
        self / r
    }
    #[inline]
    fn is_finite_elem(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn is_nan_elem(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn nan_scalar() -> Self {
        f32::NAN
    }
    #[inline]
    fn to_complex(self) -> Complex32 {
        Complex32::new(self, 0.0)
    }
    #[inline]
    fn into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::F32(arr)
    }
    #[inline]
    fn real_into_dyn(arr: NdArray<f32>) -> DynArray {
        DynArray::F32(arr)
    }
    #[inline]
    fn complex_into_dyn(arr: NdArray<Complex32>) -> DynArray {
        DynArray::C64(arr)
    }
    impl_faer_common_ops!(f32);
    impl_evd_real_body!(f32);
}

impl LinalgScalar for f64 {
    type Real = f64;
    type Complex = Complex64;
    #[inline]
    fn is_complex_type() -> bool {
        false
    }
    #[inline]
    fn from_real_val(r: f64) -> Self {
        r
    }
    #[inline]
    fn real_part(self) -> f64 {
        self
    }
    #[inline]
    fn imag_part(self) -> f64 {
        0.0
    }
    #[inline]
    fn abs_real(self) -> f64 {
        self.abs()
    }
    #[inline]
    fn conj_val(self) -> Self {
        self
    }
    #[inline]
    fn negate_val(self) -> Self {
        -self
    }
    #[inline]
    fn mul_val(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn div_val(self, rhs: Self) -> Self {
        self / rhs
    }
    #[inline]
    fn div_real(self, r: f64) -> Self {
        self / r
    }
    #[inline]
    fn is_finite_elem(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn is_nan_elem(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn nan_scalar() -> Self {
        f64::NAN
    }
    #[inline]
    fn to_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    #[inline]
    fn into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::F64(arr)
    }
    #[inline]
    fn real_into_dyn(arr: NdArray<f64>) -> DynArray {
        DynArray::F64(arr)
    }
    #[inline]
    fn complex_into_dyn(arr: NdArray<Complex64>) -> DynArray {
        DynArray::C128(arr)
    }
    impl_faer_common_ops!(f64);
    impl_evd_real_body!(f64);
}

impl LinalgScalar for Complex32 {
    type Real = f32;
    type Complex = Complex32;
    #[inline]
    fn is_complex_type() -> bool {
        true
    }
    #[inline]
    fn from_real_val(r: f32) -> Self {
        Complex32::new(r, 0.0)
    }
    #[inline]
    fn real_part(self) -> f32 {
        self.re
    }
    #[inline]
    fn imag_part(self) -> f32 {
        self.im
    }
    #[inline]
    fn abs_real(self) -> f32 {
        self.norm()
    }
    #[inline]
    fn conj_val(self) -> Self {
        self.conj()
    }
    #[inline]
    fn negate_val(self) -> Self {
        -self
    }
    #[inline]
    fn mul_val(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn div_val(self, rhs: Self) -> Self {
        self / rhs
    }
    #[inline]
    fn div_real(self, r: f32) -> Self {
        Complex32::new(self.re / r, self.im / r)
    }
    #[inline]
    fn is_finite_elem(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline]
    fn is_nan_elem(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    #[inline]
    fn nan_scalar() -> Self {
        Complex32::new(f32::NAN, f32::NAN)
    }
    #[inline]
    fn to_complex(self) -> Complex32 {
        self
    }
    #[inline]
    fn into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::C64(arr)
    }
    #[inline]
    fn real_into_dyn(arr: NdArray<f32>) -> DynArray {
        DynArray::F32(arr)
    }
    #[inline]
    fn complex_into_dyn(arr: NdArray<Complex32>) -> DynArray {
        DynArray::C64(arr)
    }
    impl_faer_common_ops!(Complex32);
    impl_evd_cplx_body!(Complex32, f32);
}

impl LinalgScalar for Complex64 {
    type Real = f64;
    type Complex = Complex64;
    #[inline]
    fn is_complex_type() -> bool {
        true
    }
    #[inline]
    fn from_real_val(r: f64) -> Self {
        Complex64::new(r, 0.0)
    }
    #[inline]
    fn real_part(self) -> f64 {
        self.re
    }
    #[inline]
    fn imag_part(self) -> f64 {
        self.im
    }
    #[inline]
    fn abs_real(self) -> f64 {
        self.norm()
    }
    #[inline]
    fn conj_val(self) -> Self {
        self.conj()
    }
    #[inline]
    fn negate_val(self) -> Self {
        -self
    }
    #[inline]
    fn mul_val(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn div_val(self, rhs: Self) -> Self {
        self / rhs
    }
    #[inline]
    fn div_real(self, r: f64) -> Self {
        Complex64::new(self.re / r, self.im / r)
    }
    #[inline]
    fn is_finite_elem(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline]
    fn is_nan_elem(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    #[inline]
    fn nan_scalar() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
    #[inline]
    fn to_complex(self) -> Complex64 {
        self
    }
    #[inline]
    fn into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::C128(arr)
    }
    #[inline]
    fn real_into_dyn(arr: NdArray<f64>) -> DynArray {
        DynArray::F64(arr)
    }
    #[inline]
    fn complex_into_dyn(arr: NdArray<Complex64>) -> DynArray {
        DynArray::C128(arr)
    }
    impl_faer_common_ops!(Complex64);
    impl_evd_cplx_body!(Complex64, f64);
}

/// Scalar trait for product and tensor operations across all 14 NumPy dtypes.
/// Uses `matrixmultiply` (for `f32`/`f64`) and `faer::linalg::matmul` (for `Complex32`/`Complex64`),
/// with wrapping arithmetic / boolean algebra for integers and `bool`, and `f16` support.
pub trait ProductScalar: Scalar + Copy + Send + Sync + 'static {
    const ZERO: Self;
    const ONE: Self;

    #[inline(always)]
    fn conj_elem(self) -> Self {
        self
    }
    fn add_elem(self, rhs: Self) -> Self;
    fn sub_elem(self, rhs: Self) -> Self;
    fn mul_elem(self, rhs: Self) -> Self;

    /// General matrix multiplication C[m, n] = A[m, k] * B[k, n] with arbitrary element strides.
    ///
    /// # Safety
    /// Pointers must be valid for all elements implied by `(m, k, n)` and row/col strides,
    /// and `c_ptr` must not alias `a_ptr` or `b_ptr`.
    unsafe fn strided_gemm(
        m: usize,
        k: usize,
        n: usize,
        a_ptr: *const Self,
        rsa: isize,
        csa: isize,
        b_ptr: *const Self,
        rsb: isize,
        csb: isize,
        c_ptr: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = Self::ZERO;
                for p in 0..k {
                    let av = unsafe { *a_ptr.offset(i as isize * rsa + p as isize * csa) };
                    let bv = unsafe { *b_ptr.offset(p as isize * rsb + j as isize * csb) };
                    acc = acc.add_elem(av.mul_elem(bv));
                }
                unsafe {
                    *c_ptr.offset(i as isize * rsc + j as isize * csc) = acc;
                }
            }
        }
    }

    /// General matrix multiplication C[m, n] = conj(A[m, k]) * B[k, n] with arbitrary element strides.
    ///
    /// # Safety
    /// Same safety invariants as `strided_gemm`.
    unsafe fn strided_gemm_conj_lhs(
        m: usize,
        k: usize,
        n: usize,
        a_ptr: *const Self,
        rsa: isize,
        csa: isize,
        b_ptr: *const Self,
        rsb: isize,
        csb: isize,
        c_ptr: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        unsafe {
            Self::strided_gemm(m, k, n, a_ptr, rsa, csa, b_ptr, rsb, csb, c_ptr, rsc, csc);
        }
    }

    fn product_into_dyn(arr: NdArray<Self>) -> DynArray;

    /// Invert stacked square matrices for `matrix_power(..., n < 0)`.
    /// Only supported natively on `f32`, `f64`, `Complex32`, `Complex64`.
    fn invert_stacked(_a: &NdArray<Self>) -> Result<NdArray<Self>> {
        Err(NumPyError::DTypeError(
            "Negative matrix_power requires floating or complex dtype; use matrix_power_dyn for auto-promotion".into(),
        ))
    }
}

impl ProductScalar for bool {
    const ZERO: Self = false;
    const ONE: Self = true;
    #[inline(always)]
    fn add_elem(self, rhs: Self) -> Self {
        self || rhs
    }
    #[inline(always)]
    fn sub_elem(self, rhs: Self) -> Self {
        self ^ rhs
    }
    #[inline(always)]
    fn mul_elem(self, rhs: Self) -> Self {
        self && rhs
    }
    #[inline]
    fn product_into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::Bool(arr)
    }
}

macro_rules! impl_int_product_scalar {
    ($t:ty, $variant:ident) => {
        impl ProductScalar for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            #[inline(always)]
            fn add_elem(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }
            #[inline(always)]
            fn sub_elem(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }
            #[inline(always)]
            fn mul_elem(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }
            #[inline]
            fn product_into_dyn(arr: NdArray<Self>) -> DynArray {
                DynArray::$variant(arr)
            }
        }
    };
}

impl_int_product_scalar!(i8, I8);
impl_int_product_scalar!(i16, I16);
impl_int_product_scalar!(i32, I32);
impl_int_product_scalar!(i64, I64);
impl_int_product_scalar!(u8, U8);
impl_int_product_scalar!(u16, U16);
impl_int_product_scalar!(u32, U32);
impl_int_product_scalar!(u64, U64);

impl ProductScalar for f16 {
    const ZERO: Self = f16::from_bits(0);
    const ONE: Self = f16::from_bits(0x3C00);
    #[inline(always)]
    fn add_elem(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32() + rhs.to_f32())
    }
    #[inline(always)]
    fn sub_elem(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32() - rhs.to_f32())
    }
    #[inline(always)]
    fn mul_elem(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32() * rhs.to_f32())
    }
    #[inline]
    fn product_into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::F16(arr)
    }
}

impl ProductScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn add_elem(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline(always)]
    fn sub_elem(self, rhs: Self) -> Self {
        self - rhs
    }
    #[inline(always)]
    fn mul_elem(self, rhs: Self) -> Self {
        self * rhs
    }
    unsafe fn strided_gemm(
        m: usize,
        k: usize,
        n: usize,
        a_ptr: *const Self,
        rsa: isize,
        csa: isize,
        b_ptr: *const Self,
        rsb: isize,
        csb: isize,
        c_ptr: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        unsafe {
            matrixmultiply::sgemm(
                m, k, n, 1.0, a_ptr, rsa, csa, b_ptr, rsb, csb, 0.0, c_ptr, rsc, csc,
            );
        }
    }
    #[inline]
    fn product_into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::F32(arr)
    }
    fn invert_stacked(a: &NdArray<Self>) -> Result<NdArray<Self>> {
        crate::solve::inv(a)
    }
}

impl ProductScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn add_elem(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline(always)]
    fn sub_elem(self, rhs: Self) -> Self {
        self - rhs
    }
    #[inline(always)]
    fn mul_elem(self, rhs: Self) -> Self {
        self * rhs
    }
    unsafe fn strided_gemm(
        m: usize,
        k: usize,
        n: usize,
        a_ptr: *const Self,
        rsa: isize,
        csa: isize,
        b_ptr: *const Self,
        rsb: isize,
        csb: isize,
        c_ptr: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        unsafe {
            matrixmultiply::dgemm(
                m, k, n, 1.0, a_ptr, rsa, csa, b_ptr, rsb, csb, 0.0, c_ptr, rsc, csc,
            );
        }
    }
    #[inline]
    fn product_into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::F64(arr)
    }
    fn invert_stacked(a: &NdArray<Self>) -> Result<NdArray<Self>> {
        crate::solve::inv(a)
    }
}

impl ProductScalar for Complex32 {
    const ZERO: Self = Complex32::new(0.0, 0.0);
    const ONE: Self = Complex32::new(1.0, 0.0);
    #[inline(always)]
    fn conj_elem(self) -> Self {
        self.conj()
    }
    #[inline(always)]
    fn add_elem(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline(always)]
    fn sub_elem(self, rhs: Self) -> Self {
        self - rhs
    }
    #[inline(always)]
    fn mul_elem(self, rhs: Self) -> Self {
        self * rhs
    }
    unsafe fn strided_gemm(
        m: usize,
        k: usize,
        n: usize,
        a_ptr: *const Self,
        rsa: isize,
        csa: isize,
        b_ptr: *const Self,
        rsb: isize,
        csb: isize,
        c_ptr: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        if k == 0 {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *c_ptr.offset(i as isize * rsc + j as isize * csc) = Complex32::ZERO;
                    }
                }
            }
            return;
        }
        unsafe {
            let a_ref = MatRef::from_raw_parts(a_ptr, m, k, rsa, csa);
            let b_ref = MatRef::from_raw_parts(b_ptr, k, n, rsb, csb);
            let c_mut = MatMut::from_raw_parts_mut(c_ptr, m, n, rsc, csc);
            faer::linalg::matmul::matmul(
                c_mut,
                faer::Accum::Replace,
                a_ref,
                b_ref,
                Complex32::ONE,
                Par::Seq,
            );
        }
    }
    unsafe fn strided_gemm_conj_lhs(
        m: usize,
        k: usize,
        n: usize,
        a_ptr: *const Self,
        rsa: isize,
        csa: isize,
        b_ptr: *const Self,
        rsb: isize,
        csb: isize,
        c_ptr: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        if k == 0 {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *c_ptr.offset(i as isize * rsc + j as isize * csc) = Complex32::ZERO;
                    }
                }
            }
            return;
        }
        unsafe {
            let a_ref = MatRef::from_raw_parts(a_ptr, m, k, rsa, csa);
            let b_ref = MatRef::from_raw_parts(b_ptr, k, n, rsb, csb);
            let c_mut = MatMut::from_raw_parts_mut(c_ptr, m, n, rsc, csc);
            faer::linalg::matmul::matmul_with_conj(
                c_mut,
                faer::Accum::Replace,
                a_ref,
                faer::Conj::Yes,
                b_ref,
                faer::Conj::No,
                Complex32::ONE,
                Par::Seq,
            );
        }
    }
    #[inline]
    fn product_into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::C64(arr)
    }
    fn invert_stacked(a: &NdArray<Self>) -> Result<NdArray<Self>> {
        crate::solve::inv(a)
    }
}

impl ProductScalar for Complex64 {
    const ZERO: Self = Complex64::new(0.0, 0.0);
    const ONE: Self = Complex64::new(1.0, 0.0);
    #[inline(always)]
    fn conj_elem(self) -> Self {
        self.conj()
    }
    #[inline(always)]
    fn add_elem(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline(always)]
    fn sub_elem(self, rhs: Self) -> Self {
        self - rhs
    }
    #[inline(always)]
    fn mul_elem(self, rhs: Self) -> Self {
        self * rhs
    }
    unsafe fn strided_gemm(
        m: usize,
        k: usize,
        n: usize,
        a_ptr: *const Self,
        rsa: isize,
        csa: isize,
        b_ptr: *const Self,
        rsb: isize,
        csb: isize,
        c_ptr: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        if k == 0 {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *c_ptr.offset(i as isize * rsc + j as isize * csc) = Complex64::ZERO;
                    }
                }
            }
            return;
        }
        unsafe {
            let a_ref = MatRef::from_raw_parts(a_ptr, m, k, rsa, csa);
            let b_ref = MatRef::from_raw_parts(b_ptr, k, n, rsb, csb);
            let c_mut = MatMut::from_raw_parts_mut(c_ptr, m, n, rsc, csc);
            faer::linalg::matmul::matmul(
                c_mut,
                faer::Accum::Replace,
                a_ref,
                b_ref,
                Complex64::ONE,
                Par::Seq,
            );
        }
    }
    unsafe fn strided_gemm_conj_lhs(
        m: usize,
        k: usize,
        n: usize,
        a_ptr: *const Self,
        rsa: isize,
        csa: isize,
        b_ptr: *const Self,
        rsb: isize,
        csb: isize,
        c_ptr: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        if k == 0 {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *c_ptr.offset(i as isize * rsc + j as isize * csc) = Complex64::ZERO;
                    }
                }
            }
            return;
        }
        unsafe {
            let a_ref = MatRef::from_raw_parts(a_ptr, m, k, rsa, csa);
            let b_ref = MatRef::from_raw_parts(b_ptr, k, n, rsb, csb);
            let c_mut = MatMut::from_raw_parts_mut(c_ptr, m, n, rsc, csc);
            faer::linalg::matmul::matmul_with_conj(
                c_mut,
                faer::Accum::Replace,
                a_ref,
                faer::Conj::Yes,
                b_ref,
                faer::Conj::No,
                Complex64::ONE,
                Par::Seq,
            );
        }
    }
    #[inline]
    fn product_into_dyn(arr: NdArray<Self>) -> DynArray {
        DynArray::C128(arr)
    }
    fn invert_stacked(a: &NdArray<Self>) -> Result<NdArray<Self>> {
        crate::solve::inv(a)
    }
}

/// Promote a dtype for linear algebra decompositions, solvers, and norms:
/// integers, `bool`, and `f16` promote to `f64`; `f32`, `f64`, `C64`, `C128` stay unchanged.
#[inline]
pub fn linalg_common_dtype(dt: DType) -> DType {
    match dt {
        DType::F32 | DType::F64 | DType::C64 | DType::C128 => dt,
        _ => DType::F64,
    }
}

/// Strict NumPy promotion for decompositions, solvers, eigenvalues, `cond`, and `matrix_rank`:
/// rejects `f16` with `DTypeError` (`FIX(F-B2-01)`), promoting other ints/bool to `f64`.
#[inline]
pub fn linalg_common_dtype_strict(dt: DType) -> Result<DType> {
    if dt == DType::F16 {
        Err(NumPyError::DTypeError(
            "array type float16 is unsupported in linalg".into(),
        ))
    } else {
        Ok(linalg_common_dtype(dt))
    }
}

/// Promote a pair of dtypes for linear algebra solvers (`solve`, `tensorsolve`).
#[inline]
pub fn linalg_common_dtype_pair(a: DType, b: DType) -> DType {
    linalg_common_dtype(promote_types(linalg_common_dtype(a), linalg_common_dtype(b)))
}

/// Strict NumPy promotion for binary solvers, rejecting `f16` with `DTypeError` (`FIX(F-B2-01)`).
#[inline]
pub fn linalg_common_dtype_pair_strict(a: DType, b: DType) -> Result<DType> {
    if a == DType::F16 || b == DType::F16 {
        Err(NumPyError::DTypeError(
            "array type float16 is unsupported in linalg".into(),
        ))
    } else {
        Ok(linalg_common_dtype_pair(a, b))
    }
}

/// Promote a dtype for `trace` matching NumPy (`bool`/small signed ints -> `I64`, small unsigned -> `U64`).
#[inline]
pub fn trace_default_dtype(dt: DType) -> DType {
    match dt {
        DType::Bool | DType::I8 | DType::I16 | DType::I32 | DType::I64 => DType::I64,
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => DType::U64,
        DType::F16 | DType::F32 | DType::F64 | DType::C64 | DType::C128 => dt,
    }
}

/// Helper macro dispatching a single `DynArray` promoted to one of the four `LinalgScalar` types (`F32`, `F64`, `C64`, `C128`),
/// rejecting `F16` per NumPy (`FIX(F-B2-01)`).
#[macro_export]
macro_rules! dispatch_linalg_dyn {
    ($arr:expr, $var:ident => $body:expr) => {{
        let target = $crate::traits::linalg_common_dtype_strict($arr.dtype())?;
        let promoted = $arr.astype(target, numpy_types::Casting::Unsafe)?;
        match &promoted {
            numpy_types::DynArray::F32($var) => $body,
            numpy_types::DynArray::F64($var) => $body,
            numpy_types::DynArray::C64($var) => $body,
            numpy_types::DynArray::C128($var) => $body,
            _ => unreachable!("linalg_common_dtype_strict only yields F32, F64, C64, C128"),
        }
    }};
}

/// Helper macro dispatching a pair of `DynArray`s promoted to a common `LinalgScalar` dtype,
/// rejecting `F16` per NumPy (`FIX(F-B2-01)`).
#[macro_export]
macro_rules! dispatch_linalg_pair_dyn {
    ($a:expr, $b:expr, ($va:ident, $vb:ident) => $body:expr) => {{
        let target = $crate::traits::linalg_common_dtype_pair_strict($a.dtype(), $b.dtype())?;
        let pa = $a.astype(target, numpy_types::Casting::Unsafe)?;
        let pb = $b.astype(target, numpy_types::Casting::Unsafe)?;
        match (&pa, &pb) {
            (numpy_types::DynArray::F32($va), numpy_types::DynArray::F32($vb)) => $body,
            (numpy_types::DynArray::F64($va), numpy_types::DynArray::F64($vb)) => $body,
            (numpy_types::DynArray::C64($va), numpy_types::DynArray::C64($vb)) => $body,
            (numpy_types::DynArray::C128($va), numpy_types::DynArray::C128($vb)) => $body,
            _ => unreachable!("common dtype pair guarantees matching variants"),
        }
    }};
}

/// Helper macro dispatching all 14 `DynArray` variants for product operations.
#[macro_export]
macro_rules! dispatch_product_dyn {
    ($arr:expr, $var:ident => $body:expr) => {
        match $arr {
            numpy_types::DynArray::Bool($var) => $body,
            numpy_types::DynArray::I8($var) => $body,
            numpy_types::DynArray::I16($var) => $body,
            numpy_types::DynArray::I32($var) => $body,
            numpy_types::DynArray::I64($var) => $body,
            numpy_types::DynArray::U8($var) => $body,
            numpy_types::DynArray::U16($var) => $body,
            numpy_types::DynArray::U32($var) => $body,
            numpy_types::DynArray::U64($var) => $body,
            numpy_types::DynArray::F16($var) => $body,
            numpy_types::DynArray::F32($var) => $body,
            numpy_types::DynArray::F64($var) => $body,
            numpy_types::DynArray::C64($var) => $body,
            numpy_types::DynArray::C128($var) => $body,
        }
    };
}

/// Helper macro dispatching a pair of `DynArray`s promoted via `promote_types`.
#[macro_export]
macro_rules! dispatch_product_pair_dyn {
    ($a:expr, $b:expr, ($va:ident, $vb:ident) => $body:expr) => {{
        let target = numpy_types::promote_types($a.dtype(), $b.dtype());
        let pa = $a.astype(target, numpy_types::Casting::Unsafe)?;
        let pb = $b.astype(target, numpy_types::Casting::Unsafe)?;
        match (&pa, &pb) {
            (numpy_types::DynArray::Bool($va), numpy_types::DynArray::Bool($vb)) => $body,
            (numpy_types::DynArray::I8($va), numpy_types::DynArray::I8($vb)) => $body,
            (numpy_types::DynArray::I16($va), numpy_types::DynArray::I16($vb)) => $body,
            (numpy_types::DynArray::I32($va), numpy_types::DynArray::I32($vb)) => $body,
            (numpy_types::DynArray::I64($va), numpy_types::DynArray::I64($vb)) => $body,
            (numpy_types::DynArray::U8($va), numpy_types::DynArray::U8($vb)) => $body,
            (numpy_types::DynArray::U16($va), numpy_types::DynArray::U16($vb)) => $body,
            (numpy_types::DynArray::U32($va), numpy_types::DynArray::U32($vb)) => $body,
            (numpy_types::DynArray::U64($va), numpy_types::DynArray::U64($vb)) => $body,
            (numpy_types::DynArray::F16($va), numpy_types::DynArray::F16($vb)) => $body,
            (numpy_types::DynArray::F32($va), numpy_types::DynArray::F32($vb)) => $body,
            (numpy_types::DynArray::F64($va), numpy_types::DynArray::F64($vb)) => $body,
            (numpy_types::DynArray::C64($va), numpy_types::DynArray::C64($vb)) => $body,
            (numpy_types::DynArray::C128($va), numpy_types::DynArray::C128($vb)) => $body,
            _ => unreachable!("promote_types guarantees matching variants"),
        }
    }};
}
