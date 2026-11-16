use crate::engine::{c2c_1d, c2c_nd, c2r_1d, c2r_nd, r2c_1d, r2c_nd};
use crate::helpers::shift_view;
use crate::planner::Norm;
use num_complex::{Complex32, Complex64};
use numpy_types::{f16, ArrayView, DynArray, NdArray, NumPyError, Result, Scalar};

/// Trait for array types accepted by complex FFT routines (`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`).
pub trait ComplexFftInput {
    type Output;
    fn fft_1d(&self, n: Option<usize>, axis: isize, is_forward: bool, norm: Norm) -> Result<Self::Output>;
    fn fft_nd(
        &self,
        s: Option<&[isize]>,
        axes: Option<&[isize]>,
        is_forward: bool,
        norm: Norm,
        default_last_two: bool,
    ) -> Result<Self::Output>;
}

/// Trait for real-valued array types accepted by `rfft`, `rfft2`, `rfftn`, and `ihfft`.
pub trait RealFftInput {
    type ComplexOutput;
    fn rfft_1d(&self, n: Option<usize>, axis: isize, norm: Norm, is_ihfft: bool) -> Result<Self::ComplexOutput>;
    fn rfft_nd(
        &self,
        s: Option<&[isize]>,
        axes: Option<&[isize]>,
        norm: Norm,
        default_last_two: bool,
    ) -> Result<Self::ComplexOutput>;
}

/// Trait for array types accepted by `irfft`, `irfft2`, `irfftn`, and `hfft`.
pub trait InverseRealFftInput {
    type RealOutput;
    fn irfft_1d(&self, n: Option<usize>, axis: isize, norm: Norm, is_hfft: bool) -> Result<Self::RealOutput>;
    fn irfft_nd(
        &self,
        s: Option<&[isize]>,
        axes: Option<&[isize]>,
        norm: Norm,
        default_last_two: bool,
    ) -> Result<Self::RealOutput>;
}

/// Trait for array types accepted by `fftshift` and `ifftshift`.
pub trait ShiftInput {
    type Output;
    fn shift(&self, axes: Option<&[isize]>, forward: bool) -> Result<Self::Output>;
}

macro_rules! impl_complex_input {
    ($in_ty:ty => $float_ty:ty, $out_ty:ty) => {
        impl ComplexFftInput for ArrayView<'_, $in_ty> {
            type Output = NdArray<$out_ty>;

            #[inline]
            fn fft_1d(&self, n: Option<usize>, axis: isize, is_forward: bool, norm: Norm) -> Result<Self::Output> {
                c2c_1d::<$in_ty, $float_ty>(self, n, axis, is_forward, norm)
            }

            #[inline]
            fn fft_nd(
                &self,
                s: Option<&[isize]>,
                axes: Option<&[isize]>,
                is_forward: bool,
                norm: Norm,
                default_last_two: bool,
            ) -> Result<Self::Output> {
                c2c_nd::<$in_ty, $float_ty>(self, s, axes, is_forward, norm, default_last_two)
            }
        }

        impl ComplexFftInput for NdArray<$in_ty> {
            type Output = NdArray<$out_ty>;

            #[inline]
            fn fft_1d(&self, n: Option<usize>, axis: isize, is_forward: bool, norm: Norm) -> Result<Self::Output> {
                self.view().fft_1d(n, axis, is_forward, norm)
            }

            #[inline]
            fn fft_nd(
                &self,
                s: Option<&[isize]>,
                axes: Option<&[isize]>,
                is_forward: bool,
                norm: Norm,
                default_last_two: bool,
            ) -> Result<Self::Output> {
                self.view().fft_nd(s, axes, is_forward, norm, default_last_two)
            }
        }
    };
}

impl_complex_input!(f32 => f32, Complex32);
impl_complex_input!(Complex32 => f32, Complex32);
impl_complex_input!(f64 => f64, Complex64);
impl_complex_input!(Complex64 => f64, Complex64);
impl_complex_input!(bool => f64, Complex64);
impl_complex_input!(i8 => f64, Complex64);
impl_complex_input!(i16 => f64, Complex64);
impl_complex_input!(i32 => f64, Complex64);
impl_complex_input!(i64 => f64, Complex64);
impl_complex_input!(u8 => f64, Complex64);
impl_complex_input!(u16 => f64, Complex64);
impl_complex_input!(u32 => f64, Complex64);
impl_complex_input!(u64 => f64, Complex64);
impl_complex_input!(f16 => f32, Complex32);

macro_rules! impl_real_input {
    ($in_ty:ty => $float_ty:ty, $out_ty:ty) => {
        impl RealFftInput for ArrayView<'_, $in_ty> {
            type ComplexOutput = NdArray<$out_ty>;

            #[inline]
            fn rfft_1d(&self, n: Option<usize>, axis: isize, norm: Norm, is_ihfft: bool) -> Result<Self::ComplexOutput> {
                r2c_1d::<$in_ty, $float_ty>(self, n, axis, norm, is_ihfft)
            }

            #[inline]
            fn rfft_nd(
                &self,
                s: Option<&[isize]>,
                axes: Option<&[isize]>,
                norm: Norm,
                default_last_two: bool,
            ) -> Result<Self::ComplexOutput> {
                r2c_nd::<$in_ty, $float_ty>(self, s, axes, norm, default_last_two)
            }
        }

        impl RealFftInput for NdArray<$in_ty> {
            type ComplexOutput = NdArray<$out_ty>;

            #[inline]
            fn rfft_1d(&self, n: Option<usize>, axis: isize, norm: Norm, is_ihfft: bool) -> Result<Self::ComplexOutput> {
                self.view().rfft_1d(n, axis, norm, is_ihfft)
            }

            #[inline]
            fn rfft_nd(
                &self,
                s: Option<&[isize]>,
                axes: Option<&[isize]>,
                norm: Norm,
                default_last_two: bool,
            ) -> Result<Self::ComplexOutput> {
                self.view().rfft_nd(s, axes, norm, default_last_two)
            }
        }
    };
}

impl_real_input!(f32 => f32, Complex32);
impl_real_input!(f64 => f64, Complex64);
impl_real_input!(bool => f64, Complex64);
impl_real_input!(i8 => f64, Complex64);
impl_real_input!(i16 => f64, Complex64);
impl_real_input!(i32 => f64, Complex64);
impl_real_input!(i64 => f64, Complex64);
impl_real_input!(u8 => f64, Complex64);
impl_real_input!(u16 => f64, Complex64);
impl_real_input!(u32 => f64, Complex64);
impl_real_input!(u64 => f64, Complex64);
impl_real_input!(f16 => f32, Complex32);

macro_rules! impl_inv_real_input {
    ($in_ty:ty => $float_ty:ty) => {
        impl InverseRealFftInput for ArrayView<'_, $in_ty> {
            type RealOutput = NdArray<$float_ty>;

            #[inline]
            fn irfft_1d(&self, n: Option<usize>, axis: isize, norm: Norm, is_hfft: bool) -> Result<Self::RealOutput> {
                c2r_1d::<$in_ty, $float_ty>(self, n, axis, norm, is_hfft)
            }

            #[inline]
            fn irfft_nd(
                &self,
                s: Option<&[isize]>,
                axes: Option<&[isize]>,
                norm: Norm,
                default_last_two: bool,
            ) -> Result<Self::RealOutput> {
                c2r_nd::<$in_ty, $float_ty>(self, s, axes, norm, default_last_two)
            }
        }

        impl InverseRealFftInput for NdArray<$in_ty> {
            type RealOutput = NdArray<$float_ty>;

            #[inline]
            fn irfft_1d(&self, n: Option<usize>, axis: isize, norm: Norm, is_hfft: bool) -> Result<Self::RealOutput> {
                self.view().irfft_1d(n, axis, norm, is_hfft)
            }

            #[inline]
            fn irfft_nd(
                &self,
                s: Option<&[isize]>,
                axes: Option<&[isize]>,
                norm: Norm,
                default_last_two: bool,
            ) -> Result<Self::RealOutput> {
                self.view().irfft_nd(s, axes, norm, default_last_two)
            }
        }
    };
}

impl_inv_real_input!(f32 => f32);
impl_inv_real_input!(Complex32 => f32);
impl_inv_real_input!(f64 => f64);
impl_inv_real_input!(Complex64 => f64);
impl_inv_real_input!(bool => f64);
impl_inv_real_input!(i8 => f64);
impl_inv_real_input!(i16 => f64);
impl_inv_real_input!(i32 => f64);
impl_inv_real_input!(i64 => f64);
impl_inv_real_input!(u8 => f64);
impl_inv_real_input!(u16 => f64);
impl_inv_real_input!(u32 => f64);
impl_inv_real_input!(u64 => f64);
impl_inv_real_input!(f16 => f32);

impl<T: Scalar> ShiftInput for ArrayView<'_, T> {
    type Output = NdArray<T>;

    #[inline]
    fn shift(&self, axes: Option<&[isize]>, forward: bool) -> Result<Self::Output> {
        shift_view(self, axes, forward)
    }
}

impl<T: Scalar> ShiftInput for NdArray<T> {
    type Output = NdArray<T>;

    #[inline]
    fn shift(&self, axes: Option<&[isize]>, forward: bool) -> Result<Self::Output> {
        shift_view(&self.view(), axes, forward)
    }
}

impl ComplexFftInput for DynArray {
    type Output = DynArray;

    fn fft_1d(&self, n: Option<usize>, axis: isize, is_forward: bool, norm: Norm) -> Result<DynArray> {
        Ok(match self {
            DynArray::F32(a) => DynArray::C64(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::C64(a) => DynArray::C64(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::F64(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::C128(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::Bool(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::I8(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::I16(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::I32(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::I64(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::U8(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::U16(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::U32(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::U64(a) => DynArray::C128(a.fft_1d(n, axis, is_forward, norm)?),
            DynArray::F16(a) => DynArray::C64(a.fft_1d(n, axis, is_forward, norm)?),
        })
    }

    fn fft_nd(
        &self,
        s: Option<&[isize]>,
        axes: Option<&[isize]>,
        is_forward: bool,
        norm: Norm,
        default_last_two: bool,
    ) -> Result<DynArray> {
        Ok(match self {
            DynArray::F32(a) => DynArray::C64(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::C64(a) => DynArray::C64(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::F64(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::C128(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::Bool(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::I8(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::I16(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::I32(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::I64(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::U8(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::U16(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::U32(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::U64(a) => DynArray::C128(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
            DynArray::F16(a) => DynArray::C64(a.fft_nd(s, axes, is_forward, norm, default_last_two)?),
        })
    }
}

impl RealFftInput for DynArray {
    type ComplexOutput = DynArray;

    fn rfft_1d(&self, n: Option<usize>, axis: isize, norm: Norm, is_ihfft: bool) -> Result<DynArray> {
        Ok(match self {
            DynArray::F32(a) => DynArray::C64(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::F64(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::Bool(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::I8(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::I16(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::I32(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::I64(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::U8(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::U16(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::U32(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::U64(a) => DynArray::C128(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::F16(a) => DynArray::C64(a.rfft_1d(n, axis, norm, is_ihfft)?),
            DynArray::C64(_) | DynArray::C128(_) => {
                return Err(NumPyError::DTypeError(format!(
                    "real FFT does not support complex input dtype '{}'",
                    self.dtype()
                )));
            }
        })
    }

    fn rfft_nd(
        &self,
        s: Option<&[isize]>,
        axes: Option<&[isize]>,
        norm: Norm,
        default_last_two: bool,
    ) -> Result<DynArray> {
        Ok(match self {
            DynArray::F32(a) => DynArray::C64(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::F64(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::Bool(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::I8(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::I16(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::I32(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::I64(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::U8(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::U16(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::U32(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::U64(a) => DynArray::C128(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::F16(a) => DynArray::C64(a.rfft_nd(s, axes, norm, default_last_two)?),
            DynArray::C64(_) | DynArray::C128(_) => {
                return Err(NumPyError::DTypeError(format!(
                    "real FFT does not support complex input dtype '{}'",
                    self.dtype()
                )));
            }
        })
    }
}

impl InverseRealFftInput for DynArray {
    type RealOutput = DynArray;

    fn irfft_1d(&self, n: Option<usize>, axis: isize, norm: Norm, is_hfft: bool) -> Result<DynArray> {
        Ok(match self {
            DynArray::F32(a) => DynArray::F32(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::C64(a) => DynArray::F32(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::F64(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::C128(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::Bool(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::I8(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::I16(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::I32(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::I64(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::U8(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::U16(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::U32(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::U64(a) => DynArray::F64(a.irfft_1d(n, axis, norm, is_hfft)?),
            DynArray::F16(a) => DynArray::F32(a.irfft_1d(n, axis, norm, is_hfft)?),
        })
    }

    fn irfft_nd(
        &self,
        s: Option<&[isize]>,
        axes: Option<&[isize]>,
        norm: Norm,
        default_last_two: bool,
    ) -> Result<DynArray> {
        Ok(match self {
            DynArray::F32(a) => DynArray::F32(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::C64(a) => DynArray::F32(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::F64(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::C128(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::Bool(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::I8(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::I16(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::I32(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::I64(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::U8(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::U16(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::U32(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::U64(a) => DynArray::F64(a.irfft_nd(s, axes, norm, default_last_two)?),
            DynArray::F16(a) => DynArray::F32(a.irfft_nd(s, axes, norm, default_last_two)?),
        })
    }
}

impl ShiftInput for DynArray {
    type Output = DynArray;

    fn shift(&self, axes: Option<&[isize]>, forward: bool) -> Result<DynArray> {
        Ok(match self {
            DynArray::Bool(a) => DynArray::Bool(a.shift(axes, forward)?),
            DynArray::I8(a) => DynArray::I8(a.shift(axes, forward)?),
            DynArray::I16(a) => DynArray::I16(a.shift(axes, forward)?),
            DynArray::I32(a) => DynArray::I32(a.shift(axes, forward)?),
            DynArray::I64(a) => DynArray::I64(a.shift(axes, forward)?),
            DynArray::U8(a) => DynArray::U8(a.shift(axes, forward)?),
            DynArray::U16(a) => DynArray::U16(a.shift(axes, forward)?),
            DynArray::U32(a) => DynArray::U32(a.shift(axes, forward)?),
            DynArray::U64(a) => DynArray::U64(a.shift(axes, forward)?),
            DynArray::F16(a) => DynArray::F16(a.shift(axes, forward)?),
            DynArray::F32(a) => DynArray::F32(a.shift(axes, forward)?),
            DynArray::F64(a) => DynArray::F64(a.shift(axes, forward)?),
            DynArray::C64(a) => DynArray::C64(a.shift(axes, forward)?),
            DynArray::C128(a) => DynArray::C128(a.shift(axes, forward)?),
        })
    }
}

/// Compute the 1-D discrete Fourier Transform (`numpy.fft.fft`).
#[inline]
pub fn fft<A: ComplexFftInput + ?Sized>(
    a: &A,
    n: Option<usize>,
    axis: isize,
    norm: Norm,
) -> Result<A::Output> {
    a.fft_1d(n, axis, true, norm)
}

/// Compute the 1-D inverse discrete Fourier Transform (`numpy.fft.ifft`).
#[inline]
pub fn ifft<A: ComplexFftInput + ?Sized>(
    a: &A,
    n: Option<usize>,
    axis: isize,
    norm: Norm,
) -> Result<A::Output> {
    a.fft_1d(n, axis, false, norm)
}

/// Compute the 2-D discrete Fourier Transform (`numpy.fft.fft2`).
#[inline]
pub fn fft2<A: ComplexFftInput + ?Sized>(
    a: &A,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
) -> Result<A::Output> {
    a.fft_nd(s, axes, true, norm, true)
}

/// Compute the 2-D inverse discrete Fourier Transform (`numpy.fft.ifft2`).
#[inline]
pub fn ifft2<A: ComplexFftInput + ?Sized>(
    a: &A,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
) -> Result<A::Output> {
    a.fft_nd(s, axes, false, norm, true)
}

/// Compute the N-D discrete Fourier Transform (`numpy.fft.fftn`).
#[inline]
pub fn fftn<A: ComplexFftInput + ?Sized>(
    a: &A,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
) -> Result<A::Output> {
    a.fft_nd(s, axes, true, norm, false)
}

/// Compute the N-D inverse discrete Fourier Transform (`numpy.fft.ifftn`).
#[inline]
pub fn ifftn<A: ComplexFftInput + ?Sized>(
    a: &A,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
) -> Result<A::Output> {
    a.fft_nd(s, axes, false, norm, false)
}

/// Compute the 1-D discrete Fourier Transform for real input (`numpy.fft.rfft`).
#[inline]
pub fn rfft<A: RealFftInput + ?Sized>(
    a: &A,
    n: Option<usize>,
    axis: isize,
    norm: Norm,
) -> Result<A::ComplexOutput> {
    a.rfft_1d(n, axis, norm, false)
}

/// Compute the inverse of `rfft` (`numpy.fft.irfft`).
#[inline]
pub fn irfft<A: InverseRealFftInput + ?Sized>(
    a: &A,
    n: Option<usize>,
    axis: isize,
    norm: Norm,
) -> Result<A::RealOutput> {
    a.irfft_1d(n, axis, norm, false)
}

/// Compute the 2-D discrete Fourier Transform of a real array (`numpy.fft.rfft2`).
#[inline]
pub fn rfft2<A: RealFftInput + ?Sized>(
    a: &A,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
) -> Result<A::ComplexOutput> {
    a.rfft_nd(s, axes, norm, true)
}

/// Compute the inverse of `rfft2` (`numpy.fft.irfft2`).
#[inline]
pub fn irfft2<A: InverseRealFftInput + ?Sized>(
    a: &A,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
) -> Result<A::RealOutput> {
    a.irfft_nd(s, axes, norm, true)
}

/// Compute the N-D discrete Fourier Transform for real input (`numpy.fft.rfftn`).
#[inline]
pub fn rfftn<A: RealFftInput + ?Sized>(
    a: &A,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
) -> Result<A::ComplexOutput> {
    a.rfft_nd(s, axes, norm, false)
}

/// Compute the inverse of `rfftn` (`numpy.fft.irfftn`).
#[inline]
pub fn irfftn<A: InverseRealFftInput + ?Sized>(
    a: &A,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
) -> Result<A::RealOutput> {
    a.irfft_nd(s, axes, norm, false)
}

/// Compute the FFT of a signal with Hermitian symmetry (`numpy.fft.hfft`).
#[inline]
pub fn hfft<A: InverseRealFftInput + ?Sized>(
    a: &A,
    n: Option<usize>,
    axis: isize,
    norm: Norm,
) -> Result<A::RealOutput> {
    a.irfft_1d(n, axis, norm, true)
}

/// Compute the inverse FFT of a signal with Hermitian symmetry (`numpy.fft.ihfft`).
#[inline]
pub fn ihfft<A: RealFftInput + ?Sized>(
    a: &A,
    n: Option<usize>,
    axis: isize,
    norm: Norm,
) -> Result<A::ComplexOutput> {
    a.rfft_1d(n, axis, norm, true)
}

/// Shift the zero-frequency component to the center of the spectrum (`numpy.fft.fftshift`).
#[inline]
pub fn fftshift<A: ShiftInput + ?Sized>(x: &A, axes: Option<&[isize]>) -> Result<A::Output> {
    x.shift(axes, true)
}

/// Inverse of `fftshift` (`numpy.fft.ifftshift`).
#[inline]
pub fn ifftshift<A: ShiftInput + ?Sized>(x: &A, axes: Option<&[isize]>) -> Result<A::Output> {
    x.shift(axes, false)
}
