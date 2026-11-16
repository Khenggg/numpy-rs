//! Runtime-dispatched x86_64 AVX2 + FMA kernels with scalar fallbacks.
//!
//! All hot paths operate directly on `&[T]` and `&mut [MaybeUninit<U>]` so callers
//! allocate uninitialized `AlignedUninit<U>` buffers without pre-zeroing.

use std::mem::MaybeUninit;

/// Returns `true` when both AVX2 and FMA are supported by the host CPU at runtime.
#[inline]
pub fn is_avx2_fma_active() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use crate::scalar_ops::{
        fmax_f32, fmax_f64, fmin_f32, fmin_f64, lshift_i32, lshift_i64, max_prop_nan_f32,
        max_prop_nan_f64, min_prop_nan_f32, min_prop_nan_f64, rshift_i32,
    };
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;

    const fn make_mask4() -> [u32; 16] {
        let mut t = [0u32; 16];
        let mut m = 0usize;
        while m < 16 {
            let b0 = (m & 1) as u32;
            let b1 = ((m >> 1) & 1) as u32;
            let b2 = ((m >> 2) & 1) as u32;
            let b3 = ((m >> 3) & 1) as u32;
            t[m] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            m += 1;
        }
        t
    }

    const fn make_mask8() -> [u64; 256] {
        let mut t = [0u64; 256];
        let mut m = 0usize;
        while m < 256 {
            let mut v = 0u64;
            let mut k = 0usize;
            while k < 8 {
                v |= (((m >> k) & 1) as u64) << (k * 8);
                k += 1;
            }
            t[m] = v;
            m += 1;
        }
        t
    }

    static MASK4: [u32; 16] = make_mask4();
    static MASK8: [u64; 256] = make_mask8();

    #[inline(always)]
    unsafe fn store_bool4(out_ptr: *mut bool, mask: i32) {
        let bits = MASK4[(mask as usize) & 0xF];
        unsafe { (out_ptr as *mut u32).write_unaligned(bits) };
    }

    #[inline(always)]
    unsafe fn store_bool8(out_ptr: *mut bool, mask: i32) {
        let bits = MASK8[(mask as usize) & 0xFF];
        unsafe { (out_ptr as *mut u64).write_unaligned(bits) };
    }

    // ------------------------------------------------------------------------
    // f64 kernels (4 lanes per __m256d)
    // ------------------------------------------------------------------------

    macro_rules! impl_f64_bin {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[f64],
                b: &[f64],
                out: &mut [MaybeUninit<f64>],
            ) {
                let n = a.len().min(b.len()).min(out.len());
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let op = out.as_mut_ptr() as *mut f64;
                let mut i = 0usize;
                unsafe {
                    while i + 4 <= n {
                        let va = _mm256_loadu_pd(ap.add(i));
                        let vb = _mm256_loadu_pd(bp.add(i));
                        let vr = ($vec_op)(va, vb);
                        _mm256_storeu_pd(op.add(i), vr);
                        i += 4;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), *bp.add(i));
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f64_bin!(add_f64, |x: f64, y: f64| x + y, |a, b| _mm256_add_pd(a, b));
    impl_f64_bin!(sub_f64, |x: f64, y: f64| x - y, |a, b| _mm256_sub_pd(a, b));
    impl_f64_bin!(mul_f64, |x: f64, y: f64| x * y, |a, b| _mm256_mul_pd(a, b));
    impl_f64_bin!(div_f64, |x: f64, y: f64| x / y, |a, b| _mm256_div_pd(a, b));

    #[inline(always)]
    unsafe fn vec_max_prop_nan_f64(a: __m256d, b: __m256d) -> __m256d {
        unsafe {
            let mask = _mm256_or_pd(
                _mm256_cmp_pd::<_CMP_GT_OQ>(a, b),
                _mm256_cmp_pd::<_CMP_UNORD_Q>(a, a),
            );
            _mm256_blendv_pd(b, a, mask)
        }
    }

    #[inline(always)]
    unsafe fn vec_min_prop_nan_f64(a: __m256d, b: __m256d) -> __m256d {
        unsafe {
            let mask = _mm256_or_pd(
                _mm256_cmp_pd::<_CMP_LT_OQ>(a, b),
                _mm256_cmp_pd::<_CMP_UNORD_Q>(a, a),
            );
            _mm256_blendv_pd(b, a, mask)
        }
    }

    #[inline(always)]
    unsafe fn vec_fmax_f64(a: __m256d, b: __m256d) -> __m256d {
        unsafe {
            let mask = _mm256_or_pd(
                _mm256_cmp_pd::<_CMP_GT_OQ>(a, b),
                _mm256_cmp_pd::<_CMP_UNORD_Q>(b, b),
            );
            _mm256_blendv_pd(b, a, mask)
        }
    }

    #[inline(always)]
    unsafe fn vec_fmin_f64(a: __m256d, b: __m256d) -> __m256d {
        unsafe {
            let mask = _mm256_or_pd(
                _mm256_cmp_pd::<_CMP_LT_OQ>(a, b),
                _mm256_cmp_pd::<_CMP_UNORD_Q>(b, b),
            );
            _mm256_blendv_pd(b, a, mask)
        }
    }

    impl_f64_bin!(maximum_f64, max_prop_nan_f64, |a, b| vec_max_prop_nan_f64(a, b));
    impl_f64_bin!(minimum_f64, min_prop_nan_f64, |a, b| vec_min_prop_nan_f64(a, b));
    impl_f64_bin!(fmax_f64_vec, fmax_f64, |a, b| vec_fmax_f64(a, b));
    impl_f64_bin!(fmin_f64_vec, fmin_f64, |a, b| vec_fmin_f64(a, b));

    macro_rules! impl_f64_scalar_rhs {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[f64],
                s: f64,
                out: &mut [MaybeUninit<f64>],
            ) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut f64;
                let mut i = 0usize;
                unsafe {
                    let vs = _mm256_set1_pd(s);
                    while i + 4 <= n {
                        let va = _mm256_loadu_pd(ap.add(i));
                        _mm256_storeu_pd(op.add(i), ($vec_op)(va, vs));
                        i += 4;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), s);
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f64_scalar_rhs!(add_f64_scalar, |x: f64, y: f64| x + y, |a, b| _mm256_add_pd(a, b));
    impl_f64_scalar_rhs!(sub_f64_scalar, |x: f64, y: f64| x - y, |a, b| _mm256_sub_pd(a, b));
    impl_f64_scalar_rhs!(rsub_f64_scalar, |x: f64, y: f64| y - x, |a, b| _mm256_sub_pd(b, a));
    impl_f64_scalar_rhs!(mul_f64_scalar, |x: f64, y: f64| x * y, |a, b| _mm256_mul_pd(a, b));
    impl_f64_scalar_rhs!(div_f64_scalar, |x: f64, y: f64| x / y, |a, b| _mm256_div_pd(a, b));
    impl_f64_scalar_rhs!(rdiv_f64_scalar, |x: f64, y: f64| y / x, |a, b| _mm256_div_pd(b, a));
    impl_f64_scalar_rhs!(maximum_f64_scalar, max_prop_nan_f64, |a, b| vec_max_prop_nan_f64(a, b));
    impl_f64_scalar_rhs!(minimum_f64_scalar, min_prop_nan_f64, |a, b| vec_min_prop_nan_f64(a, b));
    impl_f64_scalar_rhs!(
        rmaximum_f64_scalar,
        |x: f64, s: f64| max_prop_nan_f64(s, x),
        |a, b| vec_max_prop_nan_f64(b, a)
    );
    impl_f64_scalar_rhs!(
        rminimum_f64_scalar,
        |x: f64, s: f64| min_prop_nan_f64(s, x),
        |a, b| vec_min_prop_nan_f64(b, a)
    );
    impl_f64_scalar_rhs!(fmax_f64_scalar, fmax_f64, |a, b| vec_fmax_f64(a, b));
    impl_f64_scalar_rhs!(fmin_f64_scalar, fmin_f64, |a, b| vec_fmin_f64(a, b));
    impl_f64_scalar_rhs!(
        rfmax_f64_scalar,
        |x: f64, s: f64| fmax_f64(s, x),
        |a, b| vec_fmax_f64(b, a)
    );
    impl_f64_scalar_rhs!(
        rfmin_f64_scalar,
        |x: f64, s: f64| fmin_f64(s, x),
        |a, b| vec_fmin_f64(b, a)
    );

    macro_rules! impl_f64_unary {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(a: &[f64], out: &mut [MaybeUninit<f64>]) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut f64;
                let mut i = 0usize;
                unsafe {
                    while i + 4 <= n {
                        let va = _mm256_loadu_pd(ap.add(i));
                        _mm256_storeu_pd(op.add(i), ($vec_op)(va));
                        i += 4;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i));
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f64_unary!(neg_f64, |x: f64| -x, |a| _mm256_xor_pd(a, _mm256_set1_pd(-0.0)));
    impl_f64_unary!(abs_f64, f64::abs, |a| _mm256_andnot_pd(_mm256_set1_pd(-0.0), a));
    impl_f64_unary!(square_f64, |x: f64| x * x, |a| _mm256_mul_pd(a, a));
    impl_f64_unary!(sqrt_f64, f64::sqrt, |a| _mm256_sqrt_pd(a));
    impl_f64_unary!(recip_f64, |x: f64| 1.0 / x, |a| _mm256_div_pd(_mm256_set1_pd(1.0), a));
    impl_f64_unary!(floor_f64, f64::floor, |a| _mm256_floor_pd(a));
    impl_f64_unary!(ceil_f64, f64::ceil, |a| _mm256_ceil_pd(a));
    impl_f64_unary!(trunc_f64, f64::trunc, |a| _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(a));
    impl_f64_unary!(rint_f64, f64::round_ties_even, |a| _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a));
    impl_f64_unary!(deg2rad_f64, f64::to_radians, |a| _mm256_mul_pd(a, _mm256_set1_pd(std::f64::consts::PI / 180.0)));
    impl_f64_unary!(rad2deg_f64, f64::to_degrees, |a| _mm256_mul_pd(a, _mm256_set1_pd(180.0 / std::f64::consts::PI)));

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn clip_scalar_f64(
        a: &[f64],
        lo: f64,
        hi: f64,
        out: &mut [MaybeUninit<f64>],
    ) {
        let n = a.len().min(out.len());
        let ap = a.as_ptr();
        let op = out.as_mut_ptr() as *mut f64;
        if lo.is_nan() || hi.is_nan() {
            for i in 0..n {
                unsafe { *op.add(i) = f64::NAN };
            }
            return;
        }
        let mut i = 0usize;
        unsafe {
            let vlo = _mm256_set1_pd(lo);
            let vhi = _mm256_set1_pd(hi);
            while i + 4 <= n {
                let vx = _mm256_loadu_pd(ap.add(i));
                // Remember VMAXPD(src1, src2) returns src2 if either is NaN, preserving vx=NaN!
                let clamped_lo = _mm256_max_pd(vlo, vx);
                let clamped = _mm256_min_pd(vhi, clamped_lo);
                _mm256_storeu_pd(op.add(i), clamped);
                i += 4;
            }
            while i < n {
                *op.add(i) = crate::scalar_ops::clip_f64_elem(*ap.add(i), lo, hi);
                i += 1;
            }
        }
    }

    macro_rules! impl_f64_unary_bool {
        ($name:ident, $scalar:expr, $mask_fn:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(a: &[f64], out: &mut [MaybeUninit<bool>]) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    while i + 4 <= n {
                        let va = _mm256_loadu_pd(ap.add(i));
                        let m = ($mask_fn)(va);
                        store_bool4(op.add(i), m);
                        i += 4;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i));
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f64_unary_bool!(isfinite_f64, f64::is_finite, |a: __m256d| {
        let abs_bits = _mm256_and_si256(
            _mm256_castpd_si256(a),
            _mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF_u64 as i64),
        );
        let inf_bits = _mm256_set1_epi64x(0x7FF0_0000_0000_0000_u64 as i64);
        let cmp = _mm256_cmpgt_epi64(inf_bits, abs_bits);
        _mm256_movemask_pd(_mm256_castsi256_pd(cmp))
    });
    impl_f64_unary_bool!(isinf_f64, f64::is_infinite, |a: __m256d| {
        let abs_bits = _mm256_and_si256(
            _mm256_castpd_si256(a),
            _mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF_u64 as i64),
        );
        let inf_bits = _mm256_set1_epi64x(0x7FF0_0000_0000_0000_u64 as i64);
        let cmp = _mm256_cmpeq_epi64(abs_bits, inf_bits);
        _mm256_movemask_pd(_mm256_castsi256_pd(cmp))
    });
    impl_f64_unary_bool!(isnan_f64, f64::is_nan, |a: __m256d| {
        _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_UNORD_Q>(a, a))
    });
    impl_f64_unary_bool!(signbit_f64, f64::is_sign_negative, |a: __m256d| {
        _mm256_movemask_pd(a)
    });
    impl_f64_unary_bool!(logical_not_f64, |x: f64| x == 0.0, |a: __m256d| {
        _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_EQ_OQ>(a, _mm256_setzero_pd()))
    });

    macro_rules! impl_f64_cmp {
        ($name:ident, $name_scalar:ident, $scalar:expr, $imm:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[f64],
                b: &[f64],
                out: &mut [MaybeUninit<bool>],
            ) {
                let n = a.len().min(b.len()).min(out.len());
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    while i + 4 <= n {
                        let va = _mm256_loadu_pd(ap.add(i));
                        let vb = _mm256_loadu_pd(bp.add(i));
                        let m = _mm256_movemask_pd(_mm256_cmp_pd::<$imm>(va, vb));
                        store_bool4(op.add(i), m);
                        i += 4;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), *bp.add(i));
                        i += 1;
                    }
                }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name_scalar(
                a: &[f64],
                s: f64,
                out: &mut [MaybeUninit<bool>],
            ) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    let vs = _mm256_set1_pd(s);
                    while i + 4 <= n {
                        let va = _mm256_loadu_pd(ap.add(i));
                        let m = _mm256_movemask_pd(_mm256_cmp_pd::<$imm>(va, vs));
                        store_bool4(op.add(i), m);
                        i += 4;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), s);
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f64_cmp!(eq_f64, eq_f64_scalar, |x: f64, y: f64| x == y, _CMP_EQ_OQ);
    impl_f64_cmp!(ne_f64, ne_f64_scalar, |x: f64, y: f64| x != y, _CMP_NEQ_UQ);
    impl_f64_cmp!(lt_f64, lt_f64_scalar, |x: f64, y: f64| x < y, _CMP_LT_OQ);
    impl_f64_cmp!(le_f64, le_f64_scalar, |x: f64, y: f64| x <= y, _CMP_LE_OQ);
    impl_f64_cmp!(gt_f64, gt_f64_scalar, |x: f64, y: f64| x > y, _CMP_GT_OQ);
    impl_f64_cmp!(ge_f64, ge_f64_scalar, |x: f64, y: f64| x >= y, _CMP_GE_OQ);

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn isclose_f64(
        a: &[f64],
        b: &[f64],
        rtol: f64,
        atol: f64,
        equal_nan: bool,
        out: &mut [MaybeUninit<bool>],
    ) {
        let n = a.len().min(b.len()).min(out.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let op = out.as_mut_ptr() as *mut bool;
        let mut i = 0usize;
        unsafe {
            let vrtol = _mm256_set1_pd(rtol);
            let vatol = _mm256_set1_pd(atol);
            let sign_mask = _mm256_set1_pd(-0.0);
            let abs_i_mask = _mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF_u64 as i64);
            let inf_bits = _mm256_set1_epi64x(0x7FF0_0000_0000_0000_u64 as i64);
            while i + 4 <= n {
                let va = _mm256_loadu_pd(ap.add(i));
                let vb = _mm256_loadu_pd(bp.add(i));
                let eq = _mm256_cmp_pd::<_CMP_EQ_OQ>(va, vb);
                let diff = _mm256_andnot_pd(sign_mask, _mm256_sub_pd(va, vb));
                let abs_b = _mm256_andnot_pd(sign_mask, vb);
                let tol = _mm256_fmadd_pd(vrtol, abs_b, vatol);
                let within = _mm256_cmp_pd::<_CMP_LE_OQ>(diff, tol);
                let b_abs = _mm256_and_si256(_mm256_castpd_si256(vb), abs_i_mask);
                let b_fin = _mm256_castsi256_pd(_mm256_cmpgt_epi64(inf_bits, b_abs));
                let fin_within = _mm256_and_pd(b_fin, within);
                let mut res = _mm256_or_pd(eq, fin_within);
                if equal_nan {
                    let nan_ab = _mm256_and_pd(
                        _mm256_cmp_pd::<_CMP_UNORD_Q>(va, va),
                        _mm256_cmp_pd::<_CMP_UNORD_Q>(vb, vb),
                    );
                    res = _mm256_or_pd(res, nan_ab);
                }
                store_bool4(op.add(i), _mm256_movemask_pd(res));
                i += 4;
            }
            while i < n {
                *op.add(i) = crate::scalar_ops::isclose_f64_elem(
                    *ap.add(i),
                    *bp.add(i),
                    rtol,
                    atol,
                    equal_nan,
                );
                i += 1;
            }
        }
    }

    // ------------------------------------------------------------------------
    // f32 kernels (8 lanes per __m256)
    // ------------------------------------------------------------------------

    macro_rules! impl_f32_bin {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[f32],
                b: &[f32],
                out: &mut [MaybeUninit<f32>],
            ) {
                let n = a.len().min(b.len()).min(out.len());
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let op = out.as_mut_ptr() as *mut f32;
                let mut i = 0usize;
                unsafe {
                    while i + 8 <= n {
                        let va = _mm256_loadu_ps(ap.add(i));
                        let vb = _mm256_loadu_ps(bp.add(i));
                        _mm256_storeu_ps(op.add(i), ($vec_op)(va, vb));
                        i += 8;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), *bp.add(i));
                        i += 1;
                    }
                }
            }
        };
    }

    #[inline(always)]
    unsafe fn vec_max_prop_nan_f32(a: __m256, b: __m256) -> __m256 {
        unsafe {
            let mask = _mm256_or_ps(
                _mm256_cmp_ps::<_CMP_GT_OQ>(a, b),
                _mm256_cmp_ps::<_CMP_UNORD_Q>(a, a),
            );
            _mm256_blendv_ps(b, a, mask)
        }
    }

    #[inline(always)]
    unsafe fn vec_min_prop_nan_f32(a: __m256, b: __m256) -> __m256 {
        unsafe {
            let mask = _mm256_or_ps(
                _mm256_cmp_ps::<_CMP_LT_OQ>(a, b),
                _mm256_cmp_ps::<_CMP_UNORD_Q>(a, a),
            );
            _mm256_blendv_ps(b, a, mask)
        }
    }

    #[inline(always)]
    unsafe fn vec_fmax_f32(a: __m256, b: __m256) -> __m256 {
        unsafe {
            let mask = _mm256_or_ps(
                _mm256_cmp_ps::<_CMP_GT_OQ>(a, b),
                _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b),
            );
            _mm256_blendv_ps(b, a, mask)
        }
    }

    #[inline(always)]
    unsafe fn vec_fmin_f32(a: __m256, b: __m256) -> __m256 {
        unsafe {
            let mask = _mm256_or_ps(
                _mm256_cmp_ps::<_CMP_LT_OQ>(a, b),
                _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b),
            );
            _mm256_blendv_ps(b, a, mask)
        }
    }

    impl_f32_bin!(add_f32, |x: f32, y: f32| x + y, |a, b| _mm256_add_ps(a, b));
    impl_f32_bin!(sub_f32, |x: f32, y: f32| x - y, |a, b| _mm256_sub_ps(a, b));
    impl_f32_bin!(mul_f32, |x: f32, y: f32| x * y, |a, b| _mm256_mul_ps(a, b));
    impl_f32_bin!(div_f32, |x: f32, y: f32| x / y, |a, b| _mm256_div_ps(a, b));
    impl_f32_bin!(maximum_f32, max_prop_nan_f32, |a, b| vec_max_prop_nan_f32(a, b));
    impl_f32_bin!(minimum_f32, min_prop_nan_f32, |a, b| vec_min_prop_nan_f32(a, b));
    impl_f32_bin!(fmax_f32_vec, fmax_f32, |a, b| vec_fmax_f32(a, b));
    impl_f32_bin!(fmin_f32_vec, fmin_f32, |a, b| vec_fmin_f32(a, b));

    macro_rules! impl_f32_scalar_rhs {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[f32],
                s: f32,
                out: &mut [MaybeUninit<f32>],
            ) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut f32;
                let mut i = 0usize;
                unsafe {
                    let vs = _mm256_set1_ps(s);
                    while i + 8 <= n {
                        let va = _mm256_loadu_ps(ap.add(i));
                        _mm256_storeu_ps(op.add(i), ($vec_op)(va, vs));
                        i += 8;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), s);
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f32_scalar_rhs!(add_f32_scalar, |x: f32, y: f32| x + y, |a, b| _mm256_add_ps(a, b));
    impl_f32_scalar_rhs!(sub_f32_scalar, |x: f32, y: f32| x - y, |a, b| _mm256_sub_ps(a, b));
    impl_f32_scalar_rhs!(rsub_f32_scalar, |x: f32, y: f32| y - x, |a, b| _mm256_sub_ps(b, a));
    impl_f32_scalar_rhs!(mul_f32_scalar, |x: f32, y: f32| x * y, |a, b| _mm256_mul_ps(a, b));
    impl_f32_scalar_rhs!(div_f32_scalar, |x: f32, y: f32| x / y, |a, b| _mm256_div_ps(a, b));
    impl_f32_scalar_rhs!(rdiv_f32_scalar, |x: f32, y: f32| y / x, |a, b| _mm256_div_ps(b, a));
    impl_f32_scalar_rhs!(maximum_f32_scalar, max_prop_nan_f32, |a, b| vec_max_prop_nan_f32(a, b));
    impl_f32_scalar_rhs!(minimum_f32_scalar, min_prop_nan_f32, |a, b| vec_min_prop_nan_f32(a, b));
    impl_f32_scalar_rhs!(
        rmaximum_f32_scalar,
        |x: f32, s: f32| max_prop_nan_f32(s, x),
        |a, b| vec_max_prop_nan_f32(b, a)
    );
    impl_f32_scalar_rhs!(
        rminimum_f32_scalar,
        |x: f32, s: f32| min_prop_nan_f32(s, x),
        |a, b| vec_min_prop_nan_f32(b, a)
    );
    impl_f32_scalar_rhs!(fmax_f32_scalar, fmax_f32, |a, b| vec_fmax_f32(a, b));
    impl_f32_scalar_rhs!(fmin_f32_scalar, fmin_f32, |a, b| vec_fmin_f32(a, b));
    impl_f32_scalar_rhs!(
        rfmax_f32_scalar,
        |x: f32, s: f32| fmax_f32(s, x),
        |a, b| vec_fmax_f32(b, a)
    );
    impl_f32_scalar_rhs!(
        rfmin_f32_scalar,
        |x: f32, s: f32| fmin_f32(s, x),
        |a, b| vec_fmin_f32(b, a)
    );

    macro_rules! impl_f32_unary {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(a: &[f32], out: &mut [MaybeUninit<f32>]) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut f32;
                let mut i = 0usize;
                unsafe {
                    while i + 8 <= n {
                        let va = _mm256_loadu_ps(ap.add(i));
                        _mm256_storeu_ps(op.add(i), ($vec_op)(va));
                        i += 8;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i));
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f32_unary!(neg_f32, |x: f32| -x, |a| _mm256_xor_ps(a, _mm256_set1_ps(-0.0)));
    impl_f32_unary!(abs_f32, f32::abs, |a| _mm256_andnot_ps(_mm256_set1_ps(-0.0), a));
    impl_f32_unary!(square_f32, |x: f32| x * x, |a| _mm256_mul_ps(a, a));
    impl_f32_unary!(sqrt_f32, f32::sqrt, |a| _mm256_sqrt_ps(a));
    impl_f32_unary!(recip_f32, |x: f32| 1.0_f32 / x, |a| _mm256_div_ps(_mm256_set1_ps(1.0), a));
    impl_f32_unary!(floor_f32, f32::floor, |a| _mm256_floor_ps(a));
    impl_f32_unary!(ceil_f32, f32::ceil, |a| _mm256_ceil_ps(a));
    impl_f32_unary!(trunc_f32, f32::trunc, |a| _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(a));
    impl_f32_unary!(rint_f32, f32::round_ties_even, |a| _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a));
    impl_f32_unary!(deg2rad_f32, f32::to_radians, |a| _mm256_mul_ps(a, _mm256_set1_ps(std::f32::consts::PI / 180.0)));
    impl_f32_unary!(rad2deg_f32, f32::to_degrees, |a| _mm256_mul_ps(a, _mm256_set1_ps(180.0 / std::f32::consts::PI)));

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn clip_scalar_f32(
        a: &[f32],
        lo: f32,
        hi: f32,
        out: &mut [MaybeUninit<f32>],
    ) {
        let n = a.len().min(out.len());
        let ap = a.as_ptr();
        let op = out.as_mut_ptr() as *mut f32;
        if lo.is_nan() || hi.is_nan() {
            for i in 0..n {
                unsafe { *op.add(i) = f32::NAN };
            }
            return;
        }
        let mut i = 0usize;
        unsafe {
            let vlo = _mm256_set1_ps(lo);
            let vhi = _mm256_set1_ps(hi);
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(ap.add(i));
                let clamped = _mm256_min_ps(vhi, _mm256_max_ps(vlo, vx));
                _mm256_storeu_ps(op.add(i), clamped);
                i += 8;
            }
            while i < n {
                *op.add(i) = crate::scalar_ops::clip_f32_elem(*ap.add(i), lo, hi);
                i += 1;
            }
        }
    }

    macro_rules! impl_f32_unary_bool {
        ($name:ident, $scalar:expr, $mask_fn:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(a: &[f32], out: &mut [MaybeUninit<bool>]) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    while i + 8 <= n {
                        let va = _mm256_loadu_ps(ap.add(i));
                        store_bool8(op.add(i), ($mask_fn)(va));
                        i += 8;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i));
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f32_unary_bool!(isfinite_f32, f32::is_finite, |a: __m256| {
        let abs_bits = _mm256_and_si256(_mm256_castps_si256(a), _mm256_set1_epi32(0x7FFF_FFFF));
        let inf_bits = _mm256_set1_epi32(0x7F80_0000);
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(inf_bits, abs_bits)))
    });
    impl_f32_unary_bool!(isinf_f32, f32::is_infinite, |a: __m256| {
        let abs_bits = _mm256_and_si256(_mm256_castps_si256(a), _mm256_set1_epi32(0x7FFF_FFFF));
        let inf_bits = _mm256_set1_epi32(0x7F80_0000);
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(abs_bits, inf_bits)))
    });
    impl_f32_unary_bool!(isnan_f32, f32::is_nan, |a: __m256| {
        _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(a, a))
    });
    impl_f32_unary_bool!(signbit_f32, f32::is_sign_negative, |a: __m256| {
        _mm256_movemask_ps(a)
    });
    impl_f32_unary_bool!(logical_not_f32, |x: f32| x == 0.0, |a: __m256| {
        _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(a, _mm256_setzero_ps()))
    });

    macro_rules! impl_f32_cmp {
        ($name:ident, $name_scalar:ident, $scalar:expr, $imm:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[f32],
                b: &[f32],
                out: &mut [MaybeUninit<bool>],
            ) {
                let n = a.len().min(b.len()).min(out.len());
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    while i + 8 <= n {
                        let va = _mm256_loadu_ps(ap.add(i));
                        let vb = _mm256_loadu_ps(bp.add(i));
                        let m = _mm256_movemask_ps(_mm256_cmp_ps::<$imm>(va, vb));
                        store_bool8(op.add(i), m);
                        i += 8;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), *bp.add(i));
                        i += 1;
                    }
                }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name_scalar(
                a: &[f32],
                s: f32,
                out: &mut [MaybeUninit<bool>],
            ) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    let vs = _mm256_set1_ps(s);
                    while i + 8 <= n {
                        let va = _mm256_loadu_ps(ap.add(i));
                        let m = _mm256_movemask_ps(_mm256_cmp_ps::<$imm>(va, vs));
                        store_bool8(op.add(i), m);
                        i += 8;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), s);
                        i += 1;
                    }
                }
            }
        };
    }

    impl_f32_cmp!(eq_f32, eq_f32_scalar, |x: f32, y: f32| x == y, _CMP_EQ_OQ);
    impl_f32_cmp!(ne_f32, ne_f32_scalar, |x: f32, y: f32| x != y, _CMP_NEQ_UQ);
    impl_f32_cmp!(lt_f32, lt_f32_scalar, |x: f32, y: f32| x < y, _CMP_LT_OQ);
    impl_f32_cmp!(le_f32, le_f32_scalar, |x: f32, y: f32| x <= y, _CMP_LE_OQ);
    impl_f32_cmp!(gt_f32, gt_f32_scalar, |x: f32, y: f32| x > y, _CMP_GT_OQ);
    impl_f32_cmp!(ge_f32, ge_f32_scalar, |x: f32, y: f32| x >= y, _CMP_GE_OQ);

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn isclose_f32(
        a: &[f32],
        b: &[f32],
        rtol: f32,
        atol: f32,
        equal_nan: bool,
        out: &mut [MaybeUninit<bool>],
    ) {
        let n = a.len().min(b.len()).min(out.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let op = out.as_mut_ptr() as *mut bool;
        let mut i = 0usize;
        unsafe {
            let vrtol = _mm256_set1_ps(rtol);
            let vatol = _mm256_set1_ps(atol);
            let sign_mask = _mm256_set1_ps(-0.0);
            let abs_i_mask = _mm256_set1_epi32(0x7FFF_FFFF);
            let inf_bits = _mm256_set1_epi32(0x7F80_0000);
            while i + 8 <= n {
                let va = _mm256_loadu_ps(ap.add(i));
                let vb = _mm256_loadu_ps(bp.add(i));
                let eq = _mm256_cmp_ps::<_CMP_EQ_OQ>(va, vb);
                let diff = _mm256_andnot_ps(sign_mask, _mm256_sub_ps(va, vb));
                let abs_b = _mm256_andnot_ps(sign_mask, vb);
                let tol = _mm256_fmadd_ps(vrtol, abs_b, vatol);
                let within = _mm256_cmp_ps::<_CMP_LE_OQ>(diff, tol);
                let b_abs = _mm256_and_si256(_mm256_castps_si256(vb), abs_i_mask);
                let b_fin = _mm256_castsi256_ps(_mm256_cmpgt_epi32(inf_bits, b_abs));
                let fin_within = _mm256_and_ps(b_fin, within);
                let mut res = _mm256_or_ps(eq, fin_within);
                if equal_nan {
                    let nan_ab = _mm256_and_ps(
                        _mm256_cmp_ps::<_CMP_UNORD_Q>(va, va),
                        _mm256_cmp_ps::<_CMP_UNORD_Q>(vb, vb),
                    );
                    res = _mm256_or_ps(res, nan_ab);
                }
                store_bool8(op.add(i), _mm256_movemask_ps(res));
                i += 8;
            }
            while i < n {
                *op.add(i) = crate::scalar_ops::isclose_f32_elem(
                    *ap.add(i),
                    *bp.add(i),
                    rtol,
                    atol,
                    equal_nan,
                );
                i += 1;
            }
        }
    }

    // ------------------------------------------------------------------------
    // i32 kernels (8 lanes per __m256i)
    // ------------------------------------------------------------------------

    macro_rules! impl_i32_bin {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[i32],
                b: &[i32],
                out: &mut [MaybeUninit<i32>],
            ) {
                let n = a.len().min(b.len()).min(out.len());
                let ap = a.as_ptr() as *const __m256i;
                let bp = b.as_ptr() as *const __m256i;
                let op = out.as_mut_ptr() as *mut __m256i;
                let a_s = a.as_ptr();
                let b_s = b.as_ptr();
                let o_s = out.as_mut_ptr() as *mut i32;
                let mut i = 0usize;
                unsafe {
                    while i + 8 <= n {
                        let va = _mm256_loadu_si256(ap.add(i / 8));
                        let vb = _mm256_loadu_si256(bp.add(i / 8));
                        _mm256_storeu_si256(op.add(i / 8), ($vec_op)(va, vb));
                        i += 8;
                    }
                    while i < n {
                        *o_s.add(i) = ($scalar)(*a_s.add(i), *b_s.add(i));
                        i += 1;
                    }
                }
            }
        };
    }

    impl_i32_bin!(add_i32, i32::wrapping_add, |a, b| _mm256_add_epi32(a, b));
    impl_i32_bin!(sub_i32, i32::wrapping_sub, |a, b| _mm256_sub_epi32(a, b));
    impl_i32_bin!(mul_i32, i32::wrapping_mul, |a, b| _mm256_mullo_epi32(a, b));
    impl_i32_bin!(bitand_i32, |x: i32, y: i32| x & y, |a, b| _mm256_and_si256(a, b));
    impl_i32_bin!(bitor_i32, |x: i32, y: i32| x | y, |a, b| _mm256_or_si256(a, b));
    impl_i32_bin!(bitxor_i32, |x: i32, y: i32| x ^ y, |a, b| _mm256_xor_si256(a, b));
    impl_i32_bin!(maximum_i32, i32::max, |a, b| _mm256_max_epi32(a, b));
    impl_i32_bin!(minimum_i32, i32::min, |a, b| _mm256_min_epi32(a, b));
    impl_i32_bin!(lshift_i32_vec, lshift_i32, |a, b| _mm256_sllv_epi32(a, b));
    impl_i32_bin!(rshift_i32_vec, rshift_i32, |a, b| _mm256_srav_epi32(a, b));

    macro_rules! impl_i32_scalar {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[i32],
                s: i32,
                out: &mut [MaybeUninit<i32>],
            ) {
                let n = a.len().min(out.len());
                let a_s = a.as_ptr();
                let o_s = out.as_mut_ptr() as *mut i32;
                let mut i = 0usize;
                unsafe {
                    let vs = _mm256_set1_epi32(s);
                    while i + 8 <= n {
                        let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                        _mm256_storeu_si256(o_s.add(i) as *mut __m256i, ($vec_op)(va, vs));
                        i += 8;
                    }
                    while i < n {
                        *o_s.add(i) = ($scalar)(*a_s.add(i), s);
                        i += 1;
                    }
                }
            }
        };
    }

    impl_i32_scalar!(add_i32_scalar, i32::wrapping_add, |a, b| _mm256_add_epi32(a, b));
    impl_i32_scalar!(sub_i32_scalar, i32::wrapping_sub, |a, b| _mm256_sub_epi32(a, b));
    impl_i32_scalar!(rsub_i32_scalar, |x: i32, s: i32| s.wrapping_sub(x), |a, b| _mm256_sub_epi32(b, a));
    impl_i32_scalar!(mul_i32_scalar, i32::wrapping_mul, |a, b| _mm256_mullo_epi32(a, b));

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn neg_i32(a: &[i32], out: &mut [MaybeUninit<i32>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i32;
        let mut i = 0usize;
        unsafe {
            let zero = _mm256_setzero_si256();
            while i + 8 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, _mm256_sub_epi32(zero, va));
                i += 8;
            }
            while i < n {
                *o_s.add(i) = (*a_s.add(i)).wrapping_neg();
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn abs_i32(a: &[i32], out: &mut [MaybeUninit<i32>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i32;
        let mut i = 0usize;
        unsafe {
            while i + 8 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, _mm256_abs_epi32(va));
                i += 8;
            }
            while i < n {
                *o_s.add(i) = (*a_s.add(i)).wrapping_abs();
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn square_i32(a: &[i32], out: &mut [MaybeUninit<i32>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i32;
        let mut i = 0usize;
        unsafe {
            while i + 8 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, _mm256_mullo_epi32(va, va));
                i += 8;
            }
            while i < n {
                let x = *a_s.add(i);
                *o_s.add(i) = x.wrapping_mul(x);
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn invert_i32(a: &[i32], out: &mut [MaybeUninit<i32>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i32;
        let mut i = 0usize;
        unsafe {
            let ones = _mm256_set1_epi32(-1);
            while i + 8 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, _mm256_xor_si256(va, ones));
                i += 8;
            }
            while i < n {
                *o_s.add(i) = !*a_s.add(i);
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn logical_not_i32(a: &[i32], out: &mut [MaybeUninit<bool>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut bool;
        let mut i = 0usize;
        unsafe {
            let zero = _mm256_setzero_si256();
            while i + 8 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                let cmp = _mm256_cmpeq_epi32(va, zero);
                store_bool8(o_s.add(i), _mm256_movemask_ps(_mm256_castsi256_ps(cmp)));
                i += 8;
            }
            while i < n {
                *o_s.add(i) = *a_s.add(i) == 0;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn clip_scalar_i32(
        a: &[i32],
        lo: i32,
        hi: i32,
        out: &mut [MaybeUninit<i32>],
    ) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i32;
        let mut i = 0usize;
        unsafe {
            let vlo = _mm256_set1_epi32(lo);
            let vhi = _mm256_set1_epi32(hi);
            while i + 8 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                let r = _mm256_min_epi32(_mm256_max_epi32(va, vlo), vhi);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, r);
                i += 8;
            }
            while i < n {
                *o_s.add(i) = crate::scalar_ops::clip_ord(*a_s.add(i), lo, hi);
                i += 1;
            }
        }
    }

    macro_rules! impl_i32_cmp {
        ($name:ident, $name_scalar:ident, $scalar:expr, $vec_cmp:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[i32],
                b: &[i32],
                out: &mut [MaybeUninit<bool>],
            ) {
                let n = a.len().min(b.len()).min(out.len());
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    let ones = _mm256_set1_epi32(-1);
                    while i + 8 <= n {
                        let va = _mm256_loadu_si256(ap.add(i) as *const __m256i);
                        let vb = _mm256_loadu_si256(bp.add(i) as *const __m256i);
                        let cmp = ($vec_cmp)(va, vb, ones);
                        store_bool8(op.add(i), _mm256_movemask_ps(_mm256_castsi256_ps(cmp)));
                        i += 8;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), *bp.add(i));
                        i += 1;
                    }
                }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name_scalar(
                a: &[i32],
                s: i32,
                out: &mut [MaybeUninit<bool>],
            ) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    let vs = _mm256_set1_epi32(s);
                    let ones = _mm256_set1_epi32(-1);
                    while i + 8 <= n {
                        let va = _mm256_loadu_si256(ap.add(i) as *const __m256i);
                        let cmp = ($vec_cmp)(va, vs, ones);
                        store_bool8(op.add(i), _mm256_movemask_ps(_mm256_castsi256_ps(cmp)));
                        i += 8;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), s);
                        i += 1;
                    }
                }
            }
        };
    }

    impl_i32_cmp!(eq_i32, eq_i32_scalar, |x: i32, y: i32| x == y, |a, b, _| _mm256_cmpeq_epi32(a, b));
    impl_i32_cmp!(ne_i32, ne_i32_scalar, |x: i32, y: i32| x != y, |a, b, ones| {
        _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), ones)
    });
    impl_i32_cmp!(gt_i32, gt_i32_scalar, |x: i32, y: i32| x > y, |a, b, _| _mm256_cmpgt_epi32(a, b));
    impl_i32_cmp!(lt_i32, lt_i32_scalar, |x: i32, y: i32| x < y, |a, b, _| _mm256_cmpgt_epi32(b, a));
    impl_i32_cmp!(ge_i32, ge_i32_scalar, |x: i32, y: i32| x >= y, |a, b, ones| {
        _mm256_xor_si256(_mm256_cmpgt_epi32(b, a), ones)
    });
    impl_i32_cmp!(le_i32, le_i32_scalar, |x: i32, y: i32| x <= y, |a, b, ones| {
        _mm256_xor_si256(_mm256_cmpgt_epi32(a, b), ones)
    });

    // ------------------------------------------------------------------------
    // i64 kernels (4 lanes per __m256i)
    // ------------------------------------------------------------------------

    #[inline(always)]
    unsafe fn vec_mul_i64(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let lo_lo = _mm256_mul_epu32(a, b);
            let a_hi = _mm256_srli_epi64::<32>(a);
            let b_hi = _mm256_srli_epi64::<32>(b);
            let cross = _mm256_add_epi64(_mm256_mul_epu32(a_hi, b), _mm256_mul_epu32(a, b_hi));
            _mm256_add_epi64(lo_lo, _mm256_slli_epi64::<32>(cross))
        }
    }

    macro_rules! impl_i64_bin {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[i64],
                b: &[i64],
                out: &mut [MaybeUninit<i64>],
            ) {
                let n = a.len().min(b.len()).min(out.len());
                let a_s = a.as_ptr();
                let b_s = b.as_ptr();
                let o_s = out.as_mut_ptr() as *mut i64;
                let mut i = 0usize;
                unsafe {
                    while i + 4 <= n {
                        let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                        let vb = _mm256_loadu_si256(b_s.add(i) as *const __m256i);
                        _mm256_storeu_si256(o_s.add(i) as *mut __m256i, ($vec_op)(va, vb));
                        i += 4;
                    }
                    while i < n {
                        *o_s.add(i) = ($scalar)(*a_s.add(i), *b_s.add(i));
                        i += 1;
                    }
                }
            }
        };
    }

    impl_i64_bin!(add_i64, i64::wrapping_add, |a, b| _mm256_add_epi64(a, b));
    impl_i64_bin!(sub_i64, i64::wrapping_sub, |a, b| _mm256_sub_epi64(a, b));
    impl_i64_bin!(mul_i64, i64::wrapping_mul, |a, b| vec_mul_i64(a, b));
    impl_i64_bin!(bitand_i64, |x: i64, y: i64| x & y, |a, b| _mm256_and_si256(a, b));
    impl_i64_bin!(bitor_i64, |x: i64, y: i64| x | y, |a, b| _mm256_or_si256(a, b));
    impl_i64_bin!(bitxor_i64, |x: i64, y: i64| x ^ y, |a, b| _mm256_xor_si256(a, b));
    impl_i64_bin!(maximum_i64, i64::max, |a, b| {
        let gt = _mm256_castsi256_pd(_mm256_cmpgt_epi64(a, b));
        _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(b),
            _mm256_castsi256_pd(a),
            gt,
        ))
    });
    impl_i64_bin!(minimum_i64, i64::min, |a, b| {
        let gt = _mm256_castsi256_pd(_mm256_cmpgt_epi64(a, b));
        _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(a),
            _mm256_castsi256_pd(b),
            gt,
        ))
    });
    impl_i64_bin!(lshift_i64_vec, lshift_i64, |a, b| _mm256_sllv_epi64(a, b));

    macro_rules! impl_i64_scalar {
        ($name:ident, $scalar:expr, $vec_op:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[i64],
                s: i64,
                out: &mut [MaybeUninit<i64>],
            ) {
                let n = a.len().min(out.len());
                let a_s = a.as_ptr();
                let o_s = out.as_mut_ptr() as *mut i64;
                let mut i = 0usize;
                unsafe {
                    let vs = _mm256_set1_epi64x(s);
                    while i + 4 <= n {
                        let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                        _mm256_storeu_si256(o_s.add(i) as *mut __m256i, ($vec_op)(va, vs));
                        i += 4;
                    }
                    while i < n {
                        *o_s.add(i) = ($scalar)(*a_s.add(i), s);
                        i += 1;
                    }
                }
            }
        };
    }

    impl_i64_scalar!(add_i64_scalar, i64::wrapping_add, |a, b| _mm256_add_epi64(a, b));
    impl_i64_scalar!(sub_i64_scalar, i64::wrapping_sub, |a, b| _mm256_sub_epi64(a, b));
    impl_i64_scalar!(rsub_i64_scalar, |x: i64, s: i64| s.wrapping_sub(x), |a, b| _mm256_sub_epi64(b, a));
    impl_i64_scalar!(mul_i64_scalar, i64::wrapping_mul, |a, b| vec_mul_i64(a, b));

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn neg_i64(a: &[i64], out: &mut [MaybeUninit<i64>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i64;
        let mut i = 0usize;
        unsafe {
            let zero = _mm256_setzero_si256();
            while i + 4 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, _mm256_sub_epi64(zero, va));
                i += 4;
            }
            while i < n {
                *o_s.add(i) = (*a_s.add(i)).wrapping_neg();
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn abs_i64(a: &[i64], out: &mut [MaybeUninit<i64>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i64;
        let mut i = 0usize;
        unsafe {
            let zero = _mm256_setzero_si256();
            while i + 4 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                let mask = _mm256_cmpgt_epi64(zero, va);
                let res = _mm256_sub_epi64(_mm256_xor_si256(va, mask), mask);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, res);
                i += 4;
            }
            while i < n {
                *o_s.add(i) = (*a_s.add(i)).wrapping_abs();
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn square_i64(a: &[i64], out: &mut [MaybeUninit<i64>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i64;
        let mut i = 0usize;
        unsafe {
            while i + 4 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, vec_mul_i64(va, va));
                i += 4;
            }
            while i < n {
                let x = *a_s.add(i);
                *o_s.add(i) = x.wrapping_mul(x);
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn invert_i64(a: &[i64], out: &mut [MaybeUninit<i64>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i64;
        let mut i = 0usize;
        unsafe {
            let ones = _mm256_set1_epi64x(-1);
            while i + 4 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, _mm256_xor_si256(va, ones));
                i += 4;
            }
            while i < n {
                *o_s.add(i) = !*a_s.add(i);
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn logical_not_i64(a: &[i64], out: &mut [MaybeUninit<bool>]) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut bool;
        let mut i = 0usize;
        unsafe {
            let zero = _mm256_setzero_si256();
            while i + 4 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                let cmp = _mm256_cmpeq_epi64(va, zero);
                store_bool4(o_s.add(i), _mm256_movemask_pd(_mm256_castsi256_pd(cmp)));
                i += 4;
            }
            while i < n {
                *o_s.add(i) = *a_s.add(i) == 0;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn clip_scalar_i64(
        a: &[i64],
        lo: i64,
        hi: i64,
        out: &mut [MaybeUninit<i64>],
    ) {
        let n = a.len().min(out.len());
        let a_s = a.as_ptr();
        let o_s = out.as_mut_ptr() as *mut i64;
        let mut i = 0usize;
        unsafe {
            let vlo = _mm256_set1_epi64x(lo);
            let vhi = _mm256_set1_epi64x(hi);
            while i + 4 <= n {
                let va = _mm256_loadu_si256(a_s.add(i) as *const __m256i);
                let gt_lo = _mm256_castsi256_pd(_mm256_cmpgt_epi64(va, vlo));
                let cl_lo = _mm256_castpd_si256(_mm256_blendv_pd(
                    _mm256_castsi256_pd(vlo),
                    _mm256_castsi256_pd(va),
                    gt_lo,
                ));
                let gt_hi = _mm256_castsi256_pd(_mm256_cmpgt_epi64(cl_lo, vhi));
                let cl = _mm256_castpd_si256(_mm256_blendv_pd(
                    _mm256_castsi256_pd(cl_lo),
                    _mm256_castsi256_pd(vhi),
                    gt_hi,
                ));
                _mm256_storeu_si256(o_s.add(i) as *mut __m256i, cl);
                i += 4;
            }
            while i < n {
                *o_s.add(i) = crate::scalar_ops::clip_ord(*a_s.add(i), lo, hi);
                i += 1;
            }
        }
    }

    macro_rules! impl_i64_cmp {
        ($name:ident, $name_scalar:ident, $scalar:expr, $vec_cmp:expr) => {
            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name(
                a: &[i64],
                b: &[i64],
                out: &mut [MaybeUninit<bool>],
            ) {
                let n = a.len().min(b.len()).min(out.len());
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    let ones = _mm256_set1_epi64x(-1);
                    while i + 4 <= n {
                        let va = _mm256_loadu_si256(ap.add(i) as *const __m256i);
                        let vb = _mm256_loadu_si256(bp.add(i) as *const __m256i);
                        let cmp = ($vec_cmp)(va, vb, ones);
                        store_bool4(op.add(i), _mm256_movemask_pd(_mm256_castsi256_pd(cmp)));
                        i += 4;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), *bp.add(i));
                        i += 1;
                    }
                }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) unsafe fn $name_scalar(
                a: &[i64],
                s: i64,
                out: &mut [MaybeUninit<bool>],
            ) {
                let n = a.len().min(out.len());
                let ap = a.as_ptr();
                let op = out.as_mut_ptr() as *mut bool;
                let mut i = 0usize;
                unsafe {
                    let vs = _mm256_set1_epi64x(s);
                    let ones = _mm256_set1_epi64x(-1);
                    while i + 4 <= n {
                        let va = _mm256_loadu_si256(ap.add(i) as *const __m256i);
                        let cmp = ($vec_cmp)(va, vs, ones);
                        store_bool4(op.add(i), _mm256_movemask_pd(_mm256_castsi256_pd(cmp)));
                        i += 4;
                    }
                    while i < n {
                        *op.add(i) = ($scalar)(*ap.add(i), s);
                        i += 1;
                    }
                }
            }
        };
    }

    impl_i64_cmp!(eq_i64, eq_i64_scalar, |x: i64, y: i64| x == y, |a, b, _| _mm256_cmpeq_epi64(a, b));
    impl_i64_cmp!(ne_i64, ne_i64_scalar, |x: i64, y: i64| x != y, |a, b, ones| {
        _mm256_xor_si256(_mm256_cmpeq_epi64(a, b), ones)
    });
    impl_i64_cmp!(gt_i64, gt_i64_scalar, |x: i64, y: i64| x > y, |a, b, _| _mm256_cmpgt_epi64(a, b));
    impl_i64_cmp!(lt_i64, lt_i64_scalar, |x: i64, y: i64| x < y, |a, b, _| _mm256_cmpgt_epi64(b, a));
    impl_i64_cmp!(ge_i64, ge_i64_scalar, |x: i64, y: i64| x >= y, |a, b, ones| {
        _mm256_xor_si256(_mm256_cmpgt_epi64(b, a), ones)
    });
    impl_i64_cmp!(le_i64, le_i64_scalar, |x: i64, y: i64| x <= y, |a, b, ones| {
        _mm256_xor_si256(_mm256_cmpgt_epi64(a, b), ones)
    });

    // ------------------------------------------------------------------------
    // Contiguous SIMD reductions
    // ------------------------------------------------------------------------

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_sum_f64(a: &[f64]) -> f64 {
        let n = a.len();
        let ap = a.as_ptr();
        let mut i = 0usize;
        unsafe {
            let mut acc0 = _mm256_setzero_pd();
            let mut acc1 = _mm256_setzero_pd();
            while i + 8 <= n {
                acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(ap.add(i)));
                acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(ap.add(i + 4)));
                i += 8;
            }
            acc0 = _mm256_add_pd(acc0, acc1);
            if i + 4 <= n {
                acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(ap.add(i)));
                i += 4;
            }
            let mut lanes = [0.0f64; 4];
            _mm256_storeu_pd(lanes.as_mut_ptr(), acc0);
            let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            while i < n {
                sum += *ap.add(i);
                i += 1;
            }
            sum
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_prod_f64(a: &[f64]) -> f64 {
        let n = a.len();
        let ap = a.as_ptr();
        let mut i = 0usize;
        unsafe {
            let mut acc0 = _mm256_set1_pd(1.0);
            let mut acc1 = _mm256_set1_pd(1.0);
            while i + 8 <= n {
                acc0 = _mm256_mul_pd(acc0, _mm256_loadu_pd(ap.add(i)));
                acc1 = _mm256_mul_pd(acc1, _mm256_loadu_pd(ap.add(i + 4)));
                i += 8;
            }
            acc0 = _mm256_mul_pd(acc0, acc1);
            if i + 4 <= n {
                acc0 = _mm256_mul_pd(acc0, _mm256_loadu_pd(ap.add(i)));
                i += 4;
            }
            let mut lanes = [1.0f64; 4];
            _mm256_storeu_pd(lanes.as_mut_ptr(), acc0);
            let mut prod = (lanes[0] * lanes[1]) * (lanes[2] * lanes[3]);
            while i < n {
                prod *= *ap.add(i);
                i += 1;
            }
            prod
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_max_f64(a: &[f64]) -> f64 {
        let n = a.len();
        let ap = a.as_ptr();
        if n < 4 {
            let mut m = a[0];
            for &x in &a[1..] {
                m = max_prop_nan_f64(m, x);
            }
            return m;
        }
        unsafe {
            let mut acc = _mm256_loadu_pd(ap);
            let mut i = 4usize;
            while i + 4 <= n {
                acc = vec_max_prop_nan_f64(acc, _mm256_loadu_pd(ap.add(i)));
                i += 4;
            }
            let mut lanes = [0.0f64; 4];
            _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
            let mut m = max_prop_nan_f64(
                max_prop_nan_f64(lanes[0], lanes[1]),
                max_prop_nan_f64(lanes[2], lanes[3]),
            );
            while i < n {
                m = max_prop_nan_f64(m, *ap.add(i));
                i += 1;
            }
            m
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_min_f64(a: &[f64]) -> f64 {
        let n = a.len();
        let ap = a.as_ptr();
        if n < 4 {
            let mut m = a[0];
            for &x in &a[1..] {
                m = min_prop_nan_f64(m, x);
            }
            return m;
        }
        unsafe {
            let mut acc = _mm256_loadu_pd(ap);
            let mut i = 4usize;
            while i + 4 <= n {
                acc = vec_min_prop_nan_f64(acc, _mm256_loadu_pd(ap.add(i)));
                i += 4;
            }
            let mut lanes = [0.0f64; 4];
            _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
            let mut m = min_prop_nan_f64(
                min_prop_nan_f64(lanes[0], lanes[1]),
                min_prop_nan_f64(lanes[2], lanes[3]),
            );
            while i < n {
                m = min_prop_nan_f64(m, *ap.add(i));
                i += 1;
            }
            m
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_sum_f32(a: &[f32]) -> f32 {
        let n = a.len();
        let ap = a.as_ptr();
        let mut i = 0usize;
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            while i + 16 <= n {
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(ap.add(i)));
                acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(ap.add(i + 8)));
                i += 16;
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            if i + 8 <= n {
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(ap.add(i)));
                i += 8;
            }
            let mut lanes = [0.0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
            let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
                + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
            while i < n {
                sum += *ap.add(i);
                i += 1;
            }
            sum
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_prod_f32(a: &[f32]) -> f32 {
        let n = a.len();
        let ap = a.as_ptr();
        let mut i = 0usize;
        unsafe {
            let mut acc0 = _mm256_set1_ps(1.0);
            let mut acc1 = _mm256_set1_ps(1.0);
            while i + 16 <= n {
                acc0 = _mm256_mul_ps(acc0, _mm256_loadu_ps(ap.add(i)));
                acc1 = _mm256_mul_ps(acc1, _mm256_loadu_ps(ap.add(i + 8)));
                i += 16;
            }
            acc0 = _mm256_mul_ps(acc0, acc1);
            if i + 8 <= n {
                acc0 = _mm256_mul_ps(acc0, _mm256_loadu_ps(ap.add(i)));
                i += 8;
            }
            let mut lanes = [1.0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
            let mut prod = ((lanes[0] * lanes[1]) * (lanes[2] * lanes[3]))
                * ((lanes[4] * lanes[5]) * (lanes[6] * lanes[7]));
            while i < n {
                prod *= *ap.add(i);
                i += 1;
            }
            prod
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_max_f32(a: &[f32]) -> f32 {
        let n = a.len();
        let ap = a.as_ptr();
        if n < 8 {
            let mut m = a[0];
            for &x in &a[1..] {
                m = max_prop_nan_f32(m, x);
            }
            return m;
        }
        unsafe {
            let mut acc = _mm256_loadu_ps(ap);
            let mut i = 8usize;
            while i + 8 <= n {
                acc = vec_max_prop_nan_f32(acc, _mm256_loadu_ps(ap.add(i)));
                i += 8;
            }
            let mut lanes = [0.0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
            let mut m = lanes[0];
            for &v in &lanes[1..] {
                m = max_prop_nan_f32(m, v);
            }
            while i < n {
                m = max_prop_nan_f32(m, *ap.add(i));
                i += 1;
            }
            m
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_min_f32(a: &[f32]) -> f32 {
        let n = a.len();
        let ap = a.as_ptr();
        if n < 8 {
            let mut m = a[0];
            for &x in &a[1..] {
                m = min_prop_nan_f32(m, x);
            }
            return m;
        }
        unsafe {
            let mut acc = _mm256_loadu_ps(ap);
            let mut i = 8usize;
            while i + 8 <= n {
                acc = vec_min_prop_nan_f32(acc, _mm256_loadu_ps(ap.add(i)));
                i += 8;
            }
            let mut lanes = [0.0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
            let mut m = lanes[0];
            for &v in &lanes[1..] {
                m = min_prop_nan_f32(m, v);
            }
            while i < n {
                m = min_prop_nan_f32(m, *ap.add(i));
                i += 1;
            }
            m
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_sum_i32(a: &[i32]) -> i32 {
        let n = a.len();
        let ap = a.as_ptr();
        let mut i = 0usize;
        unsafe {
            let mut acc0 = _mm256_setzero_si256();
            let mut acc1 = _mm256_setzero_si256();
            while i + 16 <= n {
                acc0 = _mm256_add_epi32(acc0, _mm256_loadu_si256(ap.add(i) as *const __m256i));
                acc1 = _mm256_add_epi32(acc1, _mm256_loadu_si256(ap.add(i + 8) as *const __m256i));
                i += 16;
            }
            acc0 = _mm256_add_epi32(acc0, acc1);
            if i + 8 <= n {
                acc0 = _mm256_add_epi32(acc0, _mm256_loadu_si256(ap.add(i) as *const __m256i));
                i += 8;
            }
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc0);
            let mut sum = 0i32;
            for &v in &lanes {
                sum = sum.wrapping_add(v);
            }
            while i < n {
                sum = sum.wrapping_add(*ap.add(i));
                i += 1;
            }
            sum
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_sum_i64(a: &[i64]) -> i64 {
        let n = a.len();
        let ap = a.as_ptr();
        let mut i = 0usize;
        unsafe {
            let mut acc0 = _mm256_setzero_si256();
            let mut acc1 = _mm256_setzero_si256();
            while i + 8 <= n {
                acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(ap.add(i) as *const __m256i));
                acc1 = _mm256_add_epi64(acc1, _mm256_loadu_si256(ap.add(i + 4) as *const __m256i));
                i += 8;
            }
            acc0 = _mm256_add_epi64(acc0, acc1);
            if i + 4 <= n {
                acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(ap.add(i) as *const __m256i));
                i += 4;
            }
            let mut lanes = [0i64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc0);
            let mut sum = 0i64;
            for &v in &lanes {
                sum = sum.wrapping_add(v);
            }
            while i < n {
                sum = sum.wrapping_add(*ap.add(i));
                i += 1;
            }
            sum
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_max_i32(a: &[i32]) -> i32 {
        let n = a.len();
        let ap = a.as_ptr();
        if n < 8 {
            return a[1..].iter().copied().fold(a[0], i32::max);
        }
        unsafe {
            let mut acc = _mm256_loadu_si256(ap as *const __m256i);
            let mut i = 8usize;
            while i + 8 <= n {
                acc = _mm256_max_epi32(acc, _mm256_loadu_si256(ap.add(i) as *const __m256i));
                i += 8;
            }
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
            let mut m = lanes[1..].iter().copied().fold(lanes[0], i32::max);
            while i < n {
                m = m.max(*ap.add(i));
                i += 1;
            }
            m
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_min_i32(a: &[i32]) -> i32 {
        let n = a.len();
        let ap = a.as_ptr();
        if n < 8 {
            return a[1..].iter().copied().fold(a[0], i32::min);
        }
        unsafe {
            let mut acc = _mm256_loadu_si256(ap as *const __m256i);
            let mut i = 8usize;
            while i + 8 <= n {
                acc = _mm256_min_epi32(acc, _mm256_loadu_si256(ap.add(i) as *const __m256i));
                i += 8;
            }
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
            let mut m = lanes[1..].iter().copied().fold(lanes[0], i32::min);
            while i < n {
                m = m.min(*ap.add(i));
                i += 1;
            }
            m
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_max_i64(a: &[i64]) -> i64 {
        let n = a.len();
        let ap = a.as_ptr();
        if n < 4 {
            return a[1..].iter().copied().fold(a[0], i64::max);
        }
        unsafe {
            let mut acc = _mm256_loadu_si256(ap as *const __m256i);
            let mut i = 4usize;
            while i + 4 <= n {
                let v = _mm256_loadu_si256(ap.add(i) as *const __m256i);
                let gt = _mm256_castsi256_pd(_mm256_cmpgt_epi64(v, acc));
                acc = _mm256_castpd_si256(_mm256_blendv_pd(
                    _mm256_castsi256_pd(acc),
                    _mm256_castsi256_pd(v),
                    gt,
                ));
                i += 4;
            }
            let mut lanes = [0i64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
            let mut m = lanes[1..].iter().copied().fold(lanes[0], i64::max);
            while i < n {
                m = m.max(*ap.add(i));
                i += 1;
            }
            m
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn reduce_min_i64(a: &[i64]) -> i64 {
        let n = a.len();
        let ap = a.as_ptr();
        if n < 4 {
            return a[1..].iter().copied().fold(a[0], i64::min);
        }
        unsafe {
            let mut acc = _mm256_loadu_si256(ap as *const __m256i);
            let mut i = 4usize;
            while i + 4 <= n {
                let v = _mm256_loadu_si256(ap.add(i) as *const __m256i);
                let gt = _mm256_castsi256_pd(_mm256_cmpgt_epi64(acc, v));
                acc = _mm256_castpd_si256(_mm256_blendv_pd(
                    _mm256_castsi256_pd(acc),
                    _mm256_castsi256_pd(v),
                    gt,
                ));
                i += 4;
            }
            let mut lanes = [0i64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
            let mut m = lanes[1..].iter().copied().fold(lanes[0], i64::min);
            while i < n {
                m = m.min(*ap.add(i));
                i += 1;
            }
            m
        }
    }
}

// ============================================================================
// Runtime-dispatched public(crate) wrappers
// ============================================================================

macro_rules! dispatch_bin {
    ($name:ident, $avx:ident, $t:ty, $fallback:expr) => {
        #[inline]
        pub(crate) fn $name(a: &[$t], b: &[$t], out: &mut [MaybeUninit<$t>]) {
            #[cfg(target_arch = "x86_64")]
            if is_avx2_fma_active() {
                unsafe { avx2::$avx(a, b, out) };
                return;
            }
            let n = a.len().min(b.len()).min(out.len());
            let f = $fallback;
            for i in 0..n {
                out[i].write(f(a[i], b[i]));
            }
        }
    };
}

macro_rules! dispatch_scalar {
    ($name:ident, $avx:ident, $t:ty, $fallback:expr) => {
        #[inline]
        pub(crate) fn $name(a: &[$t], s: $t, out: &mut [MaybeUninit<$t>]) {
            #[cfg(target_arch = "x86_64")]
            if is_avx2_fma_active() {
                unsafe { avx2::$avx(a, s, out) };
                return;
            }
            let n = a.len().min(out.len());
            let f = $fallback;
            for i in 0..n {
                out[i].write(f(a[i], s));
            }
        }
    };
}

macro_rules! dispatch_unary {
    ($name:ident, $avx:ident, $src:ty, $dst:ty, $fallback:expr) => {
        #[inline]
        pub(crate) fn $name(a: &[$src], out: &mut [MaybeUninit<$dst>]) {
            #[cfg(target_arch = "x86_64")]
            if is_avx2_fma_active() {
                unsafe { avx2::$avx(a, out) };
                return;
            }
            let n = a.len().min(out.len());
            let f = $fallback;
            for i in 0..n {
                out[i].write(f(a[i]));
            }
        }
    };
}

macro_rules! dispatch_cmp {
    ($name:ident, $name_scalar:ident, $avx:ident, $avx_scalar:ident, $t:ty, $fallback:expr) => {
        #[inline]
        pub(crate) fn $name(a: &[$t], b: &[$t], out: &mut [MaybeUninit<bool>]) {
            #[cfg(target_arch = "x86_64")]
            if is_avx2_fma_active() {
                unsafe { avx2::$avx(a, b, out) };
                return;
            }
            let n = a.len().min(b.len()).min(out.len());
            let f = $fallback;
            for i in 0..n {
                out[i].write(f(a[i], b[i]));
            }
        }

        #[inline]
        pub(crate) fn $name_scalar(a: &[$t], s: $t, out: &mut [MaybeUninit<bool>]) {
            #[cfg(target_arch = "x86_64")]
            if is_avx2_fma_active() {
                unsafe { avx2::$avx_scalar(a, s, out) };
                return;
            }
            let n = a.len().min(out.len());
            let f = $fallback;
            for i in 0..n {
                out[i].write(f(a[i], s));
            }
        }
    };
}

// f64
dispatch_bin!(add_f64, add_f64, f64, |a: f64, b: f64| a + b);
dispatch_bin!(sub_f64, sub_f64, f64, |a: f64, b: f64| a - b);
dispatch_bin!(mul_f64, mul_f64, f64, |a: f64, b: f64| a * b);
dispatch_bin!(div_f64, div_f64, f64, |a: f64, b: f64| a / b);
dispatch_bin!(maximum_f64, maximum_f64, f64, crate::scalar_ops::max_prop_nan_f64);
dispatch_bin!(minimum_f64, minimum_f64, f64, crate::scalar_ops::min_prop_nan_f64);
dispatch_bin!(fmax_f64, fmax_f64_vec, f64, crate::scalar_ops::fmax_f64);
dispatch_bin!(fmin_f64, fmin_f64_vec, f64, crate::scalar_ops::fmin_f64);

dispatch_scalar!(add_f64_scalar, add_f64_scalar, f64, |a: f64, s: f64| a + s);
dispatch_scalar!(sub_f64_scalar, sub_f64_scalar, f64, |a: f64, s: f64| a - s);
dispatch_scalar!(rsub_f64_scalar, rsub_f64_scalar, f64, |a: f64, s: f64| s - a);
dispatch_scalar!(mul_f64_scalar, mul_f64_scalar, f64, |a: f64, s: f64| a * s);
dispatch_scalar!(div_f64_scalar, div_f64_scalar, f64, |a: f64, s: f64| a / s);
dispatch_scalar!(rdiv_f64_scalar, rdiv_f64_scalar, f64, |a: f64, s: f64| s / a);
dispatch_scalar!(maximum_f64_scalar, maximum_f64_scalar, f64, crate::scalar_ops::max_prop_nan_f64);
dispatch_scalar!(minimum_f64_scalar, minimum_f64_scalar, f64, crate::scalar_ops::min_prop_nan_f64);
dispatch_scalar!(rmaximum_f64_scalar, rmaximum_f64_scalar, f64, |x, s| crate::scalar_ops::max_prop_nan_f64(s, x));
dispatch_scalar!(rminimum_f64_scalar, rminimum_f64_scalar, f64, |x, s| crate::scalar_ops::min_prop_nan_f64(s, x));
dispatch_scalar!(fmax_f64_scalar, fmax_f64_scalar, f64, crate::scalar_ops::fmax_f64);
dispatch_scalar!(fmin_f64_scalar, fmin_f64_scalar, f64, crate::scalar_ops::fmin_f64);
dispatch_scalar!(rfmax_f64_scalar, rfmax_f64_scalar, f64, |x, s| crate::scalar_ops::fmax_f64(s, x));
dispatch_scalar!(rfmin_f64_scalar, rfmin_f64_scalar, f64, |x, s| crate::scalar_ops::fmin_f64(s, x));

dispatch_unary!(neg_f64, neg_f64, f64, f64, |x: f64| -x);
dispatch_unary!(abs_f64, abs_f64, f64, f64, f64::abs);
dispatch_unary!(square_f64, square_f64, f64, f64, |x: f64| x * x);
dispatch_unary!(sqrt_f64, sqrt_f64, f64, f64, f64::sqrt);
dispatch_unary!(recip_f64, recip_f64, f64, f64, |x: f64| 1.0 / x);
dispatch_unary!(floor_f64, floor_f64, f64, f64, f64::floor);
dispatch_unary!(ceil_f64, ceil_f64, f64, f64, f64::ceil);
dispatch_unary!(trunc_f64, trunc_f64, f64, f64, f64::trunc);
dispatch_unary!(rint_f64, rint_f64, f64, f64, f64::round_ties_even);
dispatch_unary!(deg2rad_f64, deg2rad_f64, f64, f64, f64::to_radians);
dispatch_unary!(rad2deg_f64, rad2deg_f64, f64, f64, f64::to_degrees);

dispatch_unary!(isfinite_f64, isfinite_f64, f64, bool, f64::is_finite);
dispatch_unary!(isinf_f64, isinf_f64, f64, bool, f64::is_infinite);
dispatch_unary!(isnan_f64, isnan_f64, f64, bool, f64::is_nan);
dispatch_unary!(signbit_f64, signbit_f64, f64, bool, f64::is_sign_negative);
dispatch_unary!(logical_not_f64, logical_not_f64, f64, bool, |x: f64| x == 0.0);

dispatch_cmp!(eq_f64, eq_f64_scalar, eq_f64, eq_f64_scalar, f64, |a: f64, b: f64| a == b);
dispatch_cmp!(ne_f64, ne_f64_scalar, ne_f64, ne_f64_scalar, f64, |a: f64, b: f64| a != b);
dispatch_cmp!(lt_f64, lt_f64_scalar, lt_f64, lt_f64_scalar, f64, |a: f64, b: f64| a < b);
dispatch_cmp!(le_f64, le_f64_scalar, le_f64, le_f64_scalar, f64, |a: f64, b: f64| a <= b);
dispatch_cmp!(gt_f64, gt_f64_scalar, gt_f64, gt_f64_scalar, f64, |a: f64, b: f64| a > b);
dispatch_cmp!(ge_f64, ge_f64_scalar, ge_f64, ge_f64_scalar, f64, |a: f64, b: f64| a >= b);

#[inline]
pub(crate) fn clip_scalar_f64(a: &[f64], lo: f64, hi: f64, out: &mut [MaybeUninit<f64>]) {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        unsafe { avx2::clip_scalar_f64(a, lo, hi, out) };
        return;
    }
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i].write(crate::scalar_ops::clip_f64_elem(a[i], lo, hi));
    }
}

#[inline]
pub(crate) fn isclose_f64(
    a: &[f64],
    b: &[f64],
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    out: &mut [MaybeUninit<bool>],
) {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        unsafe { avx2::isclose_f64(a, b, rtol, atol, equal_nan, out) };
        return;
    }
    let n = a.len().min(b.len()).min(out.len());
    for i in 0..n {
        out[i].write(crate::scalar_ops::isclose_f64_elem(
            a[i], b[i], rtol, atol, equal_nan,
        ));
    }
}

// f32
dispatch_bin!(add_f32, add_f32, f32, |a: f32, b: f32| a + b);
dispatch_bin!(sub_f32, sub_f32, f32, |a: f32, b: f32| a - b);
dispatch_bin!(mul_f32, mul_f32, f32, |a: f32, b: f32| a * b);
dispatch_bin!(div_f32, div_f32, f32, |a: f32, b: f32| a / b);
dispatch_bin!(maximum_f32, maximum_f32, f32, crate::scalar_ops::max_prop_nan_f32);
dispatch_bin!(minimum_f32, minimum_f32, f32, crate::scalar_ops::min_prop_nan_f32);
dispatch_bin!(fmax_f32, fmax_f32_vec, f32, crate::scalar_ops::fmax_f32);
dispatch_bin!(fmin_f32, fmin_f32_vec, f32, crate::scalar_ops::fmin_f32);

dispatch_scalar!(add_f32_scalar, add_f32_scalar, f32, |a: f32, s: f32| a + s);
dispatch_scalar!(sub_f32_scalar, sub_f32_scalar, f32, |a: f32, s: f32| a - s);
dispatch_scalar!(rsub_f32_scalar, rsub_f32_scalar, f32, |a: f32, s: f32| s - a);
dispatch_scalar!(mul_f32_scalar, mul_f32_scalar, f32, |a: f32, s: f32| a * s);
dispatch_scalar!(div_f32_scalar, div_f32_scalar, f32, |a: f32, s: f32| a / s);
dispatch_scalar!(rdiv_f32_scalar, rdiv_f32_scalar, f32, |a: f32, s: f32| s / a);
dispatch_scalar!(maximum_f32_scalar, maximum_f32_scalar, f32, crate::scalar_ops::max_prop_nan_f32);
dispatch_scalar!(minimum_f32_scalar, minimum_f32_scalar, f32, crate::scalar_ops::min_prop_nan_f32);
dispatch_scalar!(rmaximum_f32_scalar, rmaximum_f32_scalar, f32, |x, s| crate::scalar_ops::max_prop_nan_f32(s, x));
dispatch_scalar!(rminimum_f32_scalar, rminimum_f32_scalar, f32, |x, s| crate::scalar_ops::min_prop_nan_f32(s, x));
dispatch_scalar!(fmax_f32_scalar, fmax_f32_scalar, f32, crate::scalar_ops::fmax_f32);
dispatch_scalar!(fmin_f32_scalar, fmin_f32_scalar, f32, crate::scalar_ops::fmin_f32);
dispatch_scalar!(rfmax_f32_scalar, rfmax_f32_scalar, f32, |x, s| crate::scalar_ops::fmax_f32(s, x));
dispatch_scalar!(rfmin_f32_scalar, rfmin_f32_scalar, f32, |x, s| crate::scalar_ops::fmin_f32(s, x));

dispatch_unary!(neg_f32, neg_f32, f32, f32, |x: f32| -x);
dispatch_unary!(abs_f32, abs_f32, f32, f32, f32::abs);
dispatch_unary!(square_f32, square_f32, f32, f32, |x: f32| x * x);
dispatch_unary!(sqrt_f32, sqrt_f32, f32, f32, f32::sqrt);
dispatch_unary!(recip_f32, recip_f32, f32, f32, |x: f32| 1.0_f32 / x);
dispatch_unary!(floor_f32, floor_f32, f32, f32, f32::floor);
dispatch_unary!(ceil_f32, ceil_f32, f32, f32, f32::ceil);
dispatch_unary!(trunc_f32, trunc_f32, f32, f32, f32::trunc);
dispatch_unary!(rint_f32, rint_f32, f32, f32, f32::round_ties_even);
dispatch_unary!(deg2rad_f32, deg2rad_f32, f32, f32, f32::to_radians);
dispatch_unary!(rad2deg_f32, rad2deg_f32, f32, f32, f32::to_degrees);

dispatch_unary!(isfinite_f32, isfinite_f32, f32, bool, f32::is_finite);
dispatch_unary!(isinf_f32, isinf_f32, f32, bool, f32::is_infinite);
dispatch_unary!(isnan_f32, isnan_f32, f32, bool, f32::is_nan);
dispatch_unary!(signbit_f32, signbit_f32, f32, bool, f32::is_sign_negative);
dispatch_unary!(logical_not_f32, logical_not_f32, f32, bool, |x: f32| x == 0.0);

dispatch_cmp!(eq_f32, eq_f32_scalar, eq_f32, eq_f32_scalar, f32, |a: f32, b: f32| a == b);
dispatch_cmp!(ne_f32, ne_f32_scalar, ne_f32, ne_f32_scalar, f32, |a: f32, b: f32| a != b);
dispatch_cmp!(lt_f32, lt_f32_scalar, lt_f32, lt_f32_scalar, f32, |a: f32, b: f32| a < b);
dispatch_cmp!(le_f32, le_f32_scalar, le_f32, le_f32_scalar, f32, |a: f32, b: f32| a <= b);
dispatch_cmp!(gt_f32, gt_f32_scalar, gt_f32, gt_f32_scalar, f32, |a: f32, b: f32| a > b);
dispatch_cmp!(ge_f32, ge_f32_scalar, ge_f32, ge_f32_scalar, f32, |a: f32, b: f32| a >= b);

#[inline]
pub(crate) fn clip_scalar_f32(a: &[f32], lo: f32, hi: f32, out: &mut [MaybeUninit<f32>]) {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        unsafe { avx2::clip_scalar_f32(a, lo, hi, out) };
        return;
    }
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i].write(crate::scalar_ops::clip_f32_elem(a[i], lo, hi));
    }
}

#[inline]
pub(crate) fn isclose_f32(
    a: &[f32],
    b: &[f32],
    rtol: f32,
    atol: f32,
    equal_nan: bool,
    out: &mut [MaybeUninit<bool>],
) {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        unsafe { avx2::isclose_f32(a, b, rtol, atol, equal_nan, out) };
        return;
    }
    let n = a.len().min(b.len()).min(out.len());
    for i in 0..n {
        out[i].write(crate::scalar_ops::isclose_f32_elem(
            a[i], b[i], rtol, atol, equal_nan,
        ));
    }
}

// i32
dispatch_bin!(add_i32, add_i32, i32, i32::wrapping_add);
dispatch_bin!(sub_i32, sub_i32, i32, i32::wrapping_sub);
dispatch_bin!(mul_i32, mul_i32, i32, i32::wrapping_mul);
dispatch_bin!(bitand_i32, bitand_i32, i32, |a: i32, b: i32| a & b);
dispatch_bin!(bitor_i32, bitor_i32, i32, |a: i32, b: i32| a | b);
dispatch_bin!(bitxor_i32, bitxor_i32, i32, |a: i32, b: i32| a ^ b);
dispatch_bin!(maximum_i32, maximum_i32, i32, i32::max);
dispatch_bin!(minimum_i32, minimum_i32, i32, i32::min);
dispatch_bin!(lshift_i32, lshift_i32_vec, i32, crate::scalar_ops::lshift_i32);
dispatch_bin!(rshift_i32, rshift_i32_vec, i32, crate::scalar_ops::rshift_i32);

dispatch_scalar!(add_i32_scalar, add_i32_scalar, i32, i32::wrapping_add);
dispatch_scalar!(sub_i32_scalar, sub_i32_scalar, i32, i32::wrapping_sub);
dispatch_scalar!(rsub_i32_scalar, rsub_i32_scalar, i32, |x: i32, s: i32| s.wrapping_sub(x));
dispatch_scalar!(mul_i32_scalar, mul_i32_scalar, i32, i32::wrapping_mul);

dispatch_unary!(neg_i32, neg_i32, i32, i32, i32::wrapping_neg);
dispatch_unary!(abs_i32, abs_i32, i32, i32, i32::wrapping_abs);
dispatch_unary!(square_i32, square_i32, i32, i32, |x: i32| x.wrapping_mul(x));
dispatch_unary!(invert_i32, invert_i32, i32, i32, |x: i32| !x);
dispatch_unary!(logical_not_i32, logical_not_i32, i32, bool, |x: i32| x == 0);

dispatch_cmp!(eq_i32, eq_i32_scalar, eq_i32, eq_i32_scalar, i32, |a: i32, b: i32| a == b);
dispatch_cmp!(ne_i32, ne_i32_scalar, ne_i32, ne_i32_scalar, i32, |a: i32, b: i32| a != b);
dispatch_cmp!(lt_i32, lt_i32_scalar, lt_i32, lt_i32_scalar, i32, |a: i32, b: i32| a < b);
dispatch_cmp!(le_i32, le_i32_scalar, le_i32, le_i32_scalar, i32, |a: i32, b: i32| a <= b);
dispatch_cmp!(gt_i32, gt_i32_scalar, gt_i32, gt_i32_scalar, i32, |a: i32, b: i32| a > b);
dispatch_cmp!(ge_i32, ge_i32_scalar, ge_i32, ge_i32_scalar, i32, |a: i32, b: i32| a >= b);

#[inline]
pub(crate) fn clip_scalar_i32(a: &[i32], lo: i32, hi: i32, out: &mut [MaybeUninit<i32>]) {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        unsafe { avx2::clip_scalar_i32(a, lo, hi, out) };
        return;
    }
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i].write(crate::scalar_ops::clip_ord(a[i], lo, hi));
    }
}

// i64
dispatch_bin!(add_i64, add_i64, i64, i64::wrapping_add);
dispatch_bin!(sub_i64, sub_i64, i64, i64::wrapping_sub);
dispatch_bin!(mul_i64, mul_i64, i64, i64::wrapping_mul);
dispatch_bin!(bitand_i64, bitand_i64, i64, |a: i64, b: i64| a & b);
dispatch_bin!(bitor_i64, bitor_i64, i64, |a: i64, b: i64| a | b);
dispatch_bin!(bitxor_i64, bitxor_i64, i64, |a: i64, b: i64| a ^ b);
dispatch_bin!(maximum_i64, maximum_i64, i64, i64::max);
dispatch_bin!(minimum_i64, minimum_i64, i64, i64::min);
dispatch_bin!(lshift_i64, lshift_i64_vec, i64, crate::scalar_ops::lshift_i64);

dispatch_scalar!(add_i64_scalar, add_i64_scalar, i64, i64::wrapping_add);
dispatch_scalar!(sub_i64_scalar, sub_i64_scalar, i64, i64::wrapping_sub);
dispatch_scalar!(rsub_i64_scalar, rsub_i64_scalar, i64, |x: i64, s: i64| s.wrapping_sub(x));
dispatch_scalar!(mul_i64_scalar, mul_i64_scalar, i64, i64::wrapping_mul);

dispatch_unary!(neg_i64, neg_i64, i64, i64, i64::wrapping_neg);
dispatch_unary!(abs_i64, abs_i64, i64, i64, i64::wrapping_abs);
dispatch_unary!(square_i64, square_i64, i64, i64, |x: i64| x.wrapping_mul(x));
dispatch_unary!(invert_i64, invert_i64, i64, i64, |x: i64| !x);
dispatch_unary!(logical_not_i64, logical_not_i64, i64, bool, |x: i64| x == 0);

dispatch_cmp!(eq_i64, eq_i64_scalar, eq_i64, eq_i64_scalar, i64, |a: i64, b: i64| a == b);
dispatch_cmp!(ne_i64, ne_i64_scalar, ne_i64, ne_i64_scalar, i64, |a: i64, b: i64| a != b);
dispatch_cmp!(lt_i64, lt_i64_scalar, lt_i64, lt_i64_scalar, i64, |a: i64, b: i64| a < b);
dispatch_cmp!(le_i64, le_i64_scalar, le_i64, le_i64_scalar, i64, |a: i64, b: i64| a <= b);
dispatch_cmp!(gt_i64, gt_i64_scalar, gt_i64, gt_i64_scalar, i64, |a: i64, b: i64| a > b);
dispatch_cmp!(ge_i64, ge_i64_scalar, ge_i64, ge_i64_scalar, i64, |a: i64, b: i64| a >= b);

#[inline]
pub(crate) fn clip_scalar_i64(a: &[i64], lo: i64, hi: i64, out: &mut [MaybeUninit<i64>]) {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        unsafe { avx2::clip_scalar_i64(a, lo, hi, out) };
        return;
    }
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i].write(crate::scalar_ops::clip_ord(a[i], lo, hi));
    }
}

// Reductions
#[inline]
pub(crate) fn reduce_sum_f64(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_sum_f64(a) };
    }
    a.iter().copied().sum()
}

#[inline]
pub(crate) fn reduce_prod_f64(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_prod_f64(a) };
    }
    a.iter().copied().product()
}

#[inline]
pub(crate) fn reduce_max_f64(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NEG_INFINITY;
    }
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_max_f64(a) };
    }
    a[1..].iter().copied().fold(a[0], crate::scalar_ops::max_prop_nan_f64)
}

#[inline]
pub(crate) fn reduce_min_f64(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::INFINITY;
    }
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_min_f64(a) };
    }
    a[1..].iter().copied().fold(a[0], crate::scalar_ops::min_prop_nan_f64)
}

#[inline]
pub(crate) fn reduce_sum_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_sum_f32(a) };
    }
    a.iter().copied().sum()
}

#[inline]
pub(crate) fn reduce_prod_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_prod_f32(a) };
    }
    a.iter().copied().product()
}

#[inline]
pub(crate) fn reduce_max_f32(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::NEG_INFINITY;
    }
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_max_f32(a) };
    }
    a[1..].iter().copied().fold(a[0], crate::scalar_ops::max_prop_nan_f32)
}

#[inline]
pub(crate) fn reduce_min_f32(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::INFINITY;
    }
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_min_f32(a) };
    }
    a[1..].iter().copied().fold(a[0], crate::scalar_ops::min_prop_nan_f32)
}

#[inline]
pub(crate) fn reduce_sum_i32(a: &[i32]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_sum_i32(a) };
    }
    a.iter().copied().fold(0i32, i32::wrapping_add)
}

#[inline]
pub(crate) fn reduce_sum_i64(a: &[i64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_sum_i64(a) };
    }
    a.iter().copied().fold(0i64, i64::wrapping_add)
}

#[inline]
pub(crate) fn reduce_max_i32(a: &[i32]) -> i32 {
    if a.is_empty() {
        return i32::MIN;
    }
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_max_i32(a) };
    }
    a[1..].iter().copied().fold(a[0], i32::max)
}

#[inline]
pub(crate) fn reduce_min_i32(a: &[i32]) -> i32 {
    if a.is_empty() {
        return i32::MAX;
    }
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_min_i32(a) };
    }
    a[1..].iter().copied().fold(a[0], i32::min)
}

#[inline]
pub(crate) fn reduce_max_i64(a: &[i64]) -> i64 {
    if a.is_empty() {
        return i64::MIN;
    }
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_max_i64(a) };
    }
    a[1..].iter().copied().fold(a[0], i64::max)
}

#[inline]
pub(crate) fn reduce_min_i64(a: &[i64]) -> i64 {
    if a.is_empty() {
        return i64::MAX;
    }
    #[cfg(target_arch = "x86_64")]
    if is_avx2_fma_active() {
        return unsafe { avx2::reduce_min_i64(a) };
    }
    a[1..].iter().copied().fold(a[0], i64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_max_min_i64_edge_cases() {
        assert_eq!(reduce_max_i64(&[]), i64::MIN);
        assert_eq!(reduce_min_i64(&[]), i64::MAX);
        assert_eq!(reduce_max_i64(&[i64::MIN]), i64::MIN);
        assert_eq!(reduce_min_i64(&[i64::MAX]), i64::MAX);
        for len in [1usize, 2, 3, 4, 5, 7, 8, 9, 17, 65] {
            let mut vals: Vec<i64> = (0..len).map(|i| (i as i64) * 13 - 200).collect();
            vals[0] = i64::MIN;
            if len > 1 {
                vals[len - 1] = i64::MAX;
            }
            assert_eq!(reduce_max_i64(&vals), *vals.iter().max().unwrap());
            assert_eq!(reduce_min_i64(&vals), *vals.iter().min().unwrap());
        }
    }
}
