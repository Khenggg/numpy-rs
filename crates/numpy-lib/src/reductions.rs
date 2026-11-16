use crate::{
    SortElement,
    lanes::{Lanes, invalid},
};
use numpy_types::{Complex32, Complex64, NdArray, Result, Scalar, f16};
use numpy_ufunc::{BinaryOp, HigherOrderScalar};

/// NumPy's default accumulation dtype for sum/product and cumulative operations.
pub trait AccumulateElement: HigherOrderScalar {
    type Acc: HigherOrderScalar;
    fn widened(a: &NdArray<Self>) -> NdArray<Self::Acc>;
}
macro_rules! same_acc {
    ($($t:ty),*) => {$(impl AccumulateElement for $t {
        type Acc = Self;
        fn widened(a: &NdArray<Self>) -> NdArray<Self> { a.clone() }
    })*};
}
same_acc!(i64, u64, f16, f32, f64, Complex32, Complex64);
macro_rules! wide_acc {
    ($to:ty; $($t:ty),*) => {$(impl AccumulateElement for $t {
        type Acc = $to;
        fn widened(a: &NdArray<Self>) -> NdArray<$to> { a.map(|x| x as $to) }
    })*};
}
wide_acc!(i64; bool, i8, i16, i32);
wide_acc!(u64; u8, u16, u32);

fn reduction_axis<T>(a: &NdArray<T>, axis: Option<isize>) -> Option<isize> {
    if a.ndim() == 0 && matches!(axis, Some(0 | -1)) {
        None
    } else {
        axis
    }
}
pub fn sum<T: AccumulateElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Acc>> {
    BinaryOp::Add.reduce_nd(&T::widened(a), reduction_axis(a, axis), keepdims, None)
}
pub fn prod<T: AccumulateElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Acc>> {
    BinaryOp::Multiply.reduce_nd(&T::widened(a), reduction_axis(a, axis), keepdims, None)
}
pub fn min<T: HigherOrderScalar>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T>> {
    BinaryOp::Minimum.reduce_nd(a, reduction_axis(a, axis), keepdims, None)
}
pub fn max<T: HigherOrderScalar>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T>> {
    BinaryOp::Maximum.reduce_nd(a, reduction_axis(a, axis), keepdims, None)
}
pub fn ptp<T: HigherOrderScalar>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T>> {
    numpy_ufunc::subtract(&max(a, axis, keepdims)?, &min(a, axis, keepdims)?)
}
fn cumulative<T: AccumulateElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    op: BinaryOp,
) -> Result<NdArray<T::Acc>> {
    let a = T::widened(a);
    match axis {
        Some(axis) if a.ndim() == 0 && matches!(axis, 0 | -1) => {
            op.accumulate_nd(&a.reshape(&[1])?, 0)
        }
        Some(axis) => op.accumulate_nd(&a, axis),
        None => op.accumulate_nd(&a.to_c_contiguous().reshape(&[a.len() as isize])?, 0),
    }
}
pub fn cumsum<T: AccumulateElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
) -> Result<NdArray<T::Acc>> {
    cumulative(a, axis, BinaryOp::Add)
}
pub fn cumprod<T: AccumulateElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
) -> Result<NdArray<T::Acc>> {
    cumulative(a, axis, BinaryOp::Multiply)
}

fn arg_extreme<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    maximum: bool,
    skip_nan: bool,
) -> Result<NdArray<i64>> {
    let p = Lanes::new(a, axis, keepdims)?;
    if p.width == 0 {
        return Err(invalid("attempt to get an extreme of an empty sequence"));
    }
    let mut data = Vec::with_capacity(p.count());
    for out in 0..p.count() {
        let mut best: Option<(usize, T)> = None;
        for (k, v) in p.lane(out).enumerate() {
            if skip_nan && v.is_nan() {
                continue;
            }
            match best {
                None => best = Some((k, v)),
                Some((_, b))
                    if !b.is_nan()
                        && (v.is_nan()
                            || (if maximum {
                                v.numpy_cmp(&b).is_gt()
                            } else {
                                v.numpy_cmp(&b).is_lt()
                            })) =>
                {
                    best = Some((k, v))
                }
                _ => {}
            }
        }
        data.push(best.ok_or_else(|| invalid("all-NaN slice encountered"))?.0 as i64);
    }
    NdArray::from_shape_vec(&p.reduced_shape, data)
}
pub fn argmin<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<i64>> {
    arg_extreme(a, axis, keepdims, false, false)
}
pub fn argmax<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<i64>> {
    arg_extreme(a, axis, keepdims, true, false)
}
pub fn nanargmin<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<i64>> {
    arg_extreme(a, axis, keepdims, false, true)
}
pub fn nanargmax<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<i64>> {
    arg_extreme(a, axis, keepdims, true, true)
}

fn truth_reduce<T: Scalar, U: Copy>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    initial: U,
    f: impl Fn(U, T) -> U,
) -> Result<NdArray<U>> {
    let p = Lanes::new(a, axis, keepdims)?;
    let mut data = vec![initial; p.count()];
    // Stream contiguous rows, rather than gathering columns through the cache.
    for outer in 0..p.outer {
        let accum = &mut data[outer * p.inner..(outer + 1) * p.inner];
        for k in 0..p.width {
            let base = (outer * p.width + k) * p.inner;
            for (slot, &x) in accum.iter_mut().zip(&p.data[base..base + p.inner]) {
                *slot = f(*slot, x);
            }
        }
    }
    NdArray::from_shape_vec(&p.reduced_shape, data)
}
pub fn all<T: Scalar>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<bool>> {
    truth_reduce(a, axis, keepdims, true, |acc, x| acc && x.is_truthy())
}
pub fn any<T: Scalar>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<bool>> {
    truth_reduce(a, axis, keepdims, false, |acc, x| acc || x.is_truthy())
}
pub fn count_nonzero<T: Scalar>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<i64>> {
    truth_reduce(a, axis, keepdims, 0, |acc, x| {
        acc + i64::from(x.is_truthy())
    })
}

fn without_nan<T: SortElement>(a: &NdArray<T>, identity: T) -> NdArray<T> {
    a.map(|x| if x.is_nan() { identity } else { x })
}
pub fn nansum<T: AccumulateElement + SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Acc>> {
    sum(&without_nan(a, T::zero()), axis, keepdims)
}
pub fn nanprod<T: AccumulateElement + SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Acc>> {
    prod(&without_nan(a, T::one()), axis, keepdims)
}
pub fn nancumsum<T: AccumulateElement + SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
) -> Result<NdArray<T::Acc>> {
    cumsum(&without_nan(a, T::zero()), axis)
}
pub fn nancumprod<T: AccumulateElement + SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
) -> Result<NdArray<T::Acc>> {
    cumprod(&without_nan(a, T::one()), axis)
}
fn nan_extreme<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    maximum: bool,
) -> Result<NdArray<T>> {
    let p = Lanes::new(a, axis, keepdims)?;
    if p.width == 0 {
        return Err(invalid(
            "zero-size array has no identity for minimum/maximum",
        ));
    }
    let data = (0..p.count())
        .map(|out| {
            let mut first = None;
            let mut best = None;
            for x in p.lane(out) {
                first.get_or_insert(x);
                if x.is_nan() {
                    continue;
                }
                if best.is_none_or(|b| {
                    if maximum {
                        !x.numpy_cmp(&b).is_lt()
                    } else {
                        !x.numpy_cmp(&b).is_gt()
                    }
                }) {
                    best = Some(x);
                }
            }
            best.unwrap_or_else(|| first.expect("non-empty lane"))
        })
        .collect();
    NdArray::from_shape_vec(&p.reduced_shape, data)
}
pub fn nanmin<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T>> {
    nan_extreme(a, axis, keepdims, false)
}
pub fn nanmax<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T>> {
    nan_extreme(a, axis, keepdims, true)
}
