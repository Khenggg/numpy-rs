//! Masked typed arrays; `true` means excluded. Input NaNs remain data.
use numpy_lib::statistics::StatWork;
use numpy_lib::{AccumulateElement, SortElement, StatisticsElement};
use numpy_types::{NdArray, NumPyError, Result, Scalar, normalize_axis, shape_size};

#[derive(Clone, Debug)]
pub struct MaskedArray<T: Copy> {
    data: NdArray<T>,
    mask: NdArray<bool>,
    pub fill_value: T,
}
impl<T: Scalar> MaskedArray<T> {
    pub fn new(data: NdArray<T>, mask: NdArray<bool>) -> Result<Self> {
        let mask = mask.broadcast_to(data.shape())?;
        let fill_value = T::from_f64_lossy(match T::DTYPE {
            numpy_types::DType::Bool => 1.,
            numpy_types::DType::I8 => 999999i64 as i8 as f64,
            numpy_types::DType::U8 => 999999i64 as u8 as f64,
            numpy_types::DType::I16 => 999999i64 as i16 as f64,
            numpy_types::DType::U16 => 999999i64 as u16 as f64,
            numpy_types::DType::I32
            | numpy_types::DType::I64
            | numpy_types::DType::U32
            | numpy_types::DType::U64 => 999999.,
            _ => 1e20,
        });
        Ok(Self {
            data,
            mask,
            fill_value,
        })
    }
    pub fn with_fill_value(mut self, fill_value: T) -> Self {
        self.fill_value = fill_value;
        self
    }
    pub fn data(&self) -> &NdArray<T> {
        &self.data
    }
    pub fn mask(&self) -> &NdArray<bool> {
        &self.mask
    }
    pub fn set_mask(&mut self, mask: NdArray<bool>) -> Result<()> {
        self.mask = mask.broadcast_to(self.data.shape())?;
        Ok(())
    }
    pub fn set_data(&mut self, data: NdArray<T>) -> Result<()> {
        let mask = self.mask.broadcast_to(data.shape())?;
        self.data = data;
        self.mask = mask;
        Ok(())
    }
    pub fn filled(&self, fill_value: Option<T>) -> Result<NdArray<T>> {
        NdArray::from_shape_vec(
            self.data.shape(),
            self.data
                .iter()
                .zip(self.mask.iter())
                .map(|(x, m)| {
                    if m {
                        fill_value.unwrap_or(self.fill_value)
                    } else {
                        x
                    }
                })
                .collect(),
        )
    }
    pub fn compressed(&self) -> NdArray<T> {
        NdArray::from_vec(
            self.data
                .iter()
                .zip(self.mask.iter())
                .filter_map(|(x, m)| (!m).then_some(x))
                .collect(),
        )
    }
    pub fn count(&self, axis: Option<isize>, keepdims: bool) -> Result<NdArray<i64>> {
        numpy_lib::sum(&self.mask.map(|m| i64::from(!m)), axis, keepdims)
    }
    pub fn map<U: Scalar>(&self, f: impl Fn(T) -> U) -> Result<MaskedArray<U>> {
        MaskedArray::new(self.data.map(f), self.mask.clone())
    }
    pub fn add(&self, other: &Self) -> Result<Self>
    where
        T: numpy_ufunc::UfuncScalar,
    {
        let data = numpy_ufunc::add(&self.data, &other.data)?;
        let mask = numpy_ufunc::logical_or(&self.mask, &other.mask)?;
        Self::new(data, mask)
    }
    pub fn multiply(&self, other: &Self) -> Result<Self>
    where
        T: numpy_ufunc::UfuncScalar,
    {
        Self::new(
            numpy_ufunc::multiply(&self.data, &other.data)?,
            numpy_ufunc::logical_or(&self.mask, &other.mask)?,
        )
    }
}
impl<T: AccumulateElement> MaskedArray<T> {
    pub fn sum(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T::Acc>> {
        let clean = NdArray::from_shape_vec(
            self.data.shape(),
            self.data
                .iter()
                .zip(self.mask.iter())
                .map(|(x, m)| if m { T::zero() } else { x })
                .collect(),
        )?;
        MaskedArray::new(
            numpy_lib::sum(&clean, axis, keepdims)?,
            self.count(axis, keepdims)?.map(|n| n == 0),
        )
    }
    pub fn prod(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T::Acc>> {
        let clean = NdArray::from_shape_vec(
            self.data.shape(),
            self.data
                .iter()
                .zip(self.mask.iter())
                .map(|(x, m)| if m { T::one() } else { x })
                .collect(),
        )?;
        MaskedArray::new(
            numpy_lib::prod(&clean, axis, keepdims)?,
            self.count(axis, keepdims)?.map(|n| n == 0),
        )
    }
}
impl<T: StatisticsElement> MaskedArray<T> {
    pub fn mean(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T::Mean>> {
        let p = MaskedLanes::new(self, axis, keepdims)?;
        let mut result = Vec::with_capacity(p.count());
        let mut mask = Vec::with_capacity(p.count());
        for i in 0..p.count() {
            let mut sum = T::Work::zero();
            let mut n = 0usize;
            for (x, m) in p.lane(i) {
                if !m {
                    sum = sum + x.work();
                    n += 1;
                }
            }
            let value = T::mean_value(sum.divide(n as f64));
            let complex = value.to_complex64();
            mask.push(
                n == 0
                    || (!p.shape.is_empty()
                        && (!complex.re.is_finite() || !complex.im.is_finite())),
            );
            result.push(value);
        }
        MaskedArray::new(
            NdArray::from_shape_vec(&p.shape, result)?,
            NdArray::from_shape_vec(&p.shape, mask)?,
        )
    }
    pub fn var(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        ddof: usize,
    ) -> Result<MaskedArray<T::Real>> {
        self.variance(axis, keepdims, ddof, false)
    }
    pub fn std(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        ddof: usize,
    ) -> Result<MaskedArray<T::Real>> {
        self.variance(axis, keepdims, ddof, true)
    }
    fn variance(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        ddof: usize,
        sqrt: bool,
    ) -> Result<MaskedArray<T::Real>> {
        let p = MaskedLanes::new(self, axis, keepdims)?;
        let mut result = Vec::with_capacity(p.count());
        let mut mask = Vec::with_capacity(p.count());
        for i in 0..p.count() {
            let mut sum = T::Work::zero();
            let mut n = 0usize;
            for (x, m) in p.lane(i) {
                if !m {
                    sum = sum + x.work();
                    n += 1;
                }
            }
            let mean = sum.divide(n as f64);
            let mean_value = T::mean_value(mean).to_complex64();
            // ma.var computes a keepdims mean first. Non-finite means mask all
            // anomalies, even when the final reduction itself is scalar.
            let mean_mask = n == 0
                || (self.data.ndim() > 0
                    && (!mean_value.re.is_finite() || !mean_value.im.is_finite()));
            let complex = matches!(T::DTYPE, numpy_types::DType::C64 | numpy_types::DType::C128);
            let mut squares = 0.;
            let mut valid = 0;
            if !mean_mask {
                for (x, m) in p.lane(i) {
                    if !m {
                        let square = (x.work() - mean).norm_squared();
                        // Complex ma.power masks overflow; real in-place multiply does not.
                        if complex && !square.is_finite() {
                            continue;
                        }
                        squares += square;
                        valid += 1;
                    }
                }
            }
            let denominator = n as f64 - ddof as f64;
            let raw = squares / denominator;
            let division_mask = valid == 0
                || !raw.is_finite()
                || squares.abs() * f64::MIN_POSITIVE >= denominator.abs();
            let mut value = if !p.shape.is_empty() && division_mask {
                squares
            } else {
                raw
            };
            let mut masked = if p.shape.is_empty() {
                division_mask
            } else {
                n == 0 || denominator <= 0.
            };
            if sqrt && !masked {
                let root = value.sqrt();
                if value < 0. || !root.is_finite() {
                    masked = true;
                } else {
                    value = root;
                }
            }
            mask.push(masked);
            result.push(T::Real::from_f64_lossy(value));
        }
        MaskedArray::new(
            NdArray::from_shape_vec(&p.shape, result)?,
            NdArray::from_shape_vec(&p.shape, mask)?,
        )
    }
}
impl<T: SortElement> MaskedArray<T> {
    pub fn min(&self, axis: Option<isize>, keepdims: bool) -> Result<Self> {
        self.extreme(axis, keepdims, false)
    }
    pub fn max(&self, axis: Option<isize>, keepdims: bool) -> Result<Self> {
        self.extreme(axis, keepdims, true)
    }
    fn extreme(&self, axis: Option<isize>, keepdims: bool, maximum: bool) -> Result<Self> {
        let p = MaskedLanes::new(self, axis, keepdims)?;
        if p.width == 0 {
            return Err(NumPyError::InvalidValue(
                "zero-size array has no min/max identity".into(),
            ));
        }
        let mut result = Vec::with_capacity(p.count());
        let mut mask = Vec::with_capacity(p.count());
        for i in 0..p.count() {
            let mut best: Option<T> = None;
            for (x, m) in p.lane(i) {
                if m {
                    continue;
                }
                if best.is_none_or(|b| {
                    !b.is_nan()
                        && (x.is_nan()
                            || if maximum {
                                !x.numpy_cmp(&b).is_lt()
                            } else {
                                !x.numpy_cmp(&b).is_gt()
                            })
                }) {
                    best = Some(x);
                }
            }
            mask.push(best.is_none());
            result.push(best.unwrap_or(T::zero()));
        }
        Self::new(
            NdArray::from_shape_vec(&p.shape, result)?,
            NdArray::from_shape_vec(&p.shape, mask)?,
        )
    }
}
struct MaskedLanes<T: Scalar> {
    data: Vec<T>,
    mask: Vec<bool>,
    outer: usize,
    width: usize,
    inner: usize,
    shape: Vec<usize>,
}
impl<T: Scalar> MaskedLanes<T> {
    fn new(a: &MaskedArray<T>, axis: Option<isize>, keepdims: bool) -> Result<Self> {
        let (data, mask) = (a.data.to_vec(), a.mask.to_vec());
        if let Some(axis) = axis {
            if a.data.ndim() == 0 && (axis == 0 || axis == -1) {
                return Ok(Self {
                    data,
                    mask,
                    outer: 1,
                    width: 1,
                    inner: 1,
                    shape: vec![],
                });
            }
            let ax = normalize_axis(axis, a.data.ndim())?;
            let mut shape = a.data.shape().to_vec();
            if keepdims {
                shape[ax] = 1;
            } else {
                shape.remove(ax);
            }
            Ok(Self {
                data,
                mask,
                outer: shape_size(&a.data.shape()[..ax])?,
                width: a.data.shape()[ax],
                inner: shape_size(&a.data.shape()[ax + 1..])?,
                shape,
            })
        } else {
            Ok(Self {
                data,
                mask,
                outer: 1,
                width: a.data.len(),
                inner: 1,
                shape: if keepdims {
                    vec![1; a.data.ndim()]
                } else {
                    vec![]
                },
            })
        }
    }
    fn count(&self) -> usize {
        self.outer * self.inner
    }
    fn lane(&self, out: usize) -> impl Iterator<Item = (T, bool)> + '_ {
        let base = (out / self.inner) * self.width * self.inner + out % self.inner;
        (0..self.width).map(move |k| {
            let i = base + k * self.inner;
            (self.data[i], self.mask[i])
        })
    }
}
