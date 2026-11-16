use numpy_types::{NdArray, NumPyError, Result, normalize_axis, shape_size};
use std::borrow::Cow;

/// C-order lane geometry. Contiguous inputs are borrowed; other layouts are
/// gathered once, rather than allocating coordinates for every element.
pub(crate) struct Lanes<'a, T: Copy> {
    pub data: Cow<'a, [T]>,
    pub outer: usize,
    pub width: usize,
    pub inner: usize,
    pub shape: Vec<usize>,
    pub reduced_shape: Vec<usize>,
}

impl<'a, T: Copy> Lanes<'a, T> {
    pub fn new(a: &'a NdArray<T>, axis: Option<isize>, keepdims: bool) -> Result<Self> {
        if a.shape().iter().any(|&d| d > isize::MAX as usize) {
            return Err(NumPyError::ShapeMismatch(
                "array dimension exceeds the signed index range".into(),
            ));
        }
        let data = match a.as_slice() {
            Some(s) => Cow::Borrowed(s),
            None => Cow::Owned(a.to_vec()),
        };
        if let Some(axis) = axis {
            // NumPy reductions allow axis 0/-1 for a scalar.
            if a.ndim() == 0 && (axis == 0 || axis == -1) {
                return Ok(Self {
                    data,
                    outer: 1,
                    width: 1,
                    inner: 1,
                    shape: vec![],
                    reduced_shape: vec![],
                });
            }
            let ax = normalize_axis(axis, a.ndim())?;
            let mut reduced_shape = a.shape().to_vec();
            if keepdims {
                reduced_shape[ax] = 1;
            } else {
                reduced_shape.remove(ax);
            }
            Ok(Self {
                data,
                outer: shape_size(&a.shape()[..ax])?,
                width: a.shape()[ax],
                inner: shape_size(&a.shape()[ax + 1..])?,
                shape: a.shape().to_vec(),
                reduced_shape,
            })
        } else {
            Ok(Self {
                data,
                outer: 1,
                width: a.len(),
                inner: 1,
                shape: vec![a.len()],
                reduced_shape: if keepdims { vec![1; a.ndim()] } else { vec![] },
            })
        }
    }

    pub fn count(&self) -> usize {
        self.outer * self.inner
    }

    pub fn lane(&self, out: usize) -> impl Iterator<Item = T> + '_ {
        let base = (out / self.inner) * self.width * self.inner + out % self.inner;
        (0..self.width).map(move |k| self.data[base + k * self.inner])
    }

    pub fn base(&self, out: usize) -> usize {
        (out / self.inner) * self.width * self.inner + out % self.inner
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> NumPyError {
    NumPyError::InvalidValue(message.into())
}
