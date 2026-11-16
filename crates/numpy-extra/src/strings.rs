//! UTF-8 string arrays. Shapes follow NumPy broadcasting; character positions
//! are Unicode scalar indices, not byte offsets.
use numpy_types::{NumPyError, Result, broadcast_shapes, shape_size};
fn invalid(s: &str) -> NumPyError {
    NumPyError::InvalidValue(s.into())
}
fn whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StringArray {
    data: Vec<String>,
    shape: Vec<usize>,
}
impl StringArray {
    pub fn new(shape: &[usize], data: Vec<String>) -> Result<Self> {
        if shape_size(shape)? != data.len() {
            return Err(invalid("string data length does not match shape"));
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }
    pub fn from_vec(data: Vec<String>) -> Self {
        let n = data.len();
        Self {
            data,
            shape: vec![n],
        }
    }
    pub fn from_scalar(value: impl Into<String>) -> Self {
        Self {
            data: vec![value.into()],
            shape: vec![],
        }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[String] {
        &self.data
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn map(&self, f: impl Fn(&str) -> String) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|s| f(s)).collect(),
        }
    }
    pub fn upper(&self) -> Self {
        self.map(str::to_uppercase)
    }
    pub fn lower(&self) -> Self {
        self.map(str::to_lowercase)
    }
    pub fn strip(&self, chars: Option<&str>) -> Self {
        self.map(|s| match chars {
            Some(c) => s.trim_matches(|x| c.contains(x)).to_owned(),
            None => s.trim_matches(whitespace).to_owned(),
        })
    }
    pub fn lstrip(&self, chars: Option<&str>) -> Self {
        self.map(|s| match chars {
            Some(c) => s.trim_start_matches(|x| c.contains(x)).to_owned(),
            None => s.trim_start_matches(whitespace).to_owned(),
        })
    }
    pub fn rstrip(&self, chars: Option<&str>) -> Self {
        self.map(|s| match chars {
            Some(c) => s.trim_end_matches(|x| c.contains(x)).to_owned(),
            None => s.trim_end_matches(whitespace).to_owned(),
        })
    }
    pub fn replace(&self, old: &str, new: &str, count: Option<usize>) -> Self {
        self.map(|s| match count {
            Some(n) => s.replacen(old, new, n),
            None => s.replace(old, new),
        })
    }
    pub fn split(
        &self,
        separator: Option<&str>,
        maxsplit: Option<usize>,
    ) -> Result<Vec<Vec<String>>> {
        if separator == Some("") {
            return Err(invalid("empty separator"));
        }
        Ok(self
            .data
            .iter()
            .map(|s| match separator {
                Some(sep) => match maxsplit {
                    Some(n) => s
                        .splitn(n.saturating_add(1), sep)
                        .map(str::to_owned)
                        .collect(),
                    None => s.split(sep).map(str::to_owned).collect(),
                },
                None => {
                    let parts = s.split(whitespace).filter(|s| !s.is_empty());
                    match maxsplit {
                        None => parts.map(str::to_owned).collect(),
                        Some(n) => {
                            let mut result = Vec::new();
                            let mut tail = s.trim_start_matches(whitespace);
                            for _ in 0..n {
                                if tail.is_empty() {
                                    break;
                                }
                                let end = tail.find(whitespace).unwrap_or(tail.len());
                                result.push(tail[..end].to_owned());
                                tail = tail[end..].trim_start_matches(whitespace);
                            }
                            if !tail.is_empty() {
                                result.push(tail.to_owned());
                            }
                            result
                        }
                    }
                }
            })
            .collect())
    }
    pub fn find(&self, needle: &str) -> numpy_types::NdArray<i64> {
        // Construction is infallible: map preserves the already-validated shape.
        let indices = self
            .data
            .iter()
            .map(|s| s.find(needle).map_or(-1, |b| s[..b].chars().count() as i64))
            .collect();
        numpy_types::NdArray::from_shape_vec(&self.shape, indices).expect("validated string shape")
    }
    pub fn startswith(&self, prefix: &str) -> numpy_types::NdArray<bool> {
        numpy_types::NdArray::from_shape_vec(
            &self.shape,
            self.data.iter().map(|s| s.starts_with(prefix)).collect(),
        )
        .expect("validated string shape")
    }
    pub fn endswith(&self, suffix: &str) -> numpy_types::NdArray<bool> {
        numpy_types::NdArray::from_shape_vec(
            &self.shape,
            self.data.iter().map(|s| s.ends_with(suffix)).collect(),
        )
        .expect("validated string shape")
    }
    pub fn zfill(&self, width: usize) -> Self {
        self.map(|s| {
            let n = width.saturating_sub(s.chars().count());
            if n == 0 {
                return s.to_owned();
            }
            if let Some(tail) = s.strip_prefix(['+', '-']) {
                format!("{}{}{tail}", &s[..1], "0".repeat(n))
            } else {
                format!("{}{s}", "0".repeat(n))
            }
        })
    }
    pub fn center(&self, width: usize, fill: char) -> Self {
        self.map(|s| {
            let n = width.saturating_sub(s.chars().count());
            let left = n / 2 + (n % 2) * (width % 2);
            format!(
                "{}{s}{}",
                fill.to_string().repeat(left),
                fill.to_string().repeat(n - left)
            )
        })
    }
    pub fn ljust(&self, width: usize, fill: char) -> Self {
        self.map(|s| {
            format!(
                "{s}{}",
                fill.to_string()
                    .repeat(width.saturating_sub(s.chars().count()))
            )
        })
    }
    pub fn rjust(&self, width: usize, fill: char) -> Self {
        self.map(|s| {
            format!(
                "{}{s}",
                fill.to_string()
                    .repeat(width.saturating_sub(s.chars().count()))
            )
        })
    }
    pub fn join(&self, separator: &str) -> Self {
        self.map(|s| {
            s.chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(separator)
        })
    }
    pub fn add(&self, other: &Self) -> Result<Self> {
        let shape = broadcast_shapes(&self.shape, &other.shape)?;
        let n = shape_size(&shape)?;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.broadcast_index(i, &shape);
            let b = other.broadcast_index(i, &shape);
            data.push(format!("{}{}", self.data[a], other.data[b]));
        }
        Self::new(&shape, data)
    }
    fn broadcast_index(&self, mut i: usize, shape: &[usize]) -> usize {
        let mut out = 0;
        let mut stride = 1;
        for j in (0..shape.len()).rev() {
            let coord = i % shape[j];
            i /= shape[j];
            if j + self.shape.len() >= shape.len() {
                let d = self.shape[j + self.shape.len() - shape.len()];
                if d != 1 {
                    out += coord * stride;
                }
                stride *= d;
            }
        }
        out
    }
}
