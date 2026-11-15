use std::fmt;

/// Unified error type matching NumPy runtime exception categories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumPyError {
    /// Equivalent to Python `ValueError` for incompatible shapes.
    ShapeMismatch(String),
    /// Equivalent to Python `ValueError` when shapes cannot broadcast together.
    BroadcastError(String),
    /// Equivalent to Python `IndexError` for out-of-bounds or invalid indices.
    IndexOutOfBounds(String),
    /// Equivalent to Python `AxisError` (`ValueError` subclass).
    InvalidAxis { axis: isize, ndim: usize },
    /// Equivalent to Python `TypeError` for unsupported or incompatible dtypes/casts.
    DTypeError(String),
    /// Equivalent to `numpy.linalg.LinAlgError`.
    LinAlgError(String),
    /// Equivalent to Python `ValueError` for invalid argument values.
    InvalidValue(String),
    /// I/O error (e.g. NPY/NPZ/memmap/text loading).
    IoError(String),
    /// Feature not yet implemented.
    NotImplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, NumPyError>;

impl fmt::Display for NumPyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch(msg) => write!(f, "ShapeMismatch: {msg}"),
            Self::BroadcastError(msg) => write!(f, "BroadcastError: {msg}"),
            Self::IndexOutOfBounds(msg) => write!(f, "IndexOutOfBounds: {msg}"),
            Self::InvalidAxis { axis, ndim } => {
                write!(f, "AxisError: axis {axis} is out of bounds for array of dimension {ndim}")
            }
            Self::DTypeError(msg) => write!(f, "DTypeError: {msg}"),
            Self::LinAlgError(msg) => write!(f, "LinAlgError: {msg}"),
            Self::InvalidValue(msg) => write!(f, "InvalidValue: {msg}"),
            Self::IoError(msg) => write!(f, "IoError: {msg}"),
            Self::NotImplemented(feat) => write!(f, "NotImplemented: {feat}"),
        }
    }
}

impl std::error::Error for NumPyError {}

impl From<std::io::Error> for NumPyError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}
