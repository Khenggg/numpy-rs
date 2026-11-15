use crate::error::{NumPyError, Result};
pub use half::f16;
pub use num_complex::{Complex, Complex32, Complex64};
use std::fmt;

/// Core scalar data types supported natively by `numpy-rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DType {
    Bool = 0,
    I8 = 1,
    I16 = 2,
    I32 = 3,
    I64 = 4,
    U8 = 5,
    U16 = 6,
    U32 = 7,
    U64 = 8,
    F16 = 9,
    F32 = 10,
    F64 = 11,
    C64 = 12,
    C128 = 13,
}

/// Broad NumPy type categories (used by `can_cast(SameKind)` and `isdtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeKind {
    Bool,
    SignedInt,
    UnsignedInt,
    Float,
    Complex,
}

/// NumPy casting rules matching `numpy.can_cast(..., casting=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Casting {
    No,
    Equiv,
    #[default]
    Safe,
    SameKind,
    Unsafe,
}

/// Floating-point limits equivalent to `numpy.finfo`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FInfo {
    pub dtype: DType,
    pub bits: usize,
    pub eps: f64,
    pub epsneg: f64,
    pub max: f64,
    pub min: f64,
    pub smallest_normal: f64,
    pub smallest_subnormal: f64,
    pub resolution: f64,
    pub maxexp: i32,
    pub minexp: i32,
    pub nmant: usize,
    pub negep: i32,
    pub nexp: usize,
}

/// Integer limits equivalent to `numpy.iinfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IInfo {
    pub dtype: DType,
    pub bits: usize,
    pub min: i128,
    pub max: u128,
}

impl DType {
    /// All 14 core dtypes in canonical table order.
    pub const ALL: [DType; 14] = [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::C64,
        DType::C128,
    ];

    /// Size in bytes of a single element (`itemsize`).
    #[inline]
    pub const fn itemsize(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 | Self::F16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    /// Alignment in bytes of a single element.
    #[inline]
    pub const fn align_of(self) -> usize {
        match self {
            Self::C64 => 4,
            Self::C128 => 8,
            _ => self.itemsize(),
        }
    }

    /// Category of the dtype.
    #[inline]
    pub const fn kind(self) -> DTypeKind {
        match self {
            Self::Bool => DTypeKind::Bool,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => DTypeKind::SignedInt,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => DTypeKind::UnsignedInt,
            Self::F16 | Self::F32 | Self::F64 => DTypeKind::Float,
            Self::C64 | Self::C128 => DTypeKind::Complex,
        }
    }

    /// Canonical NumPy dtype name.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::F16 => "float16",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::C64 => "complex64",
            Self::C128 => "complex128",
        }
    }

    /// NumPy single-character type code.
    #[inline]
    pub const fn char_code(self) -> char {
        match self {
            Self::Bool => '?',
            Self::I8 => 'b',
            Self::I16 => 'h',
            Self::I32 => 'i',
            Self::I64 => 'q',
            Self::U8 => 'B',
            Self::U16 => 'H',
            Self::U32 => 'I',
            Self::U64 => 'Q',
            Self::F16 => 'e',
            Self::F32 => 'f',
            Self::F64 => 'd',
            Self::C64 => 'F',
            Self::C128 => 'D',
        }
    }

    /// Parse a NumPy dtype name or short descriptor (e.g., `"f8"`, `"float64"`, `"<i4"`).
    pub fn from_name(spec: &str) -> Result<Self> {
        let s = spec.trim();
        let s = s.strip_prefix(['<', '=', '|']).unwrap_or(s);
        match s {
            "bool" | "bool_" | "?" | "b1" => Ok(Self::Bool),
            "int8" | "i1" | "b" => Ok(Self::I8),
            "int16" | "i2" | "h" => Ok(Self::I16),
            "int32" | "i4" | "i" => Ok(Self::I32),
            "int64" | "int_" | "intp" | "i8" | "q" | "l" => Ok(Self::I64),
            "uint8" | "u1" | "B" => Ok(Self::U8),
            "uint16" | "u2" | "H" => Ok(Self::U16),
            "uint32" | "u4" | "I" => Ok(Self::U32),
            "uint64" | "uint" | "uintp" | "u8" | "Q" | "L" => Ok(Self::U64),
            "float16" | "half" | "f2" | "e" => Ok(Self::F16),
            "float32" | "single" | "f4" | "f" => Ok(Self::F32),
            "float64" | "double" | "float_" | "f8" | "d" => Ok(Self::F64),
            "complex64" | "csingle" | "c8" | "F" => Ok(Self::C64),
            "complex128" | "cdouble" | "c16" | "D" => Ok(Self::C128),
            _ => Err(NumPyError::DTypeError(format!("unknown or unsupported dtype '{spec}'"))),
        }
    }

    #[inline]
    pub const fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    #[inline]
    pub const fn is_signed_int(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    #[inline]
    pub const fn is_unsigned_int(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    #[inline]
    pub const fn is_integer(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    #[inline]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }

    #[inline]
    pub const fn is_inexact(self) -> bool {
        self.is_float() || self.is_complex()
    }

    #[inline]
    pub const fn is_numeric(self) -> bool {
        self.is_integer() || self.is_inexact()
    }

    /// Corresponding real floating dtype for complex/float types.
    #[inline]
    pub const fn real_dtype(self) -> Self {
        match self {
            Self::C64 => Self::F32,
            Self::C128 => Self::F64,
            _ => self,
        }
    }

    /// Exact floating-point properties matching `numpy.finfo`.
    pub fn finfo(self) -> Result<FInfo> {
        match self {
            Self::F16 => Ok(FInfo {
                dtype: Self::F16,
                bits: 16,
                eps: f16::EPSILON.to_f64(),
                epsneg: 4.8828125e-4, // 2^-11
                max: f16::MAX.to_f64(),
                min: f16::MIN.to_f64(),
                smallest_normal: f16::MIN_POSITIVE.to_f64(),
                smallest_subnormal: f64::from_bits(0x3e70_0000_0000_0000), // 2^-24
                resolution: 1e-3,
                maxexp: 16,
                minexp: -14,
                nmant: 10,
                negep: -11,
                nexp: 5,
            }),
            Self::F32 => Ok(FInfo {
                dtype: Self::F32,
                bits: 32,
                eps: f32::EPSILON as f64,
                epsneg: (f32::EPSILON as f64) * 0.5,
                max: f32::MAX as f64,
                min: f32::MIN as f64,
                smallest_normal: f32::MIN_POSITIVE as f64,
                smallest_subnormal: 1.401298464324817e-45, // 2^-149
                resolution: 1e-6,
                maxexp: 128,
                minexp: -126,
                nmant: 23,
                negep: -24,
                nexp: 8,
            }),
            Self::F64 => Ok(FInfo {
                dtype: Self::F64,
                bits: 64,
                eps: f64::EPSILON,
                epsneg: f64::EPSILON * 0.5,
                max: f64::MAX,
                min: f64::MIN,
                smallest_normal: f64::MIN_POSITIVE,
                smallest_subnormal: 5e-324,
                resolution: 1e-15,
                maxexp: 1024,
                minexp: -1022,
                nmant: 52,
                negep: -53,
                nexp: 11,
            }),
            Self::C64 => Self::F32.finfo(),
            Self::C128 => Self::F64.finfo(),
            _ => Err(NumPyError::DTypeError(format!("data type '{}' not inexact", self.name()))),
        }
    }

    /// Exact integer properties matching `numpy.iinfo`.
    pub fn iinfo(self) -> Result<IInfo> {
        match self {
            Self::I8 => Ok(IInfo { dtype: self, bits: 8, min: i8::MIN as i128, max: i8::MAX as u128 }),
            Self::I16 => Ok(IInfo { dtype: self, bits: 16, min: i16::MIN as i128, max: i16::MAX as u128 }),
            Self::I32 => Ok(IInfo { dtype: self, bits: 32, min: i32::MIN as i128, max: i32::MAX as u128 }),
            Self::I64 => Ok(IInfo { dtype: self, bits: 64, min: i64::MIN as i128, max: i64::MAX as u128 }),
            Self::U8 => Ok(IInfo { dtype: self, bits: 8, min: 0, max: u8::MAX as u128 }),
            Self::U16 => Ok(IInfo { dtype: self, bits: 16, min: 0, max: u16::MAX as u128 }),
            Self::U32 => Ok(IInfo { dtype: self, bits: 32, min: 0, max: u32::MAX as u128 }),
            Self::U64 => Ok(IInfo { dtype: self, bits: 64, min: 0, max: u64::MAX as u128 }),
            _ => Err(NumPyError::DTypeError(format!("data type '{}' not integer", self.name()))),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

use DType::*;
/// Exact NumPy 2.x 14x14 type promotion matrix.
const PROMOTE_TABLE: [[DType; 14]; 14] = [
    // Bool
    [Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, C64, C128],
    // I8
    [I8, I8, I16, I32, I64, I16, I32, I64, F64, F16, F32, F64, C64, C128],
    // I16
    [I16, I16, I16, I32, I64, I16, I32, I64, F64, F32, F32, F64, C64, C128],
    // I32
    [I32, I32, I32, I32, I64, I32, I32, I64, F64, F64, F64, F64, C128, C128],
    // I64
    [I64, I64, I64, I64, I64, I64, I64, I64, F64, F64, F64, F64, C128, C128],
    // U8
    [U8, I16, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, C64, C128],
    // U16
    [U16, I32, I32, I32, I64, U16, U16, U32, U64, F32, F32, F64, C64, C128],
    // U32
    [U32, I64, I64, I64, I64, U32, U32, U32, U64, F64, F64, F64, C128, C128],
    // U64
    [U64, F64, F64, F64, F64, U64, U64, U64, U64, F64, F64, F64, C128, C128],
    // F16
    [F16, F16, F32, F64, F64, F16, F32, F64, F64, F16, F32, F64, C64, C128],
    // F32
    [F32, F32, F32, F64, F64, F32, F32, F64, F64, F32, F32, F64, C64, C128],
    // F64
    [F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, C128, C128],
    // C64
    [C64, C64, C64, C128, C128, C64, C64, C128, C128, C64, C64, C128, C64, C128],
    // C128
    [C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128],
];

/// Equivalent to `numpy.promote_types(a, b)`.
#[inline]
pub const fn promote_types(a: DType, b: DType) -> DType {
    PROMOTE_TABLE[a as usize][b as usize]
}

/// Equivalent to `numpy.result_type(*dtypes)`.
pub fn result_type(dtypes: &[DType]) -> Result<DType> {
    if dtypes.is_empty() {
        return Err(NumPyError::InvalidValue("at least one dtype is required for result_type".into()));
    }
    let mut acc = dtypes[0];
    for &dt in &dtypes[1..] {
        acc = promote_types(acc, dt);
    }
    Ok(acc)
}

/// Equivalent to `numpy.can_cast(from, to, casting=...)`.
#[inline]
pub const fn can_cast(from: DType, to: DType, casting: Casting) -> bool {
    match casting {
        Casting::No | Casting::Equiv => from as u8 == to as u8,
        Casting::Safe => promote_types(from, to) as u8 == to as u8,
        Casting::SameKind => {
            if can_cast(from, to, Casting::Safe) {
                return true;
            }
            match (from.kind(), to.kind()) {
                (DTypeKind::Bool, _) => true,
                (DTypeKind::SignedInt | DTypeKind::UnsignedInt, DTypeKind::SignedInt | DTypeKind::UnsignedInt) => true,
                (DTypeKind::SignedInt | DTypeKind::UnsignedInt, DTypeKind::Float | DTypeKind::Complex) => true,
                (DTypeKind::Float, DTypeKind::Float | DTypeKind::Complex) => true,
                (DTypeKind::Complex, DTypeKind::Complex) => true,
                _ => false,
            }
        }
        Casting::Unsafe => true,
    }
}

/// Marker + conversion trait for all core NumPy element types.
///
/// # Safety
/// Implementations must guarantee that an all-zero bit pattern is a valid value
/// equal to `Self::zero()`.
pub unsafe trait Scalar: Copy + Clone + PartialEq + Send + Sync + fmt::Debug + 'static {
    const DTYPE: DType;
    fn zero() -> Self;
    fn one() -> Self;
    fn is_truthy(self) -> bool;
    fn to_f64_lossy(self) -> f64;
    fn from_f64_lossy(v: f64) -> Self;
    fn to_complex64(self) -> Complex64;
    fn from_complex64_lossy(c: Complex64) -> Self;
}

unsafe impl Scalar for bool {
    const DTYPE: DType = DType::Bool;
    #[inline] fn zero() -> Self { false }
    #[inline] fn one() -> Self { true }
    #[inline] fn is_truthy(self) -> bool { self }
    #[inline] fn to_f64_lossy(self) -> f64 { if self { 1.0 } else { 0.0 } }
    #[inline] fn from_f64_lossy(v: f64) -> Self { v != 0.0 }
    #[inline] fn to_complex64(self) -> Complex64 { Complex64::new(self.to_f64_lossy(), 0.0) }
    #[inline] fn from_complex64_lossy(c: Complex64) -> Self { c.re != 0.0 || c.im != 0.0 }
}

macro_rules! impl_int_scalar {
    ($t:ty, $dt:expr) => {
        unsafe impl Scalar for $t {
            const DTYPE: DType = $dt;
            #[inline] fn zero() -> Self { 0 }
            #[inline] fn one() -> Self { 1 }
            #[inline] fn is_truthy(self) -> bool { self != 0 }
            #[inline] fn to_f64_lossy(self) -> f64 { self as f64 }
            #[inline] fn from_f64_lossy(v: f64) -> Self { v as $t }
            #[inline] fn to_complex64(self) -> Complex64 { Complex64::new(self as f64, 0.0) }
            #[inline] fn from_complex64_lossy(c: Complex64) -> Self { c.re as $t }
        }
    };
}

impl_int_scalar!(i8, DType::I8);
impl_int_scalar!(i16, DType::I16);
impl_int_scalar!(i32, DType::I32);
impl_int_scalar!(i64, DType::I64);
impl_int_scalar!(u8, DType::U8);
impl_int_scalar!(u16, DType::U16);
impl_int_scalar!(u32, DType::U32);
impl_int_scalar!(u64, DType::U64);

unsafe impl Scalar for f16 {
    const DTYPE: DType = DType::F16;
    #[inline] fn zero() -> Self { f16::ZERO }
    #[inline] fn one() -> Self { f16::ONE }
    #[inline] fn is_truthy(self) -> bool { self != f16::ZERO && !self.is_nan() }
    #[inline] fn to_f64_lossy(self) -> f64 { self.to_f64() }
    #[inline] fn from_f64_lossy(v: f64) -> Self { f16::from_f64(v) }
    #[inline] fn to_complex64(self) -> Complex64 { Complex64::new(self.to_f64(), 0.0) }
    #[inline] fn from_complex64_lossy(c: Complex64) -> Self { f16::from_f64(c.re) }
}

unsafe impl Scalar for f32 {
    const DTYPE: DType = DType::F32;
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn is_truthy(self) -> bool { self != 0.0 }
    #[inline] fn to_f64_lossy(self) -> f64 { self as f64 }
    #[inline] fn from_f64_lossy(v: f64) -> Self { v as f32 }
    #[inline] fn to_complex64(self) -> Complex64 { Complex64::new(self as f64, 0.0) }
    #[inline] fn from_complex64_lossy(c: Complex64) -> Self { c.re as f32 }
}

unsafe impl Scalar for f64 {
    const DTYPE: DType = DType::F64;
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn is_truthy(self) -> bool { self != 0.0 }
    #[inline] fn to_f64_lossy(self) -> f64 { self }
    #[inline] fn from_f64_lossy(v: f64) -> Self { v }
    #[inline] fn to_complex64(self) -> Complex64 { Complex64::new(self, 0.0) }
    #[inline] fn from_complex64_lossy(c: Complex64) -> Self { c.re }
}

unsafe impl Scalar for Complex32 {
    const DTYPE: DType = DType::C64;
    #[inline] fn zero() -> Self { Complex32::new(0.0, 0.0) }
    #[inline] fn one() -> Self { Complex32::new(1.0, 0.0) }
    #[inline] fn is_truthy(self) -> bool { self.re != 0.0 || self.im != 0.0 }
    #[inline] fn to_f64_lossy(self) -> f64 { self.re as f64 }
    #[inline] fn from_f64_lossy(v: f64) -> Self { Complex32::new(v as f32, 0.0) }
    #[inline] fn to_complex64(self) -> Complex64 { Complex64::new(self.re as f64, self.im as f64) }
    #[inline] fn from_complex64_lossy(c: Complex64) -> Self { Complex32::new(c.re as f32, c.im as f32) }
}

unsafe impl Scalar for Complex64 {
    const DTYPE: DType = DType::C128;
    #[inline] fn zero() -> Self { Complex64::new(0.0, 0.0) }
    #[inline] fn one() -> Self { Complex64::new(1.0, 0.0) }
    #[inline] fn is_truthy(self) -> bool { self.re != 0.0 || self.im != 0.0 }
    #[inline] fn to_f64_lossy(self) -> f64 { self.re }
    #[inline] fn from_f64_lossy(v: f64) -> Self { Complex64::new(v, 0.0) }
    #[inline] fn to_complex64(self) -> Complex64 { self }
    #[inline] fn from_complex64_lossy(c: Complex64) -> Self { c }
}

/// Real floating scalar trait for generic high-precision kernels (`f32`, `f64`).
pub trait FloatElement: Scalar + PartialOrd + num_traits::Float {
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl FloatElement for f32 {
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
}

impl FloatElement for f64 {
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
}
