//! Packed structured records with named numeric fields. Access copies fields
//! into aligned typed arrays, so unaligned field offsets never form references.
use numpy_types::{Complex32, Complex64, DType, NdArray, NumPyError, Result, Scalar, f16};
use std::collections::HashSet;
fn invalid(s: &str) -> NumPyError {
    NumPyError::InvalidValue(s.into())
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordField {
    pub name: String,
    pub dtype: DType,
    pub offset: usize,
}
#[derive(Clone, Debug)]
pub struct StructuredArray {
    shape: Vec<usize>,
    fields: Vec<RecordField>,
    itemsize: usize,
    bytes: Vec<u8>,
}
pub trait RecordElement: Scalar {
    fn read(bytes: &[u8]) -> Self;
    fn write(self, bytes: &mut [u8]);
}
macro_rules! record_number{($($t:ty),*)=>{$(impl RecordElement for $t{
    fn read(bytes:&[u8])->Self{let mut raw= [0u8;std::mem::size_of::<Self>()];raw.copy_from_slice(bytes);Self::from_le_bytes(raw)}
    fn write(self,bytes:&mut[u8]){bytes.copy_from_slice(&self.to_le_bytes());}
})*};}
record_number!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
impl RecordElement for bool {
    fn read(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn write(self, bytes: &mut [u8]) {
        bytes[0] = u8::from(self);
    }
}
impl RecordElement for f16 {
    fn read(bytes: &[u8]) -> Self {
        Self::from_bits(u16::read(bytes))
    }
    fn write(self, bytes: &mut [u8]) {
        self.to_bits().write(bytes);
    }
}
macro_rules! record_complex {
    ($t:ty,$r:ty,$n:literal) => {
        impl RecordElement for $t {
            fn read(bytes: &[u8]) -> Self {
                Self::new(<$r>::read(&bytes[..$n]), <$r>::read(&bytes[$n..]))
            }
            fn write(self, bytes: &mut [u8]) {
                self.re.write(&mut bytes[..$n]);
                self.im.write(&mut bytes[$n..]);
            }
        }
    };
}
record_complex!(Complex32, f32, 4);
record_complex!(Complex64, f64, 8);
impl StructuredArray {
    pub fn zeros(shape: &[usize], fields: &[(&str, DType)]) -> Result<Self> {
        let mut size = 0usize;
        let mut names = HashSet::new();
        let mut fields_out = Vec::new();
        for &(name, dtype) in fields {
            if name.is_empty() || !names.insert(name) {
                return Err(invalid("field names must be unique and nonempty"));
            }
            fields_out.push(RecordField {
                name: name.into(),
                dtype,
                offset: size,
            });
            size = size
                .checked_add(dtype.itemsize())
                .ok_or_else(|| invalid("record size overflow"))?;
        }
        let bytes = numpy_types::shape_size(shape)?
            .checked_mul(size)
            .ok_or_else(|| invalid("record array size overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)
            .map_err(|_| invalid("record array is too large"))?;
        data.resize(bytes, 0);
        Ok(Self {
            shape: shape.to_vec(),
            fields: fields_out,
            itemsize: size,
            bytes: data,
        })
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn fields(&self) -> &[RecordField] {
        &self.fields
    }
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn field<T: RecordElement>(&self, name: &str) -> Result<NdArray<T>> {
        let field = self
            .fields
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| invalid("unknown field"))?;
        if field.dtype != T::DTYPE {
            return Err(NumPyError::DTypeError("field dtype mismatch".into()));
        }
        let n = numpy_types::shape_size(&self.shape)?;
        let mut out = Vec::with_capacity(n);
        let width = field.dtype.itemsize();
        for i in 0..n {
            let base = i * self.itemsize + field.offset;
            out.push(T::read(&self.bytes[base..base + width]));
        }
        NdArray::from_shape_vec(&self.shape, out)
    }
    pub fn set_field<T: RecordElement>(&mut self, name: &str, values: &NdArray<T>) -> Result<()> {
        let field = self
            .fields
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| invalid("unknown field"))?;
        if field.dtype != T::DTYPE {
            return Err(NumPyError::DTypeError("field dtype mismatch".into()));
        }
        let values = values.broadcast_to(&self.shape)?;
        let width = field.dtype.itemsize();
        let offset = field.offset;
        for (i, x) in values.iter().enumerate() {
            let base = i * self.itemsize + offset;
            x.write(&mut self.bytes[base..base + width]);
        }
        Ok(())
    }
    pub fn from_bytes(shape: &[usize], fields: &[(&str, DType)], bytes: Vec<u8>) -> Result<Self> {
        let mut array = Self::zeros(shape, fields)?;
        if bytes.len() != array.bytes.len() {
            return Err(invalid("record byte length mismatch"));
        }
        array.bytes = bytes;
        Ok(array)
    }
}
