//! Numeric NPY v1/v2/v3 and NPZ interchange. Object/structured pickle payloads
//! are rejected; only the fourteen core numeric dtypes are accepted.
use crate::lanes::invalid;
use numpy_types::{
    Complex32, Complex64, DType, DynArray, NdArray, NumPyError, Order, Result, Scalar,
    dispatch_dyn, f16,
};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Seek, Write},
    path::Path,
};
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

fn io_error(error: impl std::fmt::Display) -> NumPyError {
    NumPyError::IoError(error.to_string())
}
trait NpyScalar: Scalar {
    const DESCR: &'static str;
    fn encode(self, bytes: &mut [u8]);
    fn decode(bytes: &[u8], big: bool) -> Self;
}
macro_rules! npy_number {
    ($t:ty, $descr:literal, $n:literal) => {
        impl NpyScalar for $t {
            const DESCR: &'static str = $descr;
            fn encode(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_le_bytes());
            }
            fn decode(bytes: &[u8], big: bool) -> Self {
                let mut raw = [0u8; $n];
                raw.copy_from_slice(bytes);
                if big {
                    Self::from_be_bytes(raw)
                } else {
                    Self::from_le_bytes(raw)
                }
            }
        }
    };
}
npy_number!(i8, "|i1", 1);
npy_number!(u8, "|u1", 1);
npy_number!(i16, "<i2", 2);
npy_number!(u16, "<u2", 2);
npy_number!(i32, "<i4", 4);
npy_number!(u32, "<u4", 4);
npy_number!(i64, "<i8", 8);
npy_number!(u64, "<u8", 8);
npy_number!(f32, "<f4", 4);
npy_number!(f64, "<f8", 8);
impl NpyScalar for bool {
    const DESCR: &'static str = "|b1";
    fn encode(self, bytes: &mut [u8]) {
        bytes[0] = u8::from(self);
    }
    fn decode(bytes: &[u8], _: bool) -> Self {
        bytes[0] != 0
    }
}
impl NpyScalar for f16 {
    const DESCR: &'static str = "<f2";
    fn encode(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_bits().to_le_bytes());
    }
    fn decode(bytes: &[u8], big: bool) -> Self {
        Self::from_bits(u16::decode(bytes, big))
    }
}
macro_rules! npy_complex {
    ($t:ty,$r:ty,$descr:literal,$half:literal) => {
        impl NpyScalar for $t {
            const DESCR: &'static str = $descr;
            fn encode(self, bytes: &mut [u8]) {
                self.re.encode(&mut bytes[..$half]);
                self.im.encode(&mut bytes[$half..]);
            }
            fn decode(bytes: &[u8], big: bool) -> Self {
                Self::new(
                    <$r>::decode(&bytes[..$half], big),
                    <$r>::decode(&bytes[$half..], big),
                )
            }
        }
    };
}
npy_complex!(Complex32, f32, "<c8", 4);
npy_complex!(Complex64, f64, "<c16", 8);

fn write_typed<T: NpyScalar>(writer: &mut impl Write, a: &NdArray<T>) -> Result<()> {
    let shape = if a.ndim() == 0 {
        String::new()
    } else {
        format!(
            "{},",
            a.shape()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({shape}), }}",
        T::DESCR
    );
    // v2 avoids truncation for high-dimensional shapes; 64-byte alignment.
    let padding = (64 - ((12 + header.len() + 1) % 64)) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let len = u32::try_from(header.len()).map_err(|_| invalid("NPY header is too large"))?;
    writer.write_all(b"\x93NUMPY\x02\x00")?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    let size = T::DTYPE.itemsize();
    let mut buffer = vec![0u8; (65536 / size).max(1) * size];
    let mut used = 0;
    for x in a.iter() {
        x.encode(&mut buffer[used..used + size]);
        used += size;
        if used == buffer.len() {
            writer.write_all(&buffer)?;
            used = 0;
        }
    }
    writer.write_all(&buffer[..used])?;
    Ok(())
}
pub fn write_npy(writer: &mut impl Write, a: &DynArray) -> Result<()> {
    dispatch_dyn!(a, arr => write_typed(writer,arr))
}
pub fn save(path: impl AsRef<Path>, a: &DynArray) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    write_npy(&mut writer, a)?;
    writer.flush()?;
    Ok(())
}

struct HeaderParser<'a> {
    text: &'a str,
    pos: usize,
}
impl<'a> HeaderParser<'a> {
    fn whitespace(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.pos)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.pos += 1;
        }
    }
    fn take(&mut self, byte: u8) -> bool {
        self.whitespace();
        if self.text.as_bytes().get(self.pos) == Some(&byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn require(&mut self, byte: u8) -> Result<()> {
        if self.take(byte) {
            Ok(())
        } else {
            Err(io_error("invalid NPY header syntax"))
        }
    }
    fn string(&mut self) -> Result<&'a str> {
        self.whitespace();
        let quote = *self
            .text
            .as_bytes()
            .get(self.pos)
            .ok_or_else(|| io_error("missing NPY string"))?;
        if !matches!(quote, b'\'' | b'"') {
            return Err(io_error(
                "numeric NPY headers require string keys/descriptors",
            ));
        }
        self.pos += 1;
        let start = self.pos;
        while let Some(&b) = self.text.as_bytes().get(self.pos) {
            if b == quote {
                let end = self.pos;
                self.pos += 1;
                return Ok(&self.text[start..end]);
            }
            if matches!(b, b'\\' | b'\n' | b'\r') {
                return Err(io_error("escaped NPY descriptor is unsupported"));
            }
            self.pos += 1;
        }
        Err(io_error("unterminated NPY string"))
    }
    fn boolean(&mut self) -> Result<bool> {
        self.whitespace();
        let tail = &self.text[self.pos..];
        let (value, len) = if tail.starts_with("True") {
            (true, 4)
        } else if tail.starts_with("False") {
            (false, 5)
        } else {
            return Err(io_error("invalid NPY boolean"));
        };
        self.pos += len;
        self.whitespace();
        if !matches!(self.text.as_bytes().get(self.pos), Some(b',' | b'}')) {
            return Err(io_error("invalid NPY boolean suffix"));
        }
        Ok(value)
    }
    fn shape(&mut self) -> Result<Vec<usize>> {
        self.require(b'(')?;
        let mut shape = Vec::new();
        if self.take(b')') {
            return Ok(shape);
        }
        loop {
            self.whitespace();
            let start = self.pos;
            while self
                .text
                .as_bytes()
                .get(self.pos)
                .is_some_and(u8::is_ascii_digit)
            {
                self.pos += 1;
            }
            if start == self.pos {
                return Err(io_error("invalid NPY dimension"));
            }
            shape.push(
                self.text[start..self.pos]
                    .parse::<usize>()
                    .map_err(io_error)?,
            );
            if self.take(b')') {
                if shape.len() == 1 {
                    return Err(io_error("NPY shape must be a tuple"));
                }
                break;
            }
            self.require(b',')?;
            if self.take(b')') {
                break;
            }
        }
        Ok(shape)
    }
    fn parse(mut self) -> Result<(&'a str, Order, Vec<usize>)> {
        self.require(b'{')?;
        let (mut descr, mut order, mut shape) = (None, None, None);
        if !self.take(b'}') {
            loop {
                let key = self.string()?;
                self.require(b':')?;
                // Duplicate keys follow Python dict semantics: the last value wins.
                match key {
                    "descr" => descr = Some(self.string()?),
                    "fortran_order" => {
                        order = Some(if self.boolean()? { Order::F } else { Order::C })
                    }
                    "shape" => shape = Some(self.shape()?),
                    _ => return Err(io_error("unexpected NPY header key")),
                }
                if self.take(b'}') {
                    break;
                }
                self.require(b',')?;
                if self.take(b'}') {
                    break;
                }
            }
        }
        self.whitespace();
        if self.pos != self.text.len() {
            return Err(io_error("trailing NPY header data"));
        }
        Ok((
            descr.ok_or_else(|| io_error("missing NPY descr"))?,
            order.ok_or_else(|| io_error("missing NPY fortran_order"))?,
            shape.ok_or_else(|| io_error("missing NPY shape"))?,
        ))
    }
}
pub fn read_npy(reader: &mut impl Read) -> Result<DynArray> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic[..6] != b"\x93NUMPY" || magic[7] != 0 {
        return Err(io_error("invalid NPY magic/version"));
    }
    let len = match magic[6] {
        1 => {
            let mut b = [0u8; 2];
            reader.read_exact(&mut b)?;
            u16::from_le_bytes(b) as usize
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;
            u32::from_le_bytes(b) as usize
        }
        _ => return Err(io_error("unsupported NPY version")),
    };
    if len > 100_000 {
        return Err(io_error("NPY header exceeds 100000 bytes"));
    }
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    let header = std::str::from_utf8(&bytes).map_err(io_error)?;
    let (descr, order, shape) = HeaderParser {
        text: header,
        pos: 0,
    }
    .parse()?;
    if descr.len() < 2 {
        return Err(io_error("invalid NPY descr"));
    }
    let endian = descr.as_bytes()[0];
    if !matches!(endian, b'<' | b'>' | b'|' | b'=') {
        return Err(io_error("invalid NPY endian marker"));
    }
    let big = endian == b'>' || (endian == b'=' && cfg!(target_endian = "big"));
    let dtype = match &descr[1..] {
        "b1" | "?" => DType::Bool,
        "i1" => DType::I8,
        "u1" => DType::U8,
        "i2" => DType::I16,
        "u2" => DType::U16,
        "i4" => DType::I32,
        "u4" => DType::U32,
        "i8" => DType::I64,
        "u8" => DType::U64,
        "f2" => DType::F16,
        "f4" => DType::F32,
        "f8" => DType::F64,
        "c8" => DType::C64,
        "c16" => DType::C128,
        _ => return Err(io_error(format!("unsupported NPY dtype {descr}"))),
    };
    let count = numpy_types::shape_size(&shape)?;
    let nbytes = count
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| io_error("NPY payload size overflow"))?;
    // read_to_end through a limit avoids allocating the declared size before
    // discovering a truncated file (e.g. a malicious gigantic shape).
    let mut data = Vec::new();
    reader.take(nbytes as u64).read_to_end(&mut data)?;
    if data.len() != nbytes {
        return Err(io_error("truncated NPY payload"));
    }
    macro_rules! decode {
        ($t:ty) => {{
            let values = data
                .chunks_exact(dtype.itemsize())
                .map(|b| <$t>::decode(b, big))
                .collect();
            Ok(DynArray::from(NdArray::<$t>::from_shape_vec_order(
                &shape, values, order,
            )?))
        }};
    }
    match dtype {
        DType::Bool => decode!(bool),
        DType::I8 => decode!(i8),
        DType::U8 => decode!(u8),
        DType::I16 => decode!(i16),
        DType::U16 => decode!(u16),
        DType::I32 => decode!(i32),
        DType::U32 => decode!(u32),
        DType::I64 => decode!(i64),
        DType::U64 => decode!(u64),
        DType::F16 => decode!(f16),
        DType::F32 => decode!(f32),
        DType::F64 => decode!(f64),
        DType::C64 => decode!(Complex32),
        DType::C128 => decode!(Complex64),
    }
}
pub fn load(path: impl AsRef<Path>) -> Result<DynArray> {
    read_npy(&mut BufReader::new(File::open(path)?))
}
fn save_archive(
    path: impl AsRef<Path>,
    arrays: &[(&str, &DynArray)],
    compression: CompressionMethod,
) -> Result<()> {
    let mut names = std::collections::HashSet::new();
    for &(name, _) in arrays {
        if name.is_empty() || name.contains(['/', '\\']) || !names.insert(name) {
            return Err(invalid("NPZ names must be unique, nonempty basenames"));
        }
    }
    let mut zip = ZipWriter::new(File::create(path)?);
    for &(name, a) in arrays {
        zip.start_file(
            format!("{name}.npy"),
            SimpleFileOptions::default().compression_method(compression),
        )
        .map_err(io_error)?;
        write_npy(&mut zip, a)?;
    }
    zip.finish().map_err(io_error)?;
    Ok(())
}
pub fn savez(path: impl AsRef<Path>, arrays: &[(&str, &DynArray)]) -> Result<()> {
    save_archive(path, arrays, CompressionMethod::Stored)
}
pub fn savez_compressed(path: impl AsRef<Path>, arrays: &[(&str, &DynArray)]) -> Result<()> {
    save_archive(path, arrays, CompressionMethod::Deflated)
}
pub fn read_npz(reader: impl Read + Seek) -> Result<BTreeMap<String, DynArray>> {
    let mut archive = ZipArchive::new(reader).map_err(io_error)?;
    let mut out = BTreeMap::new();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(io_error)?;
        if let Some(name) = file.name().strip_suffix(".npy") {
            let name = name.to_owned();
            let a = read_npy(&mut file)?;
            if out.insert(name, a).is_some() {
                return Err(io_error("duplicate NPZ key"));
            }
        }
    }
    Ok(out)
}
pub fn loadz(path: impl AsRef<Path>) -> Result<BTreeMap<String, DynArray>> {
    read_npz(File::open(path)?)
}

/// Load a rectangular real text matrix (comments start at '#').
pub fn loadtxt(
    path: impl AsRef<Path>,
    delimiter: Option<char>,
    skiprows: usize,
) -> Result<NdArray<f64>> {
    let reader = BufReader::new(File::open(path)?);
    let mut data = Vec::new();
    let mut columns = None;
    let mut rows = 0;
    for line in reader.lines().skip(skiprows) {
        let line = line?;
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<_> = match delimiter {
            Some(d) => line.split(d).map(str::trim).collect(),
            None => line.split_whitespace().collect(),
        };
        if columns.is_some_and(|n| n != fields.len()) {
            return Err(io_error("text rows have inconsistent column counts"));
        }
        columns = Some(fields.len());
        rows += 1;
        for f in fields {
            data.push(f.parse::<f64>().map_err(io_error)?);
        }
    }
    match (rows, columns) {
        (0, _) => Ok(NdArray::from_vec(vec![])),
        (1, Some(1)) => Ok(NdArray::from_scalar(data[0])),
        (1, _) | (_, Some(1)) => Ok(NdArray::from_vec(data)),
        (_, Some(n)) => NdArray::from_shape_vec(&[rows, n], data),
        _ => Err(io_error("invalid text shape")),
    }
}
pub fn savetxt<T: Scalar>(path: impl AsRef<Path>, a: &NdArray<T>, delimiter: &str) -> Result<()> {
    if a.ndim() == 0 || a.ndim() > 2 || matches!(T::DTYPE, DType::C64 | DType::C128) {
        return Err(invalid("savetxt supports real vectors and matrices"));
    }
    let mut writer = BufWriter::new(File::create(path)?);
    let width = if a.ndim() == 1 { 1 } else { a.shape()[1] };
    if width == 0 {
        for _ in 0..a.shape()[0] {
            writeln!(writer)?;
        }
    } else {
        for (i, x) in a.iter().enumerate() {
            if i % width != 0 {
                write!(writer, "{delimiter}")?;
            }
            write!(writer, "{:.18e}", x.to_f64_lossy())?;
            if (i + 1) % width == 0 {
                writeln!(writer)?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}
