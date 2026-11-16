#![allow(dead_code)]

use numpy_types::{
    f16, Complex32, Complex64, DType, DynArray, NdArray, Scalar, SIMD_ALIGN,
};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Python NumPy 2.5.4 helper prelude embedded in oracle scripts.
pub const PY_ORACLE_PRELUDE: &str = r#"
import numpy as np

def emit_arr(tag, arr):
    arr = np.asarray(arr)
    shape_str = ",".join(str(d) for d in arr.shape)
    hex_str = np.ascontiguousarray(arr).tobytes("C").hex()
    print(f"ARR|{tag}|{arr.dtype.name}|{shape_str}|{hex_str}")

def emit_val(tag, val):
    print(f"VAL|{tag}|{val}")

def emit_err(tag, exc):
    print(f"ERR|{tag}|{type(exc).__name__}|{exc}")
"#;

/// Resolve the live Python NumPy 2.5.4 executable path.
pub fn python_exe() -> PathBuf {
    if let Ok(p) = std::env::var("NUMPY_RS_PYTHON") {
        let pb = PathBuf::from(p);
        assert!(pb.exists(), "NUMPY_RS_PYTHON does not exist: {}", pb.display());
        return pb;
    }
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let win_venv = repo_root.join(".venv").join("Scripts").join("python.exe");
    if win_venv.exists() {
        return win_venv;
    }
    let unix_venv = repo_root.join(".venv").join("bin").join("python");
    if unix_venv.exists() {
        return unix_venv;
    }
    let fallback = PathBuf::from(r"D:\numpy-rs\.venv\Scripts\python.exe");
    if fallback.exists() {
        return fallback;
    }
    for cmd in ["python3", "python"] {
        if Command::new(cmd)
            .arg("-c")
            .arg("import numpy")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return PathBuf::from(cmd);
        }
    }
    panic!(
        "Python NumPy oracle not found (set NUMPY_RS_PYTHON or create .venv; checked {} and {})",
        win_venv.display(),
        unix_venv.display()
    );
}

/// Execute a Python script via stdin on the live NumPy 2.5.4 oracle and return stdout.
pub fn run_python(script: &str) -> String {
    let exe = python_exe();
    let mut child = Command::new(&exe)
        .arg("-")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn Python oracle {}: {e}", exe.display()));

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(script.as_bytes())
            .expect("failed to write script to python stdin");
    }

    let output = child
        .wait_with_output()
        .expect("failed waiting for python oracle");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "Python oracle exited with {:?}\n--- STDERR ---\n{}\n--- STDOUT ---\n{}\n--- SCRIPT ---\n{}",
        output.status.code(),
        stderr,
        stdout,
        script
    );
    stdout
}

/// Execute a Python script with `PY_ORACLE_PRELUDE` prepended and parse `ARR|` / `VAL|` / `ERR|` outputs.
pub fn run_oracle(body: &str) -> OracleOutput {
    let full_script = format!("{PY_ORACLE_PRELUDE}\n{body}\n");
    let stdout = run_python(&full_script);
    OracleOutput::parse(&stdout)
}

#[derive(Debug, Clone)]
pub struct OracleArray {
    pub tag: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct OracleOutput {
    pub arrays: HashMap<String, OracleArray>,
    pub values: HashMap<String, String>,
    pub errors: HashMap<String, (String, String)>,
    pub raw_lines: Vec<String>,
}

impl OracleOutput {
    pub fn parse(stdout: &str) -> Self {
        let mut out = Self::default();
        for raw_line in stdout.lines() {
            let line = raw_line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            out.raw_lines.push(line.to_string());
            if let Some(rest) = line.strip_prefix("ARR|") {
                let mut parts = rest.splitn(4, '|');
                let tag = parts.next().expect("missing tag").to_string();
                let dt_name = parts.next().expect("missing dtype");
                let shape_str = parts.next().expect("missing shape");
                let hex_str = parts.next().expect("missing hex");
                let dtype = DType::from_name(dt_name)
                    .unwrap_or_else(|e| panic!("unknown oracle dtype '{dt_name}': {e}"));
                let shape = if shape_str.is_empty() {
                    Vec::new()
                } else {
                    shape_str
                        .split(',')
                        .map(|s| s.parse::<usize>().expect("invalid shape dim"))
                        .collect()
                };
                let bytes = hex_to_bytes(hex_str);
                let expected_len: usize = shape.iter().product::<usize>() * dtype.itemsize();
                assert_eq!(
                    bytes.len(),
                    expected_len,
                    "oracle byte length mismatch for tag '{tag}' ({dt_name}, shape {shape:?})"
                );
                out.arrays.insert(
                    tag.clone(),
                    OracleArray {
                        tag,
                        dtype,
                        shape,
                        bytes,
                    },
                );
            } else if let Some(rest) = line.strip_prefix("VAL|") {
                let mut parts = rest.splitn(2, '|');
                let tag = parts.next().expect("missing tag").to_string();
                let val = parts.next().unwrap_or("").to_string();
                out.values.insert(tag, val);
            } else if let Some(rest) = line.strip_prefix("ERR|") {
                let mut parts = rest.splitn(3, '|');
                let tag = parts.next().expect("missing tag").to_string();
                let exc = parts.next().unwrap_or("").to_string();
                let msg = parts.next().unwrap_or("").to_string();
                out.errors.insert(tag, (exc, msg));
            }
        }
        out
    }

    pub fn arr(&self, tag: &str) -> &OracleArray {
        self.arrays
            .get(tag)
            .unwrap_or_else(|| panic!("missing oracle array '{tag}'; available: {:?}", self.arrays.keys().collect::<Vec<_>>()))
    }

    pub fn val(&self, tag: &str) -> &str {
        self.values
            .get(tag)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("missing oracle value '{tag}'"))
    }
}

impl OracleArray {
    pub fn to_bool_vec(&self) -> Vec<bool> {
        assert_eq!(self.dtype, DType::Bool);
        self.bytes.iter().map(|&b| b != 0).collect()
    }

    pub fn to_i8_vec(&self) -> Vec<i8> {
        assert_eq!(self.dtype, DType::I8);
        self.bytes.iter().map(|&b| b as i8).collect()
    }

    pub fn to_i16_vec(&self) -> Vec<i16> {
        assert_eq!(self.dtype, DType::I16);
        self.bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    pub fn to_i32_vec(&self) -> Vec<i32> {
        assert_eq!(self.dtype, DType::I32);
        self.bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn to_i64_vec(&self) -> Vec<i64> {
        assert_eq!(self.dtype, DType::I64);
        self.bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    pub fn to_u8_vec(&self) -> Vec<u8> {
        assert_eq!(self.dtype, DType::U8);
        self.bytes.clone()
    }

    pub fn to_u16_vec(&self) -> Vec<u16> {
        assert_eq!(self.dtype, DType::U16);
        self.bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    pub fn to_u32_vec(&self) -> Vec<u32> {
        assert_eq!(self.dtype, DType::U32);
        self.bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn to_u64_vec(&self) -> Vec<u64> {
        assert_eq!(self.dtype, DType::U64);
        self.bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    pub fn to_f16_vec(&self) -> Vec<f16> {
        assert_eq!(self.dtype, DType::F16);
        self.bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        assert_eq!(self.dtype, DType::F32);
        self.bytes
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        assert_eq!(self.dtype, DType::F64);
        self.bytes
            .chunks_exact(8)
            .map(|c| f64::from_bits(u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])))
            .collect()
    }

    pub fn to_c64_vec(&self) -> Vec<Complex32> {
        assert_eq!(self.dtype, DType::C64);
        self.bytes
            .chunks_exact(8)
            .map(|c| {
                let re = f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                let im = f32::from_bits(u32::from_le_bytes([c[4], c[5], c[6], c[7]]));
                Complex32::new(re, im)
            })
            .collect()
    }

    pub fn to_c128_vec(&self) -> Vec<Complex64> {
        assert_eq!(self.dtype, DType::C128);
        self.bytes
            .chunks_exact(16)
            .map(|c| {
                let re = f64::from_bits(u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]));
                let im = f64::from_bits(u64::from_le_bytes([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]));
                Complex64::new(re, im)
            })
            .collect()
    }

    pub fn to_dyn_array(&self) -> DynArray {
        let s = &self.shape;
        match self.dtype {
            DType::Bool => DynArray::Bool(NdArray::from_shape_vec(s, self.to_bool_vec()).unwrap()),
            DType::I8 => DynArray::I8(NdArray::from_shape_vec(s, self.to_i8_vec()).unwrap()),
            DType::I16 => DynArray::I16(NdArray::from_shape_vec(s, self.to_i16_vec()).unwrap()),
            DType::I32 => DynArray::I32(NdArray::from_shape_vec(s, self.to_i32_vec()).unwrap()),
            DType::I64 => DynArray::I64(NdArray::from_shape_vec(s, self.to_i64_vec()).unwrap()),
            DType::U8 => DynArray::U8(NdArray::from_shape_vec(s, self.to_u8_vec()).unwrap()),
            DType::U16 => DynArray::U16(NdArray::from_shape_vec(s, self.to_u16_vec()).unwrap()),
            DType::U32 => DynArray::U32(NdArray::from_shape_vec(s, self.to_u32_vec()).unwrap()),
            DType::U64 => DynArray::U64(NdArray::from_shape_vec(s, self.to_u64_vec()).unwrap()),
            DType::F16 => DynArray::F16(NdArray::from_shape_vec(s, self.to_f16_vec()).unwrap()),
            DType::F32 => DynArray::F32(NdArray::from_shape_vec(s, self.to_f32_vec()).unwrap()),
            DType::F64 => DynArray::F64(NdArray::from_shape_vec(s, self.to_f64_vec()).unwrap()),
            DType::C64 => DynArray::C64(NdArray::from_shape_vec(s, self.to_c64_vec()).unwrap()),
            DType::C128 => DynArray::C128(NdArray::from_shape_vec(s, self.to_c128_vec()).unwrap()),
        }
    }
}

pub fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

pub fn hex_to_bytes(hex: &str) -> Vec<u8> {
    assert!(hex.len() % 2 == 0, "odd hex length {}", hex.len());
    let mut out = Vec::with_capacity(hex.len() / 2);
    let b = hex.as_bytes();
    for i in (0..b.len()).step_by(2) {
        let hi = (b[i] as char).to_digit(16).expect("invalid hex digit") as u8;
        let lo = (b[i + 1] as char).to_digit(16).expect("invalid hex digit") as u8;
        out.push((hi << 4) | lo);
    }
    out
}

pub fn dyn_to_le_bytes(arr: &DynArray) -> Vec<u8> {
    let mut out = Vec::with_capacity(arr.nbytes());
    match arr {
        DynArray::Bool(a) => {
            for v in a.iter() {
                out.push(u8::from(v));
            }
        }
        DynArray::I8(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DynArray::I16(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DynArray::I32(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DynArray::I64(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DynArray::U8(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DynArray::U16(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DynArray::U32(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DynArray::U64(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DynArray::F16(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        DynArray::F32(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        DynArray::F64(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        DynArray::C64(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.re.to_bits().to_le_bytes());
                out.extend_from_slice(&v.im.to_bits().to_le_bytes());
            }
        }
        DynArray::C128(a) => {
            for v in a.iter() {
                out.extend_from_slice(&v.re.to_bits().to_le_bytes());
                out.extend_from_slice(&v.im.to_bits().to_le_bytes());
            }
        }
    }
    out
}

/// Generate Python code defining variable `var` with exact bit-level contents & shape of `arr`.
pub fn py_assign_dyn(var: &str, arr: &DynArray) -> String {
    let hex = bytes_to_hex(&dyn_to_le_bytes(arr));
    let shape_tuple = if arr.ndim() == 0 {
        "()".to_string()
    } else if arr.ndim() == 1 {
        format!("({},)", arr.shape()[0])
    } else {
        let joined = arr
            .shape()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!("({joined})")
    };
    format!(
        "{var} = np.frombuffer(bytes.fromhex(\"{hex}\"), dtype=\"{}\").reshape({shape_tuple}).copy()",
        arr.dtype().name()
    )
}

pub fn py_assign_f64(var: &str, arr: &NdArray<f64>) -> String {
    py_assign_dyn(var, &DynArray::F64(arr.clone()))
}

pub fn py_assign_f32(var: &str, arr: &NdArray<f32>) -> String {
    py_assign_dyn(var, &DynArray::F32(arr.clone()))
}

pub fn py_assign_c128(var: &str, arr: &NdArray<Complex64>) -> String {
    py_assign_dyn(var, &DynArray::C128(arr.clone()))
}

pub fn py_assign_c64(var: &str, arr: &NdArray<Complex32>) -> String {
    py_assign_dyn(var, &DynArray::C64(arr.clone()))
}

/// Assert 64-byte SIMD alignment for non-empty `NdArray<T>` allocations.
pub fn assert_aligned<T>(arr: &NdArray<T>, ctx: &str) {
    if !arr.is_empty() {
        assert_eq!(
            (arr.storage_ptr() as usize) % SIMD_ALIGN,
            0,
            "{ctx}: storage pointer {:p} is not {}-byte aligned",
            arr.storage_ptr(),
            SIMD_ALIGN
        );
    }
}

/// IEEE-754 ULP distance for f64 (treating -0.0 and +0.0 as distance 0).
pub fn ulp_diff_f64(a: f64, b: f64) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let to_ordered = |x: f64| -> u64 {
        let bits = x.to_bits();
        if (bits as i64) < 0 {
            0x8000_0000_0000_0000u64.wrapping_sub(bits & 0x7fff_ffff_ffff_ffff)
        } else {
            bits | 0x8000_0000_0000_0000u64
        }
    };
    let oa = to_ordered(a);
    let ob = to_ordered(b);
    oa.abs_diff(ob)
}

/// IEEE-754 ULP distance for f32 (treating -0.0 and +0.0 as distance 0).
pub fn ulp_diff_f32(a: f32, b: f32) -> u32 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let to_ordered = |x: f32| -> u32 {
        let bits = x.to_bits();
        if (bits as i32) < 0 {
            0x8000_0000u32.wrapping_sub(bits & 0x7fff_ffff)
        } else {
            bits | 0x8000_0000u32
        }
    };
    let oa = to_ordered(a);
    let ob = to_ordered(b);
    oa.abs_diff(ob)
}

fn scalar_f64_close(a: f64, b: f64, rtol: f64, atol: f64, equal_nan: bool) -> bool {
    if a.is_nan() || b.is_nan() {
        return equal_nan && a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

pub fn assert_f64_close(
    actual: &[f64],
    expected: &[f64],
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            scalar_f64_close(a, b, rtol, atol, equal_nan),
            "{ctx}: mismatch at index {i}: actual={a:?} (bits=0x{:016x}) vs expected={b:?} (bits=0x{:016x}) [rtol={rtol}, atol={atol}]",
            a.to_bits(),
            b.to_bits()
        );
    }
}

pub fn assert_f32_close(
    actual: &[f32],
    expected: &[f32],
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            scalar_f64_close(a as f64, b as f64, rtol, atol, equal_nan),
            "{ctx}: mismatch at index {i}: actual={a:?} (bits=0x{:08x}) vs expected={b:?} (bits=0x{:08x}) [rtol={rtol}, atol={atol}]",
            a.to_bits(),
            b.to_bits()
        );
    }
}

pub fn assert_f16_close(
    actual: &[f16],
    expected: &[f16],
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            scalar_f64_close(a.to_f64(), b.to_f64(), rtol, atol, equal_nan),
            "{ctx}: f16 mismatch at index {i}: actual={a:?} vs expected={b:?}"
        );
    }
}

pub fn assert_c128_close(
    actual: &[Complex64],
    expected: &[Complex64],
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        let re_ok = scalar_f64_close(a.re, b.re, rtol, atol, equal_nan);
        let im_ok = scalar_f64_close(a.im, b.im, rtol, atol, equal_nan);
        assert!(
            re_ok && im_ok,
            "{ctx}: complex128 mismatch at index {i}: actual={a:?} vs expected={b:?}"
        );
    }
}

pub fn assert_c64_close(
    actual: &[Complex32],
    expected: &[Complex32],
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        let re_ok = scalar_f64_close(a.re as f64, b.re as f64, rtol, atol, equal_nan);
        let im_ok = scalar_f64_close(a.im as f64, b.im as f64, rtol, atol, equal_nan);
        assert!(
            re_ok && im_ok,
            "{ctx}: complex64 mismatch at index {i}: actual={a:?} vs expected={b:?}"
        );
    }
}

/// Assert exact bit equality for floats (checks `-0.0` vs `+0.0`, treats any NaN == any NaN).
pub fn assert_f64_bits_eq(actual: &[f64], expected: &[f64], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{ctx}: bit mismatch at index {i}: actual={a:?} (0x{:016x}) vs expected={b:?} (0x{:016x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

pub fn assert_f32_bits_eq(actual: &[f32], expected: &[f32], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{ctx}: bit mismatch at index {i}: actual={a:?} (0x{:08x}) vs expected={b:?} (0x{:08x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

/// Compare a `DynArray` against a live NumPy `OracleArray` checking dtype, shape, and values.
pub fn assert_dyn_matches_oracle(
    actual: &DynArray,
    expected: &OracleArray,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    check_neg_zero: bool,
    ctx: &str,
) {
    assert_eq!(
        actual.dtype(),
        expected.dtype,
        "{ctx}: dtype mismatch actual={} vs expected={}",
        actual.dtype(),
        expected.dtype
    );
    assert_eq!(
        actual.shape(),
        expected.shape.as_slice(),
        "{ctx}: shape mismatch actual={:?} vs expected={:?}",
        actual.shape(),
        expected.shape
    );
    match actual {
        DynArray::Bool(a) => assert_eq!(a.to_vec(), expected.to_bool_vec(), "{ctx}"),
        DynArray::I8(a) => assert_eq!(a.to_vec(), expected.to_i8_vec(), "{ctx}"),
        DynArray::I16(a) => assert_eq!(a.to_vec(), expected.to_i16_vec(), "{ctx}"),
        DynArray::I32(a) => assert_eq!(a.to_vec(), expected.to_i32_vec(), "{ctx}"),
        DynArray::I64(a) => assert_eq!(a.to_vec(), expected.to_i64_vec(), "{ctx}"),
        DynArray::U8(a) => assert_eq!(a.to_vec(), expected.to_u8_vec(), "{ctx}"),
        DynArray::U16(a) => assert_eq!(a.to_vec(), expected.to_u16_vec(), "{ctx}"),
        DynArray::U32(a) => assert_eq!(a.to_vec(), expected.to_u32_vec(), "{ctx}"),
        DynArray::U64(a) => assert_eq!(a.to_vec(), expected.to_u64_vec(), "{ctx}"),
        DynArray::F16(a) => {
            let av = a.to_vec();
            let ev = expected.to_f16_vec();
            assert_f16_close(&av, &ev, rtol, atol, equal_nan, ctx);
            if check_neg_zero {
                for (i, (&x, &y)) in av.iter().zip(ev.iter()).enumerate() {
                    if x == f16::ZERO && y == f16::ZERO {
                        assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: f16 -0.0 sign mismatch at {i}");
                    }
                }
            }
        }
        DynArray::F32(a) => {
            let av = a.to_vec();
            let ev = expected.to_f32_vec();
            assert_f32_close(&av, &ev, rtol, atol, equal_nan, ctx);
            if check_neg_zero {
                for (i, (&x, &y)) in av.iter().zip(ev.iter()).enumerate() {
                    if x == 0.0 && y == 0.0 {
                        assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: f32 -0.0 sign mismatch at {i}");
                    }
                }
            }
        }
        DynArray::F64(a) => {
            let av = a.to_vec();
            let ev = expected.to_f64_vec();
            assert_f64_close(&av, &ev, rtol, atol, equal_nan, ctx);
            if check_neg_zero {
                for (i, (&x, &y)) in av.iter().zip(ev.iter()).enumerate() {
                    if x == 0.0 && y == 0.0 {
                        assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: f64 -0.0 sign mismatch at {i}");
                    }
                }
            }
        }
        DynArray::C64(a) => {
            assert_c64_close(&a.to_vec(), &expected.to_c64_vec(), rtol, atol, equal_nan, ctx);
        }
        DynArray::C128(a) => {
            assert_c128_close(&a.to_vec(), &expected.to_c128_vec(), rtol, atol, equal_nan, ctx);
        }
    }
}

/// Canonical edge-case f64 test values (signed zeros, subnormals, extremes, Infs, NaN).
pub fn edge_f64_values() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        std::f64::consts::PI,
        -std::f64::consts::PI,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
        1e150,
        -1e150,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]
}

/// Canonical edge-case f32 test values.
pub fn edge_f32_values() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        std::f32::consts::PI,
        -std::f32::consts::PI,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        1e20,
        -1e20,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]
}

/// Canonical edge-case complex128 values (branch cuts, signed zeros, NaNs, Infs).
pub fn edge_c128_values() -> Vec<Complex64> {
    vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(-0.0, 0.0),
        Complex64::new(0.0, -0.0),
        Complex64::new(-0.0, -0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(1.5, -2.5),
        Complex64::new(-3.25, 4.75),
        Complex64::new(f64::INFINITY, 1.0),
        Complex64::new(1.0, f64::NEG_INFINITY),
        Complex64::new(f64::NAN, 1.0),
        Complex64::new(1.0, f64::NAN),
    ]
}

/// Generate a rich matrix of non-contiguous views (F-order, transposed, flipped, strided) from a 2D array.
pub fn non_contiguous_2d_views<T: Scalar>(arr: &NdArray<T>) -> Vec<(&'static str, NdArray<T>)> {
    assert_eq!(arr.ndim(), 2, "expected 2D array");
    vec![
        ("c_contiguous", arr.clone()),
        ("f_contiguous", arr.to_f_contiguous()),
        ("transposed", arr.t()),
        ("flipped_all", arr.flip(None).unwrap()),
        ("flipped_axis0", arr.flip(Some(&[0])).unwrap()),
        ("flipped_axis1", arr.flip(Some(&[1])).unwrap()),
    ]
}

/// Reference 2D row-major matrix multiplication `A (m x k) @ B (k x n)` for invariant verification.
pub fn ref_matmul_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Reference 2D row-major complex128 matrix multiplication `A (m x k) @ B (k x n)`.
pub fn ref_matmul_c128(
    a: &[Complex64],
    b: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Conjugate transpose of an (m x n) row-major complex128 matrix -> (n x m).
pub fn conj_transpose_c128(a: &[Complex64], m: usize, n: usize) -> Vec<Complex64> {
    assert_eq!(a.len(), m * n);
    let mut out = vec![Complex64::new(0.0, 0.0); n * m];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j].conj();
        }
    }
    out
}

/// Transpose of an (m x n) row-major f64 matrix -> (n x m).
pub fn transpose_f64(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * n);
    let mut out = vec![0.0f64; n * m];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// Assert that columns of `(m x k)` matrix `q` are orthonormal: `Q^T @ Q ≈ I_k`.
pub fn assert_cols_orthonormal_f64(q: &[f64], m: usize, k: usize, tol: f64, ctx: &str) {
    let qt = transpose_f64(q, m, k);
    let prod = ref_matmul_f64(&qt, q, k, m, k);
    let mut eye = vec![0.0f64; k * k];
    for i in 0..k {
        eye[i * k + i] = 1.0;
    }
    assert_f64_close(&prod, &eye, tol, tol, false, &format!("{ctx}: Q^T @ Q != I"));
}

/// Assert that columns of `(m x k)` complex matrix `q` are orthonormal: `Q^H @ Q ≈ I_k`.
pub fn assert_cols_orthonormal_c128(q: &[Complex64], m: usize, k: usize, tol: f64, ctx: &str) {
    let qh = conj_transpose_c128(q, m, k);
    let prod = ref_matmul_c128(&qh, q, k, m, k);
    let mut eye = vec![Complex64::new(0.0, 0.0); k * k];
    for i in 0..k {
        eye[i * k + i] = Complex64::new(1.0, 0.0);
    }
    assert_c128_close(&prod, &eye, tol, tol, false, &format!("{ctx}: Q^H @ Q != I"));
}

/// Assert upper-triangular `(m x n)` matrix (zeros strictly below diagonal).
pub fn assert_upper_triangular_f64(r: &[f64], m: usize, n: usize, tol: f64, ctx: &str) {
    assert_eq!(r.len(), m * n);
    for i in 0..m {
        for j in 0..i.min(n) {
            assert!(
                r[i * n + j].abs() <= tol,
                "{ctx}: non-zero below diagonal at ({i}, {j}) = {}",
                r[i * n + j]
            );
        }
    }
}

/// Assert lower-triangular `(m x n)` matrix (zeros strictly above diagonal).
pub fn assert_lower_triangular_f64(l: &[f64], m: usize, n: usize, tol: f64, ctx: &str) {
    assert_eq!(l.len(), m * n);
    for i in 0..m {
        for j in (i + 1)..n {
            assert!(
                l[i * n + j].abs() <= tol,
                "{ctx}: non-zero above diagonal at ({i}, {j}) = {}",
                l[i * n + j]
            );
        }
    }
}

/// Compute empirical (mean, variance) of a slice of f64 samples.
pub fn sample_mean_var(xs: &[f64]) -> (f64, f64) {
    assert!(!xs.is_empty());
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var)
}
