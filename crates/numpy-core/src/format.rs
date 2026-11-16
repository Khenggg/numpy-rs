//! Array pretty-printing (`array2string`, `array_str`, `array_repr`) matching NumPy formatting rules.

use half::f16;
use num_complex::{Complex32, Complex64};
use numpy_types::{DType, DynArray, NdArray, Result, Scalar};

/// Controls how numeric signs are rendered in formatted output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignMode {
    /// '-' for negative values only (NumPy default `'-'`).
    #[default]
    Minus,
    /// '+' for non-negative values, '-' for negative values (`'+'`).
    Plus,
    /// Leading space for non-negative values, '-' for negative values (`' '`).
    Space,
}

/// Controls floating-point digit trimming and padding behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FloatMode {
    /// Shortest unique round-trip representation, trimmed as needed.
    Unique,
    /// Trim trailing zeros to at most `precision` fractional digits and pad with spaces (`maxprec`).
    #[default]
    MaxPrec,
    /// Trim trailing zeros identically across all printed elements so fractional digits have equal width (`maxprec_equal`).
    MaxPrecEqual,
    /// Always emit exact `precision` fractional digits (`fixed`).
    Fixed,
}

/// Formatting configuration matching `numpy.set_printoptions` / `numpy.array2string`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrintOptions {
    /// Number of digits of precision for floating-point output (default: 8).
    pub precision: Option<usize>,
    /// Total number of array elements triggering summarization (default: 1000).
    pub threshold: usize,
    /// Number of array items in summary at beginning and end of each dimension (default: 3).
    pub edgeitems: usize,
    /// Number of characters per line before inserting line breaks (default: 75).
    pub linewidth: usize,
    /// Suppress scientific notation for small floating-point numbers (default: false).
    pub suppress_small: bool,
    /// String representation of floating-point NaN (default: `"nan"`).
    pub nanstr: String,
    /// String representation of floating-point infinity (default: `"inf"`).
    pub infstr: String,
    /// Sign formatting mode (default: `SignMode::Minus`).
    pub sign: SignMode,
    /// Floating-point rounding/trimming mode (default: `FloatMode::MaxPrec`).
    pub floatmode: FloatMode,
    /// Optional separator between elements (default: `" "`).
    pub separator: Option<String>,
    /// Prefix string prepended to the first line (used for indentation width calculation).
    pub prefix: String,
    /// Suffix string appended after the formatted array.
    pub suffix: String,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            precision: Some(8),
            threshold: 1000,
            edgeitems: 3,
            linewidth: 75,
            suppress_small: false,
            nanstr: "nan".to_string(),
            infstr: "inf".to_string(),
            sign: SignMode::Minus,
            floatmode: FloatMode::MaxPrec,
            separator: None,
            prefix: String::new(),
            suffix: String::new(),
        }
    }
}

/// Scalar element formatting contract used by `array2string`.
pub trait FormatElement: Scalar {
    /// Formats a slice of visible elements (ordered across the summary/array) into equal-width strings.
    fn format_elements(values: &[Self], is_scalar: bool, options: &PrintOptions) -> Vec<String>;
}

fn format_sign_char(is_negative: bool, mode: SignMode) -> &'static str {
    if is_negative {
        "-"
    } else {
        match mode {
            SignMode::Minus => "",
            SignMode::Plus => "+",
            SignMode::Space => " ",
        }
    }
}

fn format_signed_int_slice<I: ToString + Ord + Copy>(
    values: &[I],
    zero: I,
    sign_mode: SignMode,
    min_str_fn: fn(I) -> String,
) -> Vec<String> {
    let raw: Vec<String> = values
        .iter()
        .map(|&v| {
            if v < zero {
                min_str_fn(v)
            } else {
                let prefix = format_sign_char(false, sign_mode);
                if prefix.is_empty() {
                    v.to_string()
                } else {
                    format!("{prefix}{}", v.to_string())
                }
            }
        })
        .collect();
    let max_len = raw.iter().map(|s| s.len()).max().unwrap_or(0);
    raw.into_iter()
        .map(|s| format!("{s:>width$}", width = max_len))
        .collect()
}

fn format_unsigned_int_slice<U: ToString + Copy>(
    values: &[U],
    sign_mode: SignMode,
) -> Vec<String> {
    let prefix = format_sign_char(false, sign_mode);
    let raw: Vec<String> = values
        .iter()
        .map(|&v| {
            if prefix.is_empty() {
                v.to_string()
            } else {
                format!("{prefix}{}", v.to_string())
            }
        })
        .collect();
    let max_len = raw.iter().map(|s| s.len()).max().unwrap_or(0);
    raw.into_iter()
        .map(|s| format!("{s:>width$}", width = max_len))
        .collect()
}

impl FormatElement for bool {
    fn format_elements(values: &[Self], is_scalar: bool, _options: &PrintOptions) -> Vec<String> {
        if is_scalar {
            values
                .iter()
                .map(|&v| if v { "True".into() } else { "False".into() })
                .collect()
        } else {
            values
                .iter()
                .map(|&v| if v { " True".into() } else { "False".into() })
                .collect()
        }
    }
}

macro_rules! impl_format_signed {
    ($($ty:ty),*) => {
        $(
            impl FormatElement for $ty {
                fn format_elements(values: &[Self], _is_scalar: bool, options: &PrintOptions) -> Vec<String> {
                    format_signed_int_slice(values, 0, options.sign, |v| v.to_string())
                }
            }
        )*
    };
}

macro_rules! impl_format_unsigned {
    ($($ty:ty),*) => {
        $(
            impl FormatElement for $ty {
                fn format_elements(values: &[Self], _is_scalar: bool, options: &PrintOptions) -> Vec<String> {
                    format_unsigned_int_slice(values, options.sign)
                }
            }
        )*
    };
}

impl_format_signed!(i8, i16, i32, i64);
impl_format_unsigned!(u8, u16, u32, u64);

/// Internal precomputed floating-point format plan (mirrors NumPy `FloatingFormat`).
#[derive(Debug, Clone)]
struct FloatFormatPlan {
    exp_format: bool,
    precision: usize,
    pad_left: usize,
    pad_right: usize,
    trim_trailing_zeros: bool,
    keep_dot: bool,
    exp_size: usize,
    unique_mode: bool,
    suppress_small: bool,
    sign: SignMode,
    nanstr: String,
    infstr: String,
}

fn format_exp_suffix(exp: i32, exp_size: usize) -> String {
    let sign = if exp < 0 { '-' } else { '+' };
    let abs_exp = exp.unsigned_abs();
    format!("e{sign}{abs_exp:0>width$}", width = exp_size)
}

fn trim_frac_zeros(frac: &str) -> &str {
    frac.trim_end_matches('0')
}

impl FloatFormatPlan {
    fn from_f64_values(
        vals: &[f64],
        options: &PrintOptions,
        default_unique_prec: usize,
        pad_right_min: usize,
    ) -> Self {
        let mut non_zero_abs: Vec<f64> = vals
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v != 0.0)
            .map(f64::abs)
            .collect();
        non_zero_abs.sort_by(|a, b| a.total_cmp(b));

        let prec = options.precision.unwrap_or(default_unique_prec);
        let unique_mode = matches!(options.floatmode, FloatMode::Unique) || options.precision.is_none();

        if non_zero_abs.is_empty() {
            // Check whether any NaN or Inf forces a wider pad_left
            let mut max_special_len = 0usize;
            for &v in vals {
                if v.is_nan() {
                    let s_len = options.nanstr.len()
                        + if matches!(options.sign, SignMode::Plus | SignMode::Space) { 1 } else { 0 };
                    max_special_len = max_special_len.max(s_len);
                } else if v.is_infinite() {
                    let s_len = options.infstr.len()
                        + if v.is_sign_negative() || matches!(options.sign, SignMode::Plus | SignMode::Space) {
                            1
                        } else {
                            0
                        };
                    max_special_len = max_special_len.max(s_len);
                }
            }
            let trim = !matches!(options.floatmode, FloatMode::Fixed);
            let pad_right = if matches!(options.floatmode, FloatMode::Fixed) {
                prec
            } else {
                pad_right_min
            };
            let base_left = 1 + usize::from(
                vals.iter().any(|v| v.is_sign_negative())
                    || matches!(options.sign, SignMode::Plus | SignMode::Space),
            );
            return Self {
                exp_format: false,
                precision: prec,
                pad_left: base_left.max(max_special_len.saturating_sub(1 + pad_right)),
                pad_right,
                trim_trailing_zeros: trim,
                keep_dot: true,
                exp_size: 2,
                unique_mode,
                suppress_small: options.suppress_small,
                sign: options.sign,
                nanstr: options.nanstr.clone(),
                infstr: options.infstr.clone(),
            };
        }

        let min_val = non_zero_abs[0];
        let max_val = non_zero_abs[non_zero_abs.len() - 1];

        let mut exp_format = false;
        if max_val >= 1.0e8 {
            exp_format = true;
        } else if !options.suppress_small && min_val < 1.0e-4 {
            if max_val < 1.0e-4 || (max_val / min_val) > 1000.0 {
                exp_format = true;
            }
        }

        let mut exp_size = 2usize;
        if exp_format {
            for &v in vals {
                if v.is_finite() && v != 0.0 {
                    let e = v.abs().log10().floor() as i32;
                    if e.abs() >= 100 {
                        exp_size = exp_size.max(3);
                    }
                }
            }
        }

        // First render each finite value to determine pad_left and pad_right
        let mut max_int_len = 1usize;
        let mut max_frac_len = pad_right_min;
        let mut max_special_len = 0usize;

        for &v in vals {
            if v.is_nan() {
                let s_len = options.nanstr.len()
                    + if matches!(options.sign, SignMode::Plus | SignMode::Space) { 1 } else { 0 };
                max_special_len = max_special_len.max(s_len);
                continue;
            }
            if v.is_infinite() {
                let s_len = options.infstr.len()
                    + if v.is_sign_negative() || matches!(options.sign, SignMode::Plus | SignMode::Space) {
                        1
                    } else {
                        0
                    };
                max_special_len = max_special_len.max(s_len);
                continue;
            }
            let (int_part, frac_part, _) = render_raw_float_parts(
                v,
                exp_format,
                prec,
                unique_mode,
                options.suppress_small,
                options.floatmode,
                options.sign,
                exp_size,
            );
            max_int_len = max_int_len.max(int_part.len());
            max_frac_len = max_frac_len.max(frac_part.len());
        }

        let exp_extra = if exp_format { 2 + exp_size } else { 0 };
        let needed_for_special = max_special_len.saturating_sub(1 + max_frac_len + exp_extra);
        let pad_left = max_int_len.max(needed_for_special);

        Self {
            exp_format,
            precision: prec,
            pad_left,
            pad_right: max_frac_len,
            trim_trailing_zeros: !matches!(options.floatmode, FloatMode::Fixed),
            keep_dot: true,
            exp_size,
            unique_mode,
            suppress_small: options.suppress_small,
            sign: options.sign,
            nanstr: options.nanstr.clone(),
            infstr: options.infstr.clone(),
        }
    }

    fn format_value(&self, v: f64) -> String {
        if v.is_nan() {
            let prefix = format_sign_char(false, self.sign);
            let s = format!("{prefix}{}", self.nanstr);
            let total = self.total_width();
            return format!("{s:>width$}", width = total);
        }
        if v.is_infinite() {
            let prefix = format_sign_char(v.is_sign_negative(), self.sign);
            let s = format!("{prefix}{}", self.infstr);
            let total = self.total_width();
            return format!("{s:>width$}", width = total);
        }

        let (int_part, frac_part, exp_suffix) = render_raw_float_parts(
            v,
            self.exp_format,
            self.precision,
            self.unique_mode,
            self.suppress_small,
            if self.trim_trailing_zeros {
                if matches!(self.precision, 0) {
                    FloatMode::Fixed
                } else {
                    FloatMode::MaxPrec
                }
            } else {
                FloatMode::Fixed
            },
            self.sign,
            self.exp_size,
        );

        let left_padded = format!("{int_part:>width$}", width = self.pad_left);
        let frac_padded = if matches!(self.trim_trailing_zeros, false) {
            format!("{frac_part:0<width$}", width = self.pad_right)
        } else {
            format!("{frac_part:<width$}", width = self.pad_right)
        };
        let dot = if self.keep_dot || !frac_padded.is_empty() {
            "."
        } else {
            ""
        };

        if self.exp_format {
            format!("{left_padded}{dot}{frac_padded}{exp_suffix}")
        } else {
            format!("{left_padded}{dot}{frac_padded}")
        }
    }

    fn format_maxprec_equal(&self, v: f64) -> String {
        if v.is_nan() || v.is_infinite() {
            return self.format_value(v);
        }
        let (int_part, frac_part, exp_suffix) = render_raw_float_parts(
            v,
            self.exp_format,
            self.precision,
            self.unique_mode,
            self.suppress_small,
            FloatMode::MaxPrecEqual,
            self.sign,
            self.exp_size,
        );
        let left_padded = format!("{int_part:>width$}", width = self.pad_left);
        let frac_padded = format!("{frac_part:0<width$}", width = self.pad_right);
        let dot = if self.keep_dot || !frac_padded.is_empty() {
            "."
        } else {
            ""
        };
        if self.exp_format {
            format!("{left_padded}{dot}{frac_padded}{exp_suffix}")
        } else {
            format!("{left_padded}{dot}{frac_padded}")
        }
    }

    fn total_width(&self) -> usize {
        let dot_len = usize::from(self.keep_dot || self.pad_right > 0);
        let exp_len = if self.exp_format { 2 + self.exp_size } else { 0 };
        self.pad_left + dot_len + self.pad_right + exp_len
    }
}

#[allow(clippy::too_many_arguments)]
fn render_raw_float_parts(
    mut v: f64,
    exp_format: bool,
    precision: usize,
    unique_mode: bool,
    suppress_small: bool,
    floatmode: FloatMode,
    sign_mode: SignMode,
    exp_size: usize,
) -> (String, String, String) {
    let is_neg = v.is_sign_negative();
    let sign_str = format_sign_char(is_neg, sign_mode);
    v = v.abs();

    if v == 0.0 {
        let exp_suffix = if exp_format {
            format_exp_suffix(0, exp_size)
        } else {
            String::new()
        };
        let frac = if matches!(floatmode, FloatMode::Fixed) {
            "0".repeat(precision)
        } else {
            String::new()
        };
        return (format!("{sign_str}0"), frac, exp_suffix);
    }

    if suppress_small && !exp_format {
        let threshold = 0.5 * 10f64.powi(-(precision as i32));
        if v < threshold {
            let frac = if matches!(floatmode, FloatMode::Fixed) {
                "0".repeat(precision)
            } else {
                String::new()
            };
            return (format!("{sign_str}0"), frac, String::new());
        }
    }

    if exp_format {
        let s = if unique_mode {
            format!("{v:e}")
        } else {
            format!("{v:.prec$e}", prec = precision)
        };
        let (mantissa, exp_val) = if let Some((m, e_str)) = s.split_once('e') {
            let e_parsed: i32 = e_str.parse().unwrap_or(0);
            (m, e_parsed)
        } else {
            (s.as_str(), 0)
        };
        let (int_digits, raw_frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let frac = match floatmode {
            FloatMode::Fixed => raw_frac.to_string(),
            FloatMode::Unique | FloatMode::MaxPrec | FloatMode::MaxPrecEqual => {
                trim_frac_zeros(raw_frac).to_string()
            }
        };
        return (
            format!("{sign_str}{int_digits}"),
            frac,
            format_exp_suffix(exp_val, exp_size),
        );
    }

    let formatted = if unique_mode {
        format!("{v}")
    } else {
        format!("{v:.prec$}", prec = precision)
    };
    let (int_digits, raw_frac) = formatted.split_once('.').unwrap_or((formatted.as_str(), ""));
    let frac = match floatmode {
        FloatMode::Fixed => raw_frac.to_string(),
        FloatMode::Unique | FloatMode::MaxPrec | FloatMode::MaxPrecEqual => {
            trim_frac_zeros(raw_frac).to_string()
        }
    };
    (format!("{sign_str}{int_digits}"), frac, String::new())
}

fn format_f64_slice_impl(
    vals: &[f64],
    options: &PrintOptions,
    default_unique_prec: usize,
) -> Vec<String> {
    let plan = FloatFormatPlan::from_f64_values(vals, options, default_unique_prec, 0);
    vals.iter()
        .map(|&v| {
            if matches!(options.floatmode, FloatMode::MaxPrecEqual) {
                plan.format_maxprec_equal(v)
            } else {
                plan.format_value(v)
            }
        })
        .collect()
}

impl FormatElement for f16 {
    fn format_elements(values: &[Self], _is_scalar: bool, options: &PrintOptions) -> Vec<String> {
        let vals: Vec<f64> = values.iter().map(|v| f64::from(*v)).collect();
        format_f64_slice_impl(&vals, options, 4)
    }
}

impl FormatElement for f32 {
    fn format_elements(values: &[Self], _is_scalar: bool, options: &PrintOptions) -> Vec<String> {
        let vals: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
        format_f64_slice_impl(&vals, options, 7)
    }
}

impl FormatElement for f64 {
    fn format_elements(values: &[Self], _is_scalar: bool, options: &PrintOptions) -> Vec<String> {
        format_f64_slice_impl(values, options, 16)
    }
}

fn format_complex_slice_impl(
    reals: &[f64],
    imags: &[f64],
    options: &PrintOptions,
    default_unique_prec: usize,
) -> Vec<String> {
    let real_plan = FloatFormatPlan::from_f64_values(reals, options, default_unique_prec, 0);
    let mut imag_opts = options.clone();
    imag_opts.sign = SignMode::Plus;
    let imag_plan = FloatFormatPlan::from_f64_values(imags, &imag_opts, default_unique_prec, 0);

    let mut out = Vec::with_capacity(reals.len());
    for (&r, &i) in reals.iter().zip(imags.iter()) {
        let rs = if matches!(options.floatmode, FloatMode::MaxPrecEqual) {
            real_plan.format_maxprec_equal(r)
        } else {
            real_plan.format_value(r)
        };
        let mut is = if matches!(options.floatmode, FloatMode::MaxPrecEqual) {
            imag_plan.format_maxprec_equal(i)
        } else {
            imag_plan.format_value(i)
        };
        // NumPy moves leading padding spaces on the imaginary part to the end after 'j'
        // Wait: NumPy actually formats complex as `r_str + i_str + 'j'` where i_str has sign stripped of leading spaces if needed?
        // In NumPy `ComplexFloatingFormat`: `imag_format` uses `sign='+'` and right-aligns, then appends `'j'`, wait: `np.array([1+2j, 10+20j])` -> `'[ 1. +2.j 10.+20.j]'`.
        // Note how `' +2.j'` would have space before `+`, or NumPy places `'j'` immediately after trailing pad? Let's keep NumPy style: strip internal leading space before sign by left-justifying `trim_start()` if needed, OR pad trailing spaces after `'j'`.
        // Actually in NumPy: `fmt_real + fmt_imag + 'j'`, where `ComplexFloatingFormat` sets `pad_left = 1` on `imag_format`? Wait: `np.array([1.+2.j, 1.+20.j])` -> `'[1. +2.j 1.+20.j]'`! Look: space is between `1.` and `+2.j` from `pad_left`!
        let _ = &mut is;
        out.push(format!("{rs}{is}j"));
    }
    out
}

impl FormatElement for Complex32 {
    fn format_elements(values: &[Self], _is_scalar: bool, options: &PrintOptions) -> Vec<String> {
        let reals: Vec<f64> = values.iter().map(|v| f64::from(v.re)).collect();
        let imags: Vec<f64> = values.iter().map(|v| f64::from(v.im)).collect();
        format_complex_slice_impl(&reals, &imags, options, 7)
    }
}

impl FormatElement for Complex64 {
    fn format_elements(values: &[Self], _is_scalar: bool, options: &PrintOptions) -> Vec<String> {
        let reals: Vec<f64> = values.iter().map(|v| v.re).collect();
        let imags: Vec<f64> = values.iter().map(|v| v.im).collect();
        format_complex_slice_impl(&reals, &imags, options, 16)
    }
}

/// Tree node representing either a summarized ellipsis, a scalar index into formatted strings, or nested axes.
enum PrintTree {
    Ellipsis,
    Item(usize),
    List(Vec<PrintTree>),
}

fn collect_summary_tree<T: Copy>(
    arr: &NdArray<T>,
    coords: &mut Vec<isize>,
    axis: usize,
    summarize: bool,
    edgeitems: usize,
    visible: &mut Vec<T>,
) -> Result<PrintTree> {
    let ndim = arr.ndim();
    if axis == ndim {
        let idx = visible.len();
        let val = arr.get(coords)?;
        visible.push(val);
        return Ok(PrintTree::Item(idx));
    }

    let dim = arr.shape()[axis];
    let mut children = Vec::new();
    if summarize && dim > 2 * edgeitems {
        for i in 0..edgeitems {
            coords[axis] = i as isize;
            children.push(collect_summary_tree(
                arr,
                coords,
                axis + 1,
                summarize,
                edgeitems,
                visible,
            )?);
        }
        children.push(PrintTree::Ellipsis);
        for i in (dim - edgeitems)..dim {
            coords[axis] = i as isize;
            children.push(collect_summary_tree(
                arr,
                coords,
                axis + 1,
                summarize,
                edgeitems,
                visible,
            )?);
        }
    } else {
        for i in 0..dim {
            coords[axis] = i as isize;
            children.push(collect_summary_tree(
                arr,
                coords,
                axis + 1,
                summarize,
                edgeitems,
                visible,
            )?);
        }
    }
    Ok(PrintTree::List(children))
}

fn format_tree_recursive(
    node: &PrintTree,
    formatted: &[String],
    next_line_prefix: &str,
    separator: &str,
    axes_left: usize,
    linewidth: usize,
) -> String {
    match node {
        PrintTree::Item(idx) => formatted[*idx].clone(),
        PrintTree::Ellipsis => "...".to_string(),
        PrintTree::List(children) => {
            if axes_left == 0 || children.is_empty() {
                return "[]".to_string();
            }
            let inner_prefix = format!("{next_line_prefix} ");
            if matches!(children[0], PrintTree::Item(_) | PrintTree::Ellipsis) && axes_left == 1 {
                let mut out = String::from("[");
                let mut col = next_line_prefix.len() + 1;
                let sep_len = separator.len();
                let sep_no_trail = separator.trim_end();
                for (i, child) in children.iter().enumerate() {
                    let token = match child {
                        PrintTree::Item(idx) => formatted[*idx].as_str(),
                        PrintTree::Ellipsis => "...",
                        PrintTree::List(_) => "",
                    };
                    if i == 0 {
                        out.push_str(token);
                        col += token.len();
                    } else if col + sep_len + token.len() > linewidth {
                        out.push_str(sep_no_trail);
                        out.push('\n');
                        out.push_str(&inner_prefix);
                        out.push_str(token);
                        col = inner_prefix.len() + token.len();
                    } else {
                        out.push_str(separator);
                        out.push_str(token);
                        col += sep_len + token.len();
                    }
                }
                out.push(']');
                out
            } else {
                let mut out = String::from("[");
                let trailing_sep = separator.trim_end();
                let gap = "\n".repeat(axes_left - 1);
                for (idx, child) in children.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(trailing_sep);
                        out.push_str(&gap);
                        out.push_str(&inner_prefix);
                    }
                    let rendered = format_tree_recursive(
                        child,
                        formatted,
                        &inner_prefix,
                        separator,
                        axes_left - 1,
                        linewidth,
                    );
                    out.push_str(&rendered);
                }
                out.push(']');
                out
            }
        }
    }
}

/// Formats an `NdArray<T>` into a NumPy-style string representation.
pub fn array2string<T: FormatElement>(arr: &NdArray<T>, options: &PrintOptions) -> Result<String> {
    if arr.ndim() == 0 {
        let val = arr.get(&[])?;
        let formatted = T::format_elements(&[val], true, options);
        return Ok(formatted.into_iter().next().unwrap_or_default());
    }

    if arr.is_empty() {
        return Ok("[]".to_string());
    }

    let summarize = arr.len() > options.threshold;
    let mut coords = vec![0isize; arr.ndim()];
    let mut visible = Vec::new();
    let tree = collect_summary_tree(
        arr,
        &mut coords,
        0,
        summarize,
        options.edgeitems,
        &mut visible,
    )?;

    let formatted = T::format_elements(&visible, false, options);
    let separator = options.separator.as_deref().unwrap_or(" ");
    let prefix_spaces = " ".repeat(options.prefix.len());
    Ok(format_tree_recursive(
        &tree,
        &formatted,
        &prefix_spaces,
        separator,
        arr.ndim(),
        options.linewidth,
    ))
}

/// Formats an `NdArray<T>` using default `PrintOptions`.
pub fn array2string_default<T: FormatElement>(arr: &NdArray<T>) -> Result<String> {
    array2string(arr, &PrintOptions::default())
}

/// Formats an `NdArray<T>` using NumPy's `array_str` style.
pub fn array_str<T: FormatElement>(arr: &NdArray<T>, options: &PrintOptions) -> Result<String> {
    array2string(arr, options)
}

/// Formats an `NdArray<T>` using NumPy's `array_repr` style (`array(...)`).
pub fn array_repr<T: FormatElement>(arr: &NdArray<T>, options: &PrintOptions) -> Result<String> {
    let prefix = "array(";
    if arr.is_empty() {
        if arr.ndim() == 1 {
            return Ok(format!("array([], dtype={})", arr.dtype().name()));
        }
        return Ok(format!(
            "array([], shape=({}), dtype={})",
            arr.shape()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            arr.dtype().name()
        ));
    }
    let mut opts = options.clone();
    opts.prefix = prefix.to_string();
    opts.separator = Some(", ".to_string());
    let body = array2string(arr, &opts)?;
    let skip_dtype = matches!(
        arr.dtype(),
        DType::Bool | DType::I64 | DType::F64 | DType::C128
    );
    if skip_dtype {
        Ok(format!("{prefix}{body})"))
    } else {
        Ok(format!("{prefix}{body}, dtype={})", arr.dtype().name()))
    }
}

/// Formats any `DynArray` into a NumPy-style string representation.
pub fn dyn_array2string(arr: &DynArray, options: &PrintOptions) -> Result<String> {
    match arr {
        DynArray::Bool(a) => array2string(a, options),
        DynArray::I8(a) => array2string(a, options),
        DynArray::I16(a) => array2string(a, options),
        DynArray::I32(a) => array2string(a, options),
        DynArray::I64(a) => array2string(a, options),
        DynArray::U8(a) => array2string(a, options),
        DynArray::U16(a) => array2string(a, options),
        DynArray::U32(a) => array2string(a, options),
        DynArray::U64(a) => array2string(a, options),
        DynArray::F16(a) => array2string(a, options),
        DynArray::F32(a) => array2string(a, options),
        DynArray::F64(a) => array2string(a, options),
        DynArray::C64(a) => array2string(a, options),
        DynArray::C128(a) => array2string(a, options),
    }
}

/// Formats any `DynArray` using default `PrintOptions`.
pub fn dyn_array2string_default(arr: &DynArray) -> Result<String> {
    dyn_array2string(arr, &PrintOptions::default())
}

/// Formats any `DynArray` using NumPy's `array_str` style.
pub fn dyn_array_str(arr: &DynArray, options: &PrintOptions) -> Result<String> {
    dyn_array2string(arr, options)
}

/// Formats any `DynArray` using NumPy's `array_repr` style.
pub fn dyn_array_repr(arr: &DynArray, options: &PrintOptions) -> Result<String> {
    match arr {
        DynArray::Bool(a) => array_repr(a, options),
        DynArray::I8(a) => array_repr(a, options),
        DynArray::I16(a) => array_repr(a, options),
        DynArray::I32(a) => array_repr(a, options),
        DynArray::I64(a) => array_repr(a, options),
        DynArray::U8(a) => array_repr(a, options),
        DynArray::U16(a) => array_repr(a, options),
        DynArray::U32(a) => array_repr(a, options),
        DynArray::U64(a) => array_repr(a, options),
        DynArray::F16(a) => array_repr(a, options),
        DynArray::F32(a) => array_repr(a, options),
        DynArray::F64(a) => array_repr(a, options),
        DynArray::C64(a) => array_repr(a, options),
        DynArray::C128(a) => array_repr(a, options),
    }
}
