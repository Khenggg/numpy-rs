//! Real polynomial arithmetic. Legacy `poly*` functions use descending powers;
//! the series types use ascending coefficients and an explicit basis.
use crate::lanes::invalid;
use numpy_types::{Complex64, NdArray, Result};
use std::ops::{Add, Mul, Sub};

fn trim(mut c: Vec<f64>) -> Vec<f64> {
    let first = c
        .iter()
        .position(|&v| v != 0.0)
        .unwrap_or(c.len().saturating_sub(1));
    if first > 0 {
        c.drain(..first);
    }
    if c.is_empty() {
        c.push(0.0);
    }
    c
}
pub fn polyval(coefficients: &[f64], x: &NdArray<f64>) -> NdArray<f64> {
    x.map(|x| coefficients.iter().fold(0.0, |y, &c| y * x + c))
}
pub fn polyadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    let mut c = vec![0.; n];
    for (i, &x) in a.iter().enumerate() {
        c[n - a.len() + i] += x;
    }
    for (i, &x) in b.iter().enumerate() {
        c[n - b.len() + i] += x;
    }
    c
}
pub fn polysub(a: &[f64], b: &[f64]) -> Vec<f64> {
    polyadd(a, &b.iter().map(|x| -x).collect::<Vec<_>>())
}
pub fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    // FIX(F-A3-09): NumPy returns max(len(a)+len(b), 1) zeros when either input is empty.
    if a.is_empty() || b.is_empty() {
        return vec![0.; (a.len() + b.len()).max(1)];
    }
    let mut c = vec![0.; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            c[i + j] += x * y;
        }
    }
    trim(c)
}
pub fn polydiv(a: &[f64], b: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    // FIX(F-A3-04): preserve leading zeros in inputs/quotient (trim only remainder)
    if a.is_empty() || b.is_empty() {
        return Err(invalid("empty polynomial in polydiv"));
    }
    if b.iter().all(|&v| v == 0.) {
        return Err(invalid("polynomial division by zero"));
    }
    let m = a.len() - 1;
    let n = b.len() - 1;
    let q_len = (m + 1).saturating_sub(n).max(1);
    let mut q = vec![0.; q_len];
    let mut r = a.to_vec();
    if a.len() >= b.len() {
        let scale = 1.0 / b[0];
        for k in 0..=(m - n) {
            let d = scale * r[k];
            q[k] = d;
            for j in 0..b.len() {
                r[k + j] -= d * b[j];
            }
        }
    }
    Ok((q, trim(r)))
}
pub fn polyder(coefficients: &[f64], order: usize) -> Vec<f64> {
    let mut c = coefficients.to_vec();
    for _ in 0..order.min(c.len()) {
        let n = c.len();
        c = c
            .iter()
            .take(n.saturating_sub(1))
            .enumerate()
            .map(|(i, &v)| v * (n - 1 - i) as f64)
            .collect();
    }
    c
}
pub fn polyint(coefficients: &[f64], order: usize, constants: &[f64]) -> Result<Vec<f64>> {
    // FIX(F-A3-05): allow constants.len() > order (use first `order` constants)
    let k = if constants.is_empty() { None } else { Some(constants) };
    polyint_with_constants(coefficients, order, k)
}
pub fn polyint_with_constants(
    coefficients: &[f64],
    order: usize,
    constants: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let constants = match constants {
        None => &[][..],
        Some(k) => {
            if order > 0 && k.len() != 1 && k.len() < order {
                return Err(invalid(
                    "integration constants must have length 1 or >= order",
                ));
            }
            k
        }
    };
    let mut c = coefficients.to_vec();
    c.len()
        .checked_add(order)
        .ok_or_else(|| invalid("polynomial size overflow"))?;
    for i in 0..order {
        let n = c.len();
        for (j, v) in c.iter_mut().enumerate() {
            *v /= (n - j) as f64;
        }
        c.push(if constants.len() == 1 {
            constants[0]
        } else {
            constants.get(i).copied().unwrap_or(0.)
        });
    }
    Ok(c)
}
/// Coefficients of the monic polynomial with the supplied real roots.
pub fn poly(roots: &[f64]) -> Vec<f64> {
    roots.iter().fold(vec![1.], |c, &r| polymul(&c, &[1., -r]))
}
pub fn roots(coefficients: &[f64]) -> Result<NdArray<Complex64>> {
    let c = trim(coefficients.to_vec());
    if c.len() <= 1 {
        return Ok(NdArray::from_vec(vec![]));
    }
    if c.iter().any(|v| !v.is_finite()) {
        return Err(invalid("root coefficients must be finite"));
    }
    let n = c.len() - 1;
    let len = n
        .checked_mul(n)
        .ok_or_else(|| invalid("companion matrix size overflow"))?;
    let mut companion = vec![0.; len];
    for j in 0..n {
        companion[j] = -c[j + 1] / c[0];
    }
    for i in 1..n {
        companion[i * n + i - 1] = 1.;
    }
    numpy_linalg::eigvals(&NdArray::from_shape_vec(&[n, n], companion)?)
}
/// Scaled Vandermonde least-squares polynomial fit (descending coefficients).
pub fn polyfit(
    x: &NdArray<f64>,
    y: &NdArray<f64>,
    degree: usize,
    rcond: Option<f64>,
) -> Result<NdArray<f64>> {
    if x.ndim() != 1 || x.is_empty() || y.ndim() != 1 || y.len() != x.len() {
        return Err(invalid("polyfit requires nonempty vectors of equal length"));
    }
    let cols = degree
        .checked_add(1)
        .ok_or_else(|| invalid("polynomial degree overflow"))?;
    let len = x
        .len()
        .checked_mul(cols)
        .ok_or_else(|| invalid("Vandermonde size overflow"))?;
    let mut data = vec![0.; len];
    for (i, x) in x.iter().enumerate() {
        let mut v = 1.;
        for j in (0..cols).rev() {
            data[i * cols + j] = v;
            v *= x;
        }
    }
    let mut scales = vec![0.; cols];
    for j in 0..cols {
        scales[j] = (0..x.len())
            .map(|i| data[i * cols + j] * data[i * cols + j])
            .sum::<f64>()
            .sqrt();
        if scales[j] == 0. {
            scales[j] = 1.;
        }
        for i in 0..x.len() {
            data[i * cols + j] /= scales[j];
        }
    }
    let out = numpy_linalg::lstsq(
        &NdArray::from_shape_vec(&[x.len(), cols], data)?,
        y,
        rcond.or(Some(x.len() as f64 * f64::EPSILON)),
    )?;
    Ok(NdArray::from_vec(
        out.x.iter().zip(scales).map(|(v, s)| v / s).collect(),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    Power,
    Chebyshev,
    Legendre,
    Laguerre,
    Hermite,
    HermiteE,
}
/// Basis recurrence `P_(n+1) = (a*x+b) P_n - c P_(n-1)`.
fn recurrence(basis: Basis, n: usize) -> (f64, f64, f64) {
    let n = n as f64;
    match basis {
        Basis::Power => (1., 0., 0.),
        Basis::Chebyshev => (2., 0., 1.),
        Basis::Legendre => ((2. * n + 1.) / (n + 1.), 0., n / (n + 1.)),
        Basis::Laguerre => (-1. / (n + 1.), (2. * n + 1.) / (n + 1.), n / (n + 1.)),
        Basis::Hermite => (2., 0., 2. * n),
        Basis::HermiteE => (1., 0., n),
    }
}
fn first(basis: Basis, x: f64) -> f64 {
    match basis {
        Basis::Laguerre => 1. - x,
        Basis::Hermite => 2. * x,
        _ => x,
    }
}
fn trim_ascending(mut c: Vec<f64>) -> Vec<f64> {
    while c.len() > 1 && c.last().is_some_and(|&v| v == 0.0) {
        c.pop();
    }
    if c.is_empty() {
        c.push(0.0);
    }
    c
}
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialSeries {
    pub coefficients: Vec<f64>,
    pub basis: Basis,
}
impl PolynomialSeries {
    pub fn new(coefficients: Vec<f64>, basis: Basis) -> Self {
        let coefficients = if coefficients.is_empty() {
            vec![0.0]
        } else {
            coefficients
        };
        Self {
            coefficients,
            basis,
        }
    }
    pub fn eval(&self, x: f64) -> f64 {
        if self.basis == Basis::Power {
            return self.coefficients.iter().rev().fold(0., |s, &c| s * x + c);
        }
        let mut previous = 1.;
        let mut current = first(self.basis, x);
        let mut out = self.coefficients.first().copied().unwrap_or(0.);
        if let Some(&c) = self.coefficients.get(1) {
            out += c * current;
        }
        for n in 1..self.coefficients.len().saturating_sub(1) {
            let (a, b, c) = recurrence(self.basis, n);
            let next = (a * x + b) * current - c * previous;
            out += self.coefficients[n + 1] * next;
            previous = current;
            current = next;
        }
        out
    }
    pub fn evaluate(&self, x: &NdArray<f64>) -> NdArray<f64> {
        x.map(|x| self.eval(x))
    }
    /// Convert to a power-series polynomial, retaining ascending coefficients.
    pub fn to_polynomial(&self) -> Polynomial {
        let mut previous = vec![1.];
        let mut current = vec![
            first(self.basis, 0.),
            first(self.basis, 1.) - first(self.basis, 0.),
        ];
        let mut result = vec![0.; self.coefficients.len().max(1)];
        for (n, &coefficient) in self.coefficients.iter().enumerate() {
            let p = if n == 0 { &previous } else { &current };
            for (i, &v) in p.iter().enumerate() {
                result[i] += coefficient * v;
            }
            if n >= 1 {
                let (a, b, c) = recurrence(self.basis, n);
                let mut next = vec![0.; current.len() + 1];
                for (i, &v) in current.iter().enumerate() {
                    next[i] += b * v;
                    next[i + 1] += a * v;
                }
                for (i, &v) in previous.iter().enumerate() {
                    next[i] -= c * v;
                }
                previous = current;
                current = next;
            }
        }
        Polynomial::new(result)
    }
    /// Convert a power-series [`Polynomial`] into this basis via Horner's method.
    pub fn from_polynomial(poly: &Polynomial, basis: Basis) -> Self {
        let coeffs = poly.coefficients();
        if basis == Basis::Power {
            return Self::new(coeffs.to_vec(), basis);
        }
        let mut res: Vec<f64> = Vec::with_capacity(coeffs.len());
        for &p in coeffs.iter().rev() {
            if !res.is_empty() {
                let mut next = vec![0.0; res.len() + 1];
                match basis {
                    Basis::Laguerre => {
                        next[0] += res[0];
                        next[1] -= res[0];
                    }
                    Basis::Hermite => {
                        next[1] += 0.5 * res[0];
                    }
                    _ => {
                        next[1] += res[0];
                    }
                }
                for k in 1..res.len() {
                    let s = res[k];
                    let (a, b, c) = recurrence(basis, k);
                    next[k + 1] += s / a;
                    next[k] -= s * b / a;
                    next[k - 1] += s * c / a;
                }
                res = next;
            }
            if res.is_empty() {
                res.push(p);
            } else {
                res[0] += p;
            }
        }
        Self::new(res, basis)
    }
}
macro_rules! series {
    ($name:ident,$basis:ident) => {
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name(pub PolynomialSeries);
        impl $name {
            pub fn new(coefficients: Vec<f64>) -> Self {
                Self(PolynomialSeries::new(coefficients, Basis::$basis))
            }
            pub fn eval(&self, x: f64) -> f64 {
                self.0.eval(x)
            }
            pub fn evaluate(&self, x: &NdArray<f64>) -> NdArray<f64> {
                self.0.evaluate(x)
            }
            pub fn coefficients(&self) -> &[f64] {
                &self.0.coefficients
            }
            pub fn to_polynomial(&self) -> Polynomial {
                self.0.to_polynomial()
            }
            pub fn from_polynomial(poly: &Polynomial) -> Self {
                Self(PolynomialSeries::from_polynomial(poly, Basis::$basis))
            }
        }
    };
}
series!(Polynomial, Power);
series!(Chebyshev, Chebyshev);
series!(Legendre, Legendre);
series!(Laguerre, Laguerre);
series!(Hermite, Hermite);
series!(HermiteE, HermiteE);
impl Polynomial {
    pub fn deriv(&self, order: usize) -> Self {
        let mut c = self.coefficients().to_vec();
        c.reverse();
        let mut c = polyder(&c, order);
        c.reverse();
        Self::new(c)
    }
    pub fn integ(&self, order: usize, constants: &[f64]) -> Result<Self> {
        if constants.len() > order {
            return Err(invalid("too many integration constants"));
        }
        if order == 0 {
            return Ok(self.clone());
        }
        let mut c = self.coefficients().to_vec();
        c.len()
            .checked_add(order)
            .ok_or_else(|| invalid("polynomial size overflow"))?;
        for i in 0..order {
            let k = constants.get(i).copied().unwrap_or(0.0);
            if c.len() == 1 && c[0] == 0.0 {
                c[0] += k;
            } else {
                let mut next = Vec::with_capacity(c.len() + 1);
                next.push(k);
                for (deg, &coeff) in c.iter().enumerate() {
                    next.push(coeff / (deg + 1) as f64);
                }
                c = next;
            }
        }
        Ok(Self::new(c))
    }
    pub fn roots(&self) -> Result<NdArray<Complex64>> {
        let mut c = self.coefficients().to_vec();
        c.reverse();
        roots(&c)
    }
    /// Least-squares fit in the mapped `[-1, 1]` domain, converted back to ascending power basis.
    pub fn fit(
        x: &NdArray<f64>,
        y: &NdArray<f64>,
        degree: usize,
        rcond: Option<f64>,
    ) -> Result<Self> {
        if x.ndim() != 1 || x.is_empty() || y.ndim() != 1 || y.len() != x.len() {
            return Err(invalid(
                "Polynomial::fit requires nonempty vectors of equal length",
            ));
        }
        let xv = x.to_vec();
        let yv = y.to_vec();
        if xv.iter().any(|v| !v.is_finite()) || yv.iter().any(|v| !v.is_finite()) {
            return Err(invalid("Polynomial::fit inputs must be finite"));
        }
        let mut lo = xv.iter().copied().fold(f64::INFINITY, f64::min);
        let mut hi = xv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo == hi {
            lo -= 1.0;
            hi += 1.0;
        }
        let mid = 0.5 * lo + 0.5 * hi;
        let half = 0.5 * hi - 0.5 * lo;
        let scl = 1.0 / half;
        let off = -mid * scl;
        let u = NdArray::from_vec(xv.iter().map(|&v| off + scl * v).collect());
        let fit_desc = polyfit(&u, y, degree, rcond)?;
        let c_u: Vec<f64> = fit_desc.iter().collect::<Vec<_>>().into_iter().rev().collect();
        let lin = Self::new(vec![off, scl]);
        let mut acc = Self::new(vec![0.0]);
        for &coef in c_u.iter().rev() {
            acc = &(&acc * &lin) + &Self::new(vec![coef]);
        }
        Ok(acc)
    }
    fn add_slice(&self, other: &Self) -> Self {
        let a = self.coefficients();
        let b = other.coefficients();
        let n = a.len().max(b.len());
        let mut out = vec![0.0; n];
        for (i, &v) in a.iter().enumerate() {
            out[i] += v;
        }
        for (i, &v) in b.iter().enumerate() {
            out[i] += v;
        }
        Self::new(trim_ascending(out))
    }
    fn sub_slice(&self, other: &Self) -> Self {
        let a = self.coefficients();
        let b = other.coefficients();
        let n = a.len().max(b.len());
        let mut out = vec![0.0; n];
        for (i, &v) in a.iter().enumerate() {
            out[i] += v;
        }
        for (i, &v) in b.iter().enumerate() {
            out[i] -= v;
        }
        Self::new(trim_ascending(out))
    }
    fn mul_slice(&self, other: &Self) -> Self {
        let a = self.coefficients();
        let b = other.coefficients();
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                out[i + j] += x * y;
            }
        }
        Self::new(trim_ascending(out))
    }
}
macro_rules! impl_poly_op {
    ($trait:ident, $method:ident, $helper:ident) => {
        impl $trait<Polynomial> for Polynomial {
            type Output = Polynomial;
            fn $method(self, rhs: Polynomial) -> Polynomial {
                self.$helper(&rhs)
            }
        }
        impl $trait<&Polynomial> for Polynomial {
            type Output = Polynomial;
            fn $method(self, rhs: &Polynomial) -> Polynomial {
                self.$helper(rhs)
            }
        }
        impl $trait<Polynomial> for &Polynomial {
            type Output = Polynomial;
            fn $method(self, rhs: Polynomial) -> Polynomial {
                self.$helper(&rhs)
            }
        }
        impl $trait<&Polynomial> for &Polynomial {
            type Output = Polynomial;
            fn $method(self, rhs: &Polynomial) -> Polynomial {
                self.$helper(rhs)
            }
        }
    };
}
impl_poly_op!(Add, add, add_slice);
impl_poly_op!(Sub, sub, sub_slice);
impl_poly_op!(Mul, mul, mul_slice);

macro_rules! impl_ortho_methods {
    ($($name:ident),*) => {$(
        impl $name {
            pub fn deriv(&self, order: usize) -> Self {
                Self::from_polynomial(&self.to_polynomial().deriv(order))
            }
            pub fn integ(&self, order: usize, constants: &[f64]) -> Result<Self> {
                Ok(Self::from_polynomial(&self.to_polynomial().integ(order, constants)?))
            }
            pub fn roots(&self) -> Result<NdArray<Complex64>> {
                self.to_polynomial().roots()
            }
        }
    )*};
}
impl_ortho_methods!(Chebyshev, Legendre, Laguerre, Hermite, HermiteE);
#[derive(Clone, Debug)]
pub struct Poly1d {
    pub coefficients: Vec<f64>,
}
impl Poly1d {
    pub fn new(coefficients: Vec<f64>) -> Self {
        Self {
            coefficients: trim(coefficients),
        }
    }
    pub fn eval(&self, x: f64) -> f64 {
        self.coefficients.iter().fold(0., |y, &c| y * x + c)
    }
    pub fn deriv(&self, n: usize) -> Self {
        Self::new(polyder(&self.coefficients, n))
    }
    pub fn integ(&self, n: usize, k: &[f64]) -> Result<Self> {
        Ok(Self::new(polyint(&self.coefficients, n, k)?))
    }
    pub fn roots(&self) -> Result<NdArray<Complex64>> {
        roots(&self.coefficients)
    }
}
pub fn poly1d(coefficients: Vec<f64>) -> Poly1d {
    Poly1d::new(coefficients)
}
