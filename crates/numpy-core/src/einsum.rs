//! Einstein summation (`einsum`, `einsum_path`, `dyn_einsum`) with greedy and optimal
//! contraction-path planning, single-operand zero-copy view optimization, cache-blocked
//! auto-vectorizing `i-k-j` contractions, and Rayon parallelization.

use crate::creation::CreationElement;
use crate::indexing::diagonal;
use crate::util::{RAYON_THRESHOLD, finish_c_array};
use half::f16;
use num_complex::{Complex32, Complex64};
use numpy_types::{
    shape_size, AlignedBuffer, Casting, DType, DynArray, NdArray, NumPyError, Result,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// =============================================================================
// 1. Public Types & Trait
// =============================================================================

/// Contraction-order optimization strategy matching `numpy.einsum(..., optimize=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EinsumOptimize {
    /// Evaluate naively without pairwise contraction reordering (`optimize=False`).
    None,
    /// Fast polynomial greedy path (`optimize=True` or `optimize='greedy'`).
    #[default]
    Greedy,
    /// Exhaustive pruned search for minimal FLOP cost (`optimize='optimal'`).
    Optimal,
    /// Use `Optimal` for <= 4 operands and `Greedy` otherwise (`optimize='auto'`).
    Auto,
}

/// Metadata describing a single contraction step in an `EinsumPathInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumContraction {
    /// Indices into the remaining operand list contracted at this step.
    pub operands: Vec<usize>,
    /// Index labels summed away during this step.
    pub contraction_labels: String,
    /// Contraction subscript expression for this step (e.g. `"ij,jk->ik"`).
    pub subscript: String,
    /// Result shape produced by this step.
    pub output_shape: Vec<usize>,
    /// Whether this step maps cleanly to BLAS-like GEMM/dot (no Hadamard/broadcast repeats).
    pub can_blas: bool,
}

/// Complete contraction plan and cost report returned by `einsum_path`.
#[derive(Debug, Clone)]
pub struct EinsumPathInfo {
    /// Contraction index sequence compatible with `numpy.einsum_path()[0]`.
    pub path: Vec<Vec<usize>>,
    /// Per-step contraction details.
    pub contractions: Vec<EinsumContraction>,
    /// Input shapes after ellipsis expansion.
    pub input_shapes: Vec<Vec<usize>>,
    /// Naive single-loop FLOP count.
    pub naive_cost: f64,
    /// Optimized path FLOP count.
    pub opt_cost: f64,
    /// `naive_cost / opt_cost.max(1.0)`.
    pub speedup: f64,
    /// Largest intermediate array element count across all contraction steps.
    pub largest_intermediate: usize,
    /// Human-readable formatted path summary matching NumPy's `einsum_path` print output.
    pub report: String,
}

impl fmt::Display for EinsumPathInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report)
    }
}

/// Trait providing einsum multiplication/accumulation across all 14 scalar dtypes.
/// Matches NumPy semantics where `bool` sums using logical OR and multiplies using logical AND.
pub trait EinsumElement: CreationElement {
    /// Fused multiply-accumulate step `acc = acc + a * b`.
    fn einsum_fma(acc: Self, a: Self, b: Self) -> Self;
}

impl EinsumElement for bool {
    #[inline]
    fn einsum_fma(acc: Self, a: Self, b: Self) -> Self {
        acc || (a && b)
    }
}

macro_rules! impl_einsum_numeric {
    ($($ty:ty),*) => {
        $(
            impl EinsumElement for $ty {
                #[inline]
                fn einsum_fma(acc: Self, a: Self, b: Self) -> Self {
                    acc.wrapping_add(a.wrapping_mul(b))
                }
            }
        )*
    };
}

impl_einsum_numeric!(i8, i16, i32, i64, u8, u16, u32, u64);

impl EinsumElement for f16 {
    #[inline]
    fn einsum_fma(acc: Self, a: Self, b: Self) -> Self {
        acc + a * b
    }
}

impl EinsumElement for f32 {
    #[inline]
    fn einsum_fma(acc: Self, a: Self, b: Self) -> Self {
        acc + a * b
    }
}

impl EinsumElement for f64 {
    #[inline]
    fn einsum_fma(acc: Self, a: Self, b: Self) -> Self {
        acc + a * b
    }
}

impl EinsumElement for Complex32 {
    #[inline]
    fn einsum_fma(acc: Self, a: Self, b: Self) -> Self {
        acc + a * b
    }
}

impl EinsumElement for Complex64 {
    #[inline]
    fn einsum_fma(acc: Self, a: Self, b: Self) -> Self {
        acc + a * b
    }
}

// =============================================================================
// 2. Subscript Parsing & Validation
// =============================================================================

/// Internal numeric label ID for subscript axes (0..51 = user chars, 1000+ = synthetic ellipsis axes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LabelId(u16);

impl LabelId {
    fn from_ascii(b: u8) -> Result<Self> {
        if b.is_ascii_alphabetic() {
            Ok(Self(u16::from(b)))
        } else {
            Err(NumPyError::InvalidValue(format!(
                "invalid einsum subscript character '{}'",
                b as char
            )))
        }
    }

    fn ellipsis(idx: usize) -> Self {
        Self(1000 + idx as u16)
    }

    fn display_char(self) -> String {
        if self.0 < 1000 {
            ((self.0 as u8) as char).to_string()
        } else {
            format!("#{}", self.0 - 1000)
        }
    }
}

#[derive(Debug, Clone)]
enum ParsedToken {
    Label(LabelId),
    Ellipsis,
}

fn tokenize_term(term: &str) -> Result<Vec<ParsedToken>> {
    let bytes = term.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let mut has_ellipsis = false;
    while i < bytes.len() {
        if bytes[i] == b'.' {
            if i + 2 >= bytes.len() || bytes[i + 1] != b'.' || bytes[i + 2] != b'.' {
                return Err(NumPyError::InvalidValue(format!(
                    "invalid ellipsis syntax in einsum term '{term}'"
                )));
            }
            if has_ellipsis {
                return Err(NumPyError::InvalidValue(format!(
                    "more than one ellipsis in einsum term '{term}'"
                )));
            }
            has_ellipsis = true;
            tokens.push(ParsedToken::Ellipsis);
            i += 3;
        } else {
            tokens.push(ParsedToken::Label(LabelId::from_ascii(bytes[i])?));
            i += 1;
        }
    }
    Ok(tokens)
}

#[derive(Debug, Clone)]
struct ParsedEinsum {
    inputs: Vec<Vec<LabelId>>,
    output: Vec<LabelId>,
    dim_map: BTreeMap<LabelId, usize>,
    has_ellipsis: bool,
    ellipsis_len: usize,
}

fn parse_einsum(subscripts: &str, operand_shapes: &[&[usize]]) -> Result<ParsedEinsum> {
    if operand_shapes.is_empty() {
        return Err(NumPyError::InvalidValue(
            "must specify at least one operand for einsum".into(),
        ));
    }

    let cleaned: String = subscripts.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs_str, explicit_rhs) = match cleaned.split_once("->") {
        Some((l, r)) => {
            if r.contains("->") || r.contains(',') {
                return Err(NumPyError::InvalidValue(format!(
                    "invalid einsum output subscripts '{cleaned}'"
                )));
            }
            (l, Some(r))
        }
        None => (cleaned.as_str(), None),
    };

    let lhs_terms: Vec<&str> = lhs_str.split(',').collect();
    if lhs_terms.len() != operand_shapes.len() {
        return Err(NumPyError::InvalidValue(format!(
            "number of subscripts ({}) does not match number of operands ({})",
            lhs_terms.len(),
            operand_shapes.len()
        )));
    }

    let mut tokenized_inputs = Vec::with_capacity(lhs_terms.len());
    let mut ell_shapes: Vec<&[usize]> = Vec::new();
    let mut any_ellipsis = false;

    for (k, (&term, &shape)) in lhs_terms.iter().zip(operand_shapes.iter()).enumerate() {
        let toks = tokenize_term(term)?;
        let non_ell = toks
            .iter()
            .filter(|t| matches!(t, ParsedToken::Label(_)))
            .count();
        let has_ell = toks.iter().any(|t| matches!(t, ParsedToken::Ellipsis));
        if has_ell {
            any_ellipsis = true;
            if shape.len() < non_ell {
                return Err(NumPyError::InvalidValue(format!(
                    "operand {k} does not have enough dimensions for subscript '{term}' (got {}, need at least {non_ell})",
                    shape.len()
                )));
            }
            let ell_start = toks
                .iter()
                .position(|t| matches!(t, ParsedToken::Ellipsis))
                .unwrap_or(0);
            let ell_count = shape.len() - non_ell;
            ell_shapes.push(&shape[ell_start..ell_start + ell_count]);
        } else if shape.len() != non_ell {
            return Err(NumPyError::InvalidValue(format!(
                "operand {k} has {} dimensions, but subscript '{term}' specifies {non_ell}",
                shape.len()
            )));
        }
        tokenized_inputs.push(toks);
    }

    let ell_out_shape = if any_ellipsis {
        numpy_types::broadcast_many_shapes(&ell_shapes)?
    } else {
        Vec::new()
    };
    let ellipsis_len = ell_out_shape.len();

    let mut inputs: Vec<Vec<LabelId>> = Vec::with_capacity(lhs_terms.len());
    let mut dim_map: BTreeMap<LabelId, usize> = BTreeMap::new();
    let mut char_counts: BTreeMap<LabelId, usize> = BTreeMap::new();

    for (k, (toks, &shape)) in tokenized_inputs
        .iter()
        .zip(operand_shapes.iter())
        .enumerate()
    {
        let non_ell = toks
            .iter()
            .filter(|t| matches!(t, ParsedToken::Label(_)))
            .count();
        let ell_count = shape.len() - non_ell;
        let pad_left = ellipsis_len.saturating_sub(ell_count);
        let mut labels = Vec::with_capacity(non_ell + ell_count);
        let mut ax = 0usize;
        let mut same_op_dims: HashMap<LabelId, usize> = HashMap::new();

        for tok in toks {
            match tok {
                ParsedToken::Label(lbl) => {
                    let dim = shape[ax];
                    ax += 1;
                    if let Some(&prev) = same_op_dims.get(lbl) {
                        if prev != dim {
                            return Err(NumPyError::ShapeMismatch(format!(
                                "dimensions in operand {k} for collapsing index '{}' do not match ({prev} vs {dim})",
                                lbl.display_char()
                            )));
                        }
                    } else {
                        same_op_dims.insert(*lbl, dim);
                    }
                    if let Some(existing) = dim_map.get_mut(lbl) {
                        if *existing == 1 {
                            *existing = dim;
                        } else if dim != 1 && *existing != dim {
                            return Err(NumPyError::ShapeMismatch(format!(
                                "operands could not be broadcast together along einsum label '{}' ({existing} vs {dim})",
                                lbl.display_char()
                            )));
                        }
                    } else {
                        dim_map.insert(*lbl, dim);
                    }
                    *char_counts.entry(*lbl).or_insert(0) += 1;
                    labels.push(*lbl);
                }
                ParsedToken::Ellipsis => {
                    for j in 0..ell_count {
                        let lbl = LabelId::ellipsis(pad_left + j);
                        let dim = shape[ax];
                        ax += 1;
                        dim_map.insert(lbl, ell_out_shape[pad_left + j]);
                        let _ = dim;
                        labels.push(lbl);
                    }
                }
            }
        }
        inputs.push(labels);
    }

    let output = match explicit_rhs {
        Some(rhs) => {
            let rhs_toks = tokenize_term(rhs)?;
            let mut out = Vec::new();
            let mut seen_out: HashSet<LabelId> = HashSet::new();
            let mut rhs_has_ellipsis = false;
            for tok in rhs_toks {
                match tok {
                    ParsedToken::Label(lbl) => {
                        if !dim_map.contains_key(&lbl) {
                            return Err(NumPyError::InvalidValue(format!(
                                "einsum(): subscript output label '{}' did not appear in any input",
                                lbl.display_char()
                            )));
                        }
                        if !seen_out.insert(lbl) {
                            return Err(NumPyError::InvalidValue(format!(
                                "einsum(): subscript output includes repeated label '{}'",
                                lbl.display_char()
                            )));
                        }
                        out.push(lbl);
                    }
                    ParsedToken::Ellipsis => {
                        if !any_ellipsis {
                            return Err(NumPyError::InvalidValue(
                                "output cannot specify '...' when inputs contain no '...'".into(),
                            ));
                        }
                        rhs_has_ellipsis = true;
                        for j in 0..ellipsis_len {
                            out.push(LabelId::ellipsis(j));
                        }
                    }
                }
            }
            if any_ellipsis && ellipsis_len > 0 && !rhs_has_ellipsis {
                return Err(NumPyError::InvalidValue(
                    "output has more dimensions than subscripts given in einstein sum, \
                     but no '...' ellipsis provided to broadcast the extra dimensions."
                        .into(),
                ));
            }
            out
        }
        None => {
            let mut out = Vec::new();
            if any_ellipsis {
                for j in 0..ellipsis_len {
                    out.push(LabelId::ellipsis(j));
                }
            }
            for (&lbl, &cnt) in &char_counts {
                if cnt == 1 {
                    out.push(lbl);
                }
            }
            out
        }
    };

    Ok(ParsedEinsum {
        inputs,
        output,
        dim_map,
        has_ellipsis: any_ellipsis,
        ellipsis_len,
    })
}

// =============================================================================
// 3. Path Planning (`einsum_path` Greedy & Optimal)
// =============================================================================

#[inline]
fn set_size(labels: &BTreeSet<LabelId>, dim_map: &BTreeMap<LabelId, usize>) -> usize {
    labels
        .iter()
        .map(|l| dim_map.get(l).copied().unwrap_or(1))
        .fold(1usize, |a, b| a.saturating_mul(b))
}

fn contraction_flops(
    in_sets: &[&BTreeSet<LabelId>],
    out_set: &BTreeSet<LabelId>,
    dim_map: &BTreeMap<LabelId, usize>,
) -> f64 {
    let mut union = BTreeSet::new();
    for &s in in_sets {
        for &l in s {
            union.insert(l);
        }
    }
    let idx_cost: f64 = union
        .iter()
        .map(|l| dim_map.get(l).copied().unwrap_or(1) as f64)
        .product();
    let has_contraction = union.len() > out_set.len() || in_sets.len() > 2;
    let mult = if has_contraction && in_sets.len() > 1 {
        2.0
    } else {
        1.0
    };
    idx_cost * mult
}

fn parse_contraction(
    positions: &[usize],
    input_sets: &[BTreeSet<LabelId>],
    output_set: &BTreeSet<LabelId>,
    dim_map: &BTreeMap<LabelId, usize>,
) -> (BTreeSet<LabelId>, Vec<LabelId>, f64, usize) {
    let mut remaining_counts: BTreeMap<LabelId, usize> = BTreeMap::new();
    for (idx, s) in input_sets.iter().enumerate() {
        if !positions.contains(&idx) {
            for &l in s {
                *remaining_counts.entry(l).or_insert(0) += 1;
            }
        }
    }
    for &l in output_set {
        *remaining_counts.entry(l).or_insert(0) += 1;
    }

    let mut new_set = BTreeSet::new();
    let mut inner_labels = Vec::new();
    let mut involved = BTreeSet::new();

    for &p in positions {
        for &l in &input_sets[p] {
            involved.insert(l);
            if remaining_counts.contains_key(&l) {
                new_set.insert(l);
            } else if !inner_labels.contains(&l) {
                inner_labels.push(l);
            }
        }
    }

    let idx_cost: f64 = involved
        .iter()
        .map(|l| dim_map.get(l).copied().unwrap_or(1) as f64)
        .product();
    let mult = if inner_labels.is_empty() { 1.0 } else { 2.0 };
    let new_size = set_size(&new_set, dim_map);
    (new_set, inner_labels, idx_cost * mult, new_size)
}

fn greedy_path(
    input_sets: &[BTreeSet<LabelId>],
    output_set: &BTreeSet<LabelId>,
    dim_map: &BTreeMap<LabelId, usize>,
    memory_limit: usize,
) -> Vec<Vec<usize>> {
    let n = input_sets.len();
    if n <= 1 {
        return if n == 1 { vec![vec![0]] } else { vec![] };
    }
    if n == 2 {
        return vec![vec![0, 1]];
    }

    let mut sets = input_sets.to_vec();
    let mut path = Vec::with_capacity(n - 1);

    while sets.len() > 1 {
        let cur_n = sets.len();
        if cur_n == 2 {
            path.push(vec![0, 1]);
            break;
        }

        // Evaluate all pairwise contractions
        let mut best: Option<(i128, f64, usize, usize, BTreeSet<LabelId>)> = None;
        for i in 0..cur_n {
            for j in (i + 1)..cur_n {
                // Prefer pairs that share at least one index unless only outer products remain
                let shares = sets[i].iter().any(|l| sets[j].contains(l));
                let s1 = set_size(&sets[i], dim_map) as i128;
                let s2 = set_size(&sets[j], dim_map) as i128;
                let (new_set, _, flops, new_size) =
                    parse_contraction(&[i, j], &sets, output_set, dim_map);

                if (new_size as usize) > memory_limit {
                    continue;
                }
                let size_removed = (s1 + s2) - (new_size as i128);
                // Priority: (shares_index, size_removed, -flops)
                let score_primary = if shares {
                    size_removed
                } else {
                    i128::MIN / 4 + size_removed
                };

                let is_better = match &best {
                    None => true,
                    Some((b_prim, b_flops, _, _, _)) => {
                        score_primary > *b_prim
                            || (score_primary == *b_prim && flops < *b_flops)
                    }
                };
                if is_better {
                    best = Some((score_primary, flops, i, j, new_set));
                }
            }
        }

        let (i, j, new_set) = match best {
            Some((_, _, i, j, ns)) => (i, j, ns),
            None => {
                // Memory limit exceeded by every pair; fallback to contracting remaining together
                let all: Vec<usize> = (0..cur_n).collect();
                path.push(all);
                break;
            }
        };

        path.push(vec![i, j]);
        // Remove j then i, push new_set at end (matching NumPy path index convention!)
        sets.remove(j);
        sets.remove(i);
        sets.push(new_set);
    }

    path
}

fn optimal_path(
    input_sets: &[BTreeSet<LabelId>],
    output_set: &BTreeSet<LabelId>,
    dim_map: &BTreeMap<LabelId, usize>,
    memory_limit: usize,
) -> Vec<Vec<usize>> {
    let n = input_sets.len();
    if n <= 1 {
        return if n == 1 { vec![vec![0]] } else { vec![] };
    }
    if n == 2 {
        return vec![vec![0, 1]];
    }

    // If too many operands for exhaustive DFS (> 8), seed with greedy bound first
    let mut best_path = greedy_path(input_sets, output_set, dim_map, memory_limit);
    let mut best_cost = eval_path_cost(input_sets, output_set, dim_map, &best_path);

    if n > 9 {
        return best_path;
    }

    struct State<'a> {
        output_set: &'a BTreeSet<LabelId>,
        dim_map: &'a BTreeMap<LabelId, usize>,
        memory_limit: usize,
        best_cost: f64,
        best_path: Vec<Vec<usize>>,
    }

    fn dfs(
        sets: &Vec<BTreeSet<LabelId>>,
        cur_cost: f64,
        cur_path: &mut Vec<Vec<usize>>,
        st: &mut State<'_>,
    ) {
        if cur_cost >= st.best_cost {
            return;
        }
        let m = sets.len();
        if m == 2 {
            let (_, _, step_cost, _) = parse_contraction(&[0, 1], sets, st.output_set, st.dim_map);
            let total = cur_cost + step_cost;
            if total < st.best_cost {
                st.best_cost = total;
                cur_path.push(vec![0, 1]);
                st.best_path = cur_path.clone();
                cur_path.pop();
            }
            return;
        }

        let mut candidates = Vec::new();
        for i in 0..m {
            for j in (i + 1)..m {
                let (new_set, _, step_cost, new_size) =
                    parse_contraction(&[i, j], sets, st.output_set, st.dim_map);
                if new_size <= st.memory_limit && cur_cost + step_cost < st.best_cost {
                    candidates.push((step_cost, i, j, new_set));
                }
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

        for (step_cost, i, j, new_set) in candidates {
            if cur_cost + step_cost >= st.best_cost {
                break;
            }
            let mut next_sets = sets.clone();
            next_sets.remove(j);
            next_sets.remove(i);
            next_sets.push(new_set);
            cur_path.push(vec![i, j]);
            dfs(&next_sets, cur_cost + step_cost, cur_path, st);
            cur_path.pop();
        }
    }

    let mut state = State {
        output_set,
        dim_map,
        memory_limit,
        best_cost,
        best_path: best_path.clone(),
    };
    let mut cur_path = Vec::with_capacity(n);
    dfs(&input_sets.to_vec(), 0.0, &mut cur_path, &mut state);
    let _ = &mut best_cost;
    let _ = &mut best_path;
    state.best_path
}

fn eval_path_cost(
    input_sets: &[BTreeSet<LabelId>],
    output_set: &BTreeSet<LabelId>,
    dim_map: &BTreeMap<LabelId, usize>,
    path: &[Vec<usize>],
) -> f64 {
    let mut sets = input_sets.to_vec();
    let mut total = 0.0f64;
    for step in path {
        let (new_set, _, cost, _) = parse_contraction(step, &sets, output_set, dim_map);
        total += cost;
        let mut sorted = step.clone();
        sorted.sort_unstable();
        for &p in sorted.iter().rev() {
            sets.remove(p);
        }
        sets.push(new_set);
    }
    total
}

/// Computes the contraction plan and cost statistics matching `numpy.einsum_path`.
pub fn einsum_path_shapes(
    subscripts: &str,
    shapes: &[&[usize]],
    optimize: EinsumOptimize,
    memory_limit: Option<usize>,
) -> Result<EinsumPathInfo> {
    let parsed = parse_einsum(subscripts, shapes)?;
    let n = shapes.len();

    let input_sets: Vec<BTreeSet<LabelId>> = parsed
        .inputs
        .iter()
        .map(|v| v.iter().copied().collect())
        .collect();
    let output_set: BTreeSet<LabelId> = parsed.output.iter().copied().collect();

    let union_all: BTreeSet<LabelId> = parsed.dim_map.keys().copied().collect();
    let naive_cost: f64 = union_all
        .iter()
        .map(|l| parsed.dim_map.get(l).copied().unwrap_or(1) as f64)
        .product::<f64>()
        * (n.max(1) as f64);

    let mem_lim = memory_limit.unwrap_or(usize::MAX);
    let resolved_opt = match optimize {
        EinsumOptimize::Auto => {
            if n <= 4 {
                EinsumOptimize::Optimal
            } else {
                EinsumOptimize::Greedy
            }
        }
        other => other,
    };

    let path = match resolved_opt {
        EinsumOptimize::None => vec![(0..n).collect()],
        EinsumOptimize::Greedy => greedy_path(&input_sets, &output_set, &parsed.dim_map, mem_lim),
        EinsumOptimize::Optimal => optimal_path(&input_sets, &output_set, &parsed.dim_map, mem_lim),
        EinsumOptimize::Auto => unreachable!(),
    };

    // Build contraction metadata and human-readable report
    let mut cur_labels: Vec<Vec<LabelId>> = parsed.inputs.clone();
    let mut cur_sets = input_sets.clone();
    let mut contractions = Vec::with_capacity(path.len());
    let mut opt_cost = 0.0f64;
    let mut largest_intermediate = shapes
        .iter()
        .try_fold(0usize, |acc, s| Ok::<_, NumPyError>(acc.max(shape_size(s)?)))?;

    let mut report = String::new();
    report.push_str(&format!("  Complete contraction:  {subscripts}\n"));
    let naive_val = contraction_flops(
        &input_sets.iter().collect::<Vec<_>>(),
        &output_set,
        &parsed.dim_map,
    ) * (n.saturating_sub(1).max(1) as f64);
    let _ = naive_val;

    for (step_idx, step) in path.iter().enumerate() {
        let is_last = step_idx + 1 == path.len();
        let (new_set, inner_lbls, cost, out_size) =
            parse_contraction(step, &cur_sets, &output_set, &parsed.dim_map);
        opt_cost += cost;
        largest_intermediate = largest_intermediate.max(out_size);

        let out_order: Vec<LabelId> = if is_last {
            parsed.output.clone()
        } else {
            new_set.iter().copied().collect()
        };

        let in_exprs: Vec<String> = step
            .iter()
            .map(|&p| {
                cur_labels[p]
                    .iter()
                    .map(|l| l.display_char())
                    .collect::<String>()
            })
            .collect();
        let out_expr: String = out_order.iter().map(|l| l.display_char()).collect();
        let sub_str = format!("{}->{out_expr}", in_exprs.join(","));
        let out_shape: Vec<usize> = out_order
            .iter()
            .map(|l| parsed.dim_map.get(l).copied().unwrap_or(1))
            .collect();
        let can_blas = step.len() == 2 && !inner_lbls.is_empty();
        let contract_lbls: String = inner_lbls.iter().map(|l| l.display_char()).collect();

        contractions.push(EinsumContraction {
            operands: step.clone(),
            contraction_labels: contract_lbls,
            subscript: sub_str,
            output_shape: out_shape,
            can_blas,
        });

        let mut sorted_pos = step.clone();
        sorted_pos.sort_unstable();
        for &p in sorted_pos.iter().rev() {
            cur_sets.remove(p);
            cur_labels.remove(p);
        }
        cur_sets.push(new_set);
        cur_labels.push(out_order);
    }

    let speedup = if opt_cost > 0.0 {
        naive_cost / opt_cost
    } else {
        1.0
    };
    report.push_str(&format!(
        "         Naive scaling:  {}\n",
        union_all.len()
    ));
    report.push_str(&format!(
        "     Optimized scaling:  {}\n",
        contractions
            .iter()
            .map(|c| c.subscript.split("->").next().unwrap_or("").len())
            .max()
            .unwrap_or(0)
    ));
    report.push_str(&format!("      Naive FLOP count:  {naive_cost:.3e}\n"));
    report.push_str(&format!("  Optimized FLOP count:  {opt_cost:.3e}\n"));
    report.push_str(&format!("   Theoretical speedup:  {speedup:.3}\n"));
    report.push_str(&format!(
        "  Largest intermediate:  {largest_intermediate} elements\n"
    ));

    let input_shapes = shapes.iter().map(|s| s.to_vec()).collect();
    let _ = parsed.has_ellipsis;
    let _ = parsed.ellipsis_len;

    Ok(EinsumPathInfo {
        path,
        contractions,
        input_shapes,
        naive_cost,
        opt_cost,
        speedup,
        largest_intermediate,
        report,
    })
}

/// Computes `EinsumPathInfo` from `DynArray` operands.
pub fn einsum_path(
    subscripts: &str,
    operands: &[&DynArray],
    optimize: EinsumOptimize,
    memory_limit: Option<usize>,
) -> Result<EinsumPathInfo> {
    let shapes: Vec<&[usize]> = operands.iter().map(|a| a.shape()).collect();
    einsum_path_shapes(subscripts, &shapes, optimize, memory_limit)
}

// =============================================================================
// 4. Contraction Execution Engine
// =============================================================================

/// Removes duplicate labels within a single operand by extracting diagonal views/arrays,
/// and sums out any label in `in_labels` that does not appear in `target_labels`.
fn preprocess_single_operand<T: EinsumElement>(
    arr: &NdArray<T>,
    in_labels: &[LabelId],
    target_labels: &[LabelId],
) -> Result<(NdArray<T>, Vec<LabelId>)> {
    let mut cur_arr = arr.clone();
    let mut cur_lbls = in_labels.to_vec();

    // Step 1: Collapse repeated labels inside this single operand via `diagonal(0, ax1, ax2)`.
    loop {
        let mut seen: HashMap<LabelId, usize> = HashMap::new();
        let mut dup: Option<(usize, usize)> = None;
        for (i, &lbl) in cur_lbls.iter().enumerate() {
            if let Some(&first) = seen.get(&lbl) {
                dup = Some((first, i));
                break;
            }
            seen.insert(lbl, i);
        }
        let Some((ax1, ax2)) = dup else { break };
        let collapsed = diagonal(&cur_arr, 0, ax1 as isize, ax2 as isize)?;
        let lbl = cur_lbls[ax1];
        let mut next_lbls = Vec::with_capacity(cur_lbls.len() - 1);
        for (i, &l) in cur_lbls.iter().enumerate() {
            if i != ax1 && i != ax2 {
                next_lbls.push(l);
            }
        }
        next_lbls.push(lbl);
        cur_arr = collapsed;
        cur_lbls = next_lbls;
    }

    // Step 2: Check if any axes need to be summed out (not in `target_labels`).
    let target_set: HashSet<LabelId> = target_labels.iter().copied().collect();
    let keep_axes: Vec<usize> = cur_lbls
        .iter()
        .enumerate()
        .filter_map(|(i, l)| if target_set.contains(l) { Some(i) } else { None })
        .collect();

    if keep_axes.len() < cur_lbls.len() {
        let sum_axes: Vec<usize> = cur_lbls
            .iter()
            .enumerate()
            .filter_map(|(i, l)| if !target_set.contains(l) { Some(i) } else { None })
            .collect();

        // Permute kept axes to front, summed axes to back (zero-copy transpose!)
        let mut perm: Vec<isize> = keep_axes.iter().map(|&a| a as isize).collect();
        perm.extend(sum_axes.iter().map(|&a| a as isize));
        let transposed = cur_arr.transpose(Some(&perm))?;
        let contiguous = transposed.to_c_contiguous();
        let src = contiguous.as_slice().unwrap_or(&[]);

        let out_shape: Vec<usize> = keep_axes.iter().map(|&a| cur_arr.shape()[a]).collect();
        let out_lbls: Vec<LabelId> = keep_axes.iter().map(|&a| cur_lbls[a]).collect();
        let out_len = shape_size(&out_shape)?;
        let sum_len: usize = sum_axes.iter().map(|&a| cur_arr.shape()[a]).product();

        if out_len == 0 {
            return Ok((
                finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape)?,
                out_lbls,
            ));
        }

        let mut uninit = AlignedBuffer::<T>::try_new_uninit(out_len)?;
        let dst = uninit.as_mut_slice();
        let zero = T::zero();

        if out_len * sum_len >= RAYON_THRESHOLD && out_len >= 4 {
            dst.par_chunks_mut(1)
                .enumerate()
                .for_each(|(i, slot)| {
                    let mut acc = zero;
                    if sum_len > 0 {
                        let base = i * sum_len;
                        for &v in &src[base..base + sum_len] {
                            acc = acc.elem_add(v);
                        }
                    }
                    slot[0].write(acc);
                });
        } else if out_len == 1 && sum_len >= RAYON_THRESHOLD {
            let acc = src
                .par_chunks(4096)
                .map(|chunk| {
                    let mut local = zero;
                    for &v in chunk {
                        local = local.elem_add(v);
                    }
                    local
                })
                .reduce(|| zero, |a, b| a.elem_add(b));
            dst[0].write(acc);
        } else {
            for (i, slot) in dst.iter_mut().enumerate() {
                let mut acc = zero;
                if sum_len > 0 {
                    let base = i * sum_len;
                    for &v in &src[base..base + sum_len] {
                        acc = acc.elem_add(v);
                    }
                }
                slot.write(acc);
            }
        }

        let buf = unsafe { uninit.assume_init() };
        cur_arr = finish_c_array(buf, &out_shape)?;
        cur_lbls = out_lbls;
    }

    // Step 3: If `cur_lbls` is a permutation of `target_labels`, return ZERO-COPY transpose view!
    if cur_lbls.len() == target_labels.len() && cur_lbls != target_labels {
        let mut matches_exact = true;
        let mut perm = Vec::with_capacity(target_labels.len());
        for &tl in target_labels {
            if let Some(pos) = cur_lbls.iter().position(|&cl| cl == tl) {
                perm.push(pos as isize);
            } else {
                matches_exact = false;
                break;
            }
        }
        if matches_exact {
            cur_arr = cur_arr.transpose(Some(&perm))?;
            cur_lbls = target_labels.to_vec();
        }
    }

    Ok((cur_arr, cur_lbls))
}

/// Contracts two deduplicated operands `(A, labels_A)` and `(B, labels_B)` into `target_labels`.
fn contract_pair<T: EinsumElement>(
    a: &NdArray<T>,
    labels_a: &[LabelId],
    b: &NdArray<T>,
    labels_b: &[LabelId],
    target_labels: &[LabelId],
    dim_map: &BTreeMap<LabelId, usize>,
) -> Result<(NdArray<T>, Vec<LabelId>)> {
    // First remove any labels unique to A or B that aren't in `target_labels`
    let mut keep_for_a: Vec<LabelId> = target_labels.to_vec();
    for &l in labels_b {
        if !keep_for_a.contains(&l) {
            keep_for_a.push(l);
        }
    }
    let mut keep_for_b: Vec<LabelId> = target_labels.to_vec();
    for &l in labels_a {
        if !keep_for_b.contains(&l) {
            keep_for_b.push(l);
        }
    }

    // Only pre-reduce if A or B has duplicates or labels outside partner/output
    let a_needs_pre = {
        let mut s = HashSet::new();
        labels_a.iter().any(|l| !s.insert(*l) || !keep_for_a.contains(l))
    };
    let (a_clean, a_lbls) = if a_needs_pre {
        let needed: Vec<LabelId> = labels_a
            .iter()
            .copied()
            .filter(|l| keep_for_a.contains(l))
            .collect();
        preprocess_single_operand(a, labels_a, &needed)?
    } else {
        (a.clone(), labels_a.to_vec())
    };

    let b_needs_pre = {
        let mut s = HashSet::new();
        labels_b.iter().any(|l| !s.insert(*l) || !keep_for_b.contains(l))
    };
    let (b_clean, b_lbls) = if b_needs_pre {
        let needed: Vec<LabelId> = labels_b
            .iter()
            .copied()
            .filter(|l| keep_for_b.contains(l))
            .collect();
        preprocess_single_operand(b, labels_b, &needed)?
    } else {
        (b.clone(), labels_b.to_vec())
    };

    // Broadcast size-1 dims to canonical dims in `dim_map` if needed
    let a_canon_shape: Vec<usize> = a_lbls.iter().map(|l| dim_map[l]).collect();
    let a_bcast = if a_clean.shape() != a_canon_shape.as_slice() {
        a_clean.broadcast_to(&a_canon_shape)?
    } else {
        a_clean
    };
    let b_canon_shape: Vec<usize> = b_lbls.iter().map(|l| dim_map[l]).collect();
    let b_bcast = if b_clean.shape() != b_canon_shape.as_slice() {
        b_clean.broadcast_to(&b_canon_shape)?
    } else {
        b_clean
    };

    let b_set: HashSet<LabelId> = b_lbls.iter().copied().collect();
    let a_set: HashSet<LabelId> = a_lbls.iter().copied().collect();
    let out_set: HashSet<LabelId> = target_labels.iter().copied().collect();

    // Partition labels:
    // - batch: in A, in B, in Out
    // - contract: in A, in B, not in Out
    // - free_a: in A, not in B, in Out
    // - free_b: in B, not in A, in Out
    let mut batch = Vec::new();
    let mut contract = Vec::new();
    let mut free_a = Vec::new();
    let mut free_b = Vec::new();

    for &l in &a_lbls {
        if b_set.contains(&l) {
            if out_set.contains(&l) {
                batch.push(l);
            } else {
                contract.push(l);
            }
        } else {
            free_a.push(l);
        }
    }
    for &l in &b_lbls {
        if !a_set.contains(&l) {
            free_b.push(l);
        }
    }

    let mut a_perm_lbls = Vec::with_capacity(a_lbls.len());
    a_perm_lbls.extend_from_slice(&batch);
    a_perm_lbls.extend_from_slice(&free_a);
    a_perm_lbls.extend_from_slice(&contract);

    let mut b_perm_lbls = Vec::with_capacity(b_lbls.len());
    b_perm_lbls.extend_from_slice(&batch);
    b_perm_lbls.extend_from_slice(&contract);
    b_perm_lbls.extend_from_slice(&free_b);

    let perm_a: Vec<isize> = a_perm_lbls
        .iter()
        .map(|l| a_lbls.iter().position(|x| x == l).unwrap_or(0) as isize)
        .collect();
    let perm_b: Vec<isize> = b_perm_lbls
        .iter()
        .map(|l| b_lbls.iter().position(|x| x == l).unwrap_or(0) as isize)
        .collect();

    let a_cont = a_bcast.transpose(Some(&perm_a))?.to_c_contiguous();
    let b_cont = b_bcast.transpose(Some(&perm_b))?.to_c_contiguous();

    let batch_size: usize = batch.iter().map(|l| dim_map[l]).product();
    let m: usize = free_a.iter().map(|l| dim_map[l]).product();
    let n: usize = free_b.iter().map(|l| dim_map[l]).product();
    let k: usize = contract.iter().map(|l| dim_map[l]).product();

    let mut raw_out_lbls = Vec::with_capacity(batch.len() + free_a.len() + free_b.len());
    raw_out_lbls.extend_from_slice(&batch);
    raw_out_lbls.extend_from_slice(&free_a);
    raw_out_lbls.extend_from_slice(&free_b);
    let raw_out_shape: Vec<usize> = raw_out_lbls.iter().map(|l| dim_map[l]).collect();
    let raw_total = shape_size(&raw_out_shape)?;

    if raw_total == 0 {
        let zeroed = finish_c_array(AlignedBuffer::new_zeroed(0), &raw_out_shape)?;
        let final_perm: Vec<isize> = target_labels
            .iter()
            .map(|l| raw_out_lbls.iter().position(|x| x == l).unwrap_or(0) as isize)
            .collect();
        return Ok((zeroed.transpose(Some(&final_perm))?, target_labels.to_vec()));
    }

    let a_sl = a_cont.as_slice().unwrap_or(&[]);
    let b_sl = b_cont.as_slice().unwrap_or(&[]);
    let mut out_buf = AlignedBuffer::<T>::try_new_zeroed(raw_total)?;
    let out_sl = out_buf.as_mut_slice();

    if k > 0 {
        let a_mat_stride = m * k;
        let b_mat_stride = k * n;
        let c_mat_stride = m * n;
        let work_ops = batch_size.saturating_mul(m).saturating_mul(n).saturating_mul(k);

        let matmul_batch = |a_mat: &[T], b_mat: &[T], c_mat: &mut [T]| {
            // `i-k-j` loop order for streaming access and auto-vectorization
            for i in 0..m {
                let c_row = &mut c_mat[i * n..(i + 1) * n];
                let a_row = &a_mat[i * k..(i + 1) * k];
                for p in 0..k {
                    let a_val = a_row[p];
                    let b_row = &b_mat[p * n..(p + 1) * n];
                    for (c_cell, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                        *c_cell = T::einsum_fma(*c_cell, a_val, b_val);
                    }
                }
            }
        };

        if work_ops >= RAYON_THRESHOLD && batch_size >= 2 && m * n >= 16 {
            out_sl
                .par_chunks_mut(c_mat_stride)
                .enumerate()
                .for_each(|(b_idx, c_mat)| {
                    let a_mat = &a_sl[b_idx * a_mat_stride..(b_idx + 1) * a_mat_stride];
                    let b_mat = &b_sl[b_idx * b_mat_stride..(b_idx + 1) * b_mat_stride];
                    matmul_batch(a_mat, b_mat, c_mat);
                });
        } else if work_ops >= RAYON_THRESHOLD && batch_size * m >= 4 && n >= 1 {
            // Parallelize across all `(batch, i)` output rows
            out_sl
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(global_row, c_row)| {
                    let b_idx = global_row / m;
                    let i = global_row % m;
                    let a_row = &a_sl[b_idx * a_mat_stride + i * k..b_idx * a_mat_stride + (i + 1) * k];
                    let b_mat = &b_sl[b_idx * b_mat_stride..(b_idx + 1) * b_mat_stride];
                    for p in 0..k {
                        let a_val = a_row[p];
                        let b_row = &b_mat[p * n..(p + 1) * n];
                        for (c_cell, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                            *c_cell = T::einsum_fma(*c_cell, a_val, b_val);
                        }
                    }
                });
        } else if work_ops >= RAYON_THRESHOLD && batch_size == 1 && m == 1 && n == 1 {
            // Large single dot product
            let acc = a_sl
                .par_chunks(4096)
                .zip(b_sl.par_chunks(4096))
                .map(|(ac, bc)| {
                    let mut local = T::zero();
                    for (&x, &y) in ac.iter().zip(bc.iter()) {
                        local = T::einsum_fma(local, x, y);
                    }
                    local
                })
                .reduce(T::zero, |x, y| x.elem_add(y));
            out_sl[0] = acc;
        } else {
            for b_idx in 0..batch_size {
                let a_mat = &a_sl[b_idx * a_mat_stride..(b_idx + 1) * a_mat_stride];
                let b_mat = &b_sl[b_idx * b_mat_stride..(b_idx + 1) * b_mat_stride];
                let c_mat = &mut out_sl[b_idx * c_mat_stride..(b_idx + 1) * c_mat_stride];
                matmul_batch(a_mat, b_mat, c_mat);
            }
        }
    }

    let raw_arr = finish_c_array(out_buf, &raw_out_shape)?;
    if raw_out_lbls == target_labels {
        return Ok((raw_arr, raw_out_lbls));
    }
    let final_perm: Vec<isize> = target_labels
        .iter()
        .map(|l| raw_out_lbls.iter().position(|x| x == l).unwrap_or(0) as isize)
        .collect();
    Ok((
        raw_arr.transpose(Some(&final_perm))?,
        target_labels.to_vec(),
    ))
}

/// Evaluates an Einstein summation expression with the specified `EinsumOptimize` strategy.
pub fn einsum_with_optimize<T: EinsumElement>(
    subscripts: &str,
    operands: &[&NdArray<T>],
    optimize: EinsumOptimize,
) -> Result<NdArray<T>> {
    let shapes: Vec<&[usize]> = operands.iter().map(|a| a.shape()).collect();
    let parsed = parse_einsum(subscripts, &shapes)?;

    // Fast path for 1-operand einsum (view transpose or diagonal/sum reduction)
    if operands.len() == 1 {
        let (res, _) = preprocess_single_operand(operands[0], &parsed.inputs[0], &parsed.output)?;
        return Ok(res);
    }

    // Naive 2+ operand or optimized path planning:
    let effective_opt = match optimize {
        EinsumOptimize::None => {
            if operands.len() == 2 {
                EinsumOptimize::Greedy
            } else {
                // Sequential left-fold without reordering
                let mut cur_arr = operands[0].clone();
                let mut cur_lbls = parsed.inputs[0].clone();
                let n = operands.len();
                for k in 1..n {
                    let is_last = k + 1 == n;
                    let mut target_lbls = if is_last {
                        parsed.output.clone()
                    } else {
                        let mut needed: BTreeSet<LabelId> = parsed.output.iter().copied().collect();
                        for rem in (k + 1)..n {
                            for &l in &parsed.inputs[rem] {
                                needed.insert(l);
                            }
                        }
                        let mut next = Vec::new();
                        for &l in &cur_lbls {
                            if needed.contains(&l) && !next.contains(&l) {
                                next.push(l);
                            }
                        }
                        for &l in &parsed.inputs[k] {
                            if needed.contains(&l) && !next.contains(&l) {
                                next.push(l);
                            }
                        }
                        next
                    };
                    let (next_arr, next_lbls) = contract_pair(
                        &cur_arr,
                        &cur_lbls,
                        operands[k],
                        &parsed.inputs[k],
                        &target_lbls,
                        &parsed.dim_map,
                    )?;
                    cur_arr = next_arr;
                    cur_lbls = next_lbls;
                    let _ = &mut target_lbls;
                }
                return Ok(cur_arr.to_c_contiguous());
            }
        }
        other => other,
    };

    let info = einsum_path_shapes(subscripts, &shapes, effective_opt, None)?;
    let output_set: BTreeSet<LabelId> = parsed.output.iter().copied().collect();

    let mut cur_arrays: Vec<NdArray<T>> = operands.iter().map(|&a| a.clone()).collect();
    let mut cur_labels: Vec<Vec<LabelId>> = parsed.inputs.clone();
    let mut cur_sets: Vec<BTreeSet<LabelId>> = parsed
        .inputs
        .iter()
        .map(|v| v.iter().copied().collect())
        .collect();

    let total_steps = info.path.len();
    for (step_idx, step) in info.path.iter().enumerate() {
        let is_last = step_idx + 1 == total_steps;
        let (new_set, _, _, _) =
            parse_contraction(step, &cur_sets, &output_set, &parsed.dim_map);
        let target_lbls: Vec<LabelId> = if is_last {
            parsed.output.clone()
        } else {
            new_set.iter().copied().collect()
        };

        // Sequential pairwise collapse if a step selects > 2 operands (e.g. memory-limit fallback)
        let mut acc_arr = cur_arrays[step[0]].clone();
        let mut acc_lbls = cur_labels[step[0]].clone();
        for &pos in step.iter().skip(1) {
            let step_target = if pos == *step.last().unwrap_or(&pos) {
                target_lbls.clone()
            } else {
                let mut tmp: BTreeSet<LabelId> = target_lbls.iter().copied().collect();
                for &rem_pos in step {
                    if rem_pos != pos {
                        for &l in &cur_labels[rem_pos] {
                            tmp.insert(l);
                        }
                    }
                }
                tmp.into_iter().collect()
            };
            let (next_arr, next_lbls) = contract_pair(
                &acc_arr,
                &acc_lbls,
                &cur_arrays[pos],
                &cur_labels[pos],
                &step_target,
                &parsed.dim_map,
            )?;
            acc_arr = next_arr;
            acc_lbls = next_lbls;
        }

        let mut sorted = step.clone();
        sorted.sort_unstable();
        for &p in sorted.iter().rev() {
            cur_arrays.remove(p);
            cur_labels.remove(p);
            cur_sets.remove(p);
        }
        cur_arrays.push(acc_arr);
        cur_labels.push(acc_lbls);
        cur_sets.push(new_set);
    }

    Ok(cur_arrays[0].to_c_contiguous())
}

/// Evaluates an Einstein summation expression using `EinsumOptimize::Greedy` optimization.
pub fn einsum<T: EinsumElement>(subscripts: &str, operands: &[&NdArray<T>]) -> Result<NdArray<T>> {
    einsum_with_optimize(subscripts, operands, EinsumOptimize::Greedy)
}

// =============================================================================
// 5. Dynamic Einsum
// =============================================================================

/// Evaluates an Einstein summation over heterogeneous `DynArray` operands (automatically promoting types via `result_type`).
pub fn dyn_einsum(
    subscripts: &str,
    operands: &[&DynArray],
    optimize: EinsumOptimize,
) -> Result<DynArray> {
    if operands.is_empty() {
        return Err(NumPyError::InvalidValue(
            "must specify at least one operand for einsum".into(),
        ));
    }
    let dtypes: Vec<DType> = operands.iter().map(|a| a.dtype()).collect();
    let target = numpy_types::result_type(&dtypes)?;

    let cast: Vec<DynArray> = operands
        .iter()
        .map(|&a| {
            if a.dtype() == target {
                Ok(a.clone())
            } else {
                a.astype(target, Casting::Unsafe)
            }
        })
        .collect::<Result<_>>()?;

    macro_rules! do_einsum {
        ($t:ty, $var:ident) => {{
            let refs: Vec<&NdArray<$t>> = cast
                .iter()
                .map(|d| match d {
                    DynArray::$var(arr) => Ok(arr),
                    _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                })
                .collect::<Result<_>>()?;
            Ok(DynArray::$var(einsum_with_optimize(
                subscripts,
                &refs,
                optimize,
            )?))
        }};
    }

    match target {
        DType::Bool => do_einsum!(bool, Bool),
        DType::I8 => do_einsum!(i8, I8),
        DType::I16 => do_einsum!(i16, I16),
        DType::I32 => do_einsum!(i32, I32),
        DType::I64 => do_einsum!(i64, I64),
        DType::U8 => do_einsum!(u8, U8),
        DType::U16 => do_einsum!(u16, U16),
        DType::U32 => do_einsum!(u32, U32),
        DType::U64 => do_einsum!(u64, U64),
        DType::F16 => do_einsum!(f16, F16),
        DType::F32 => do_einsum!(f32, F32),
        DType::F64 => do_einsum!(f64, F64),
        DType::C64 => do_einsum!(Complex32, C64),
        DType::C128 => do_einsum!(Complex64, C128),
    }
}
