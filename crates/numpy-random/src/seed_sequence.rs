use numpy_types::{NumPyError, Result};
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16;

static ENTROPY_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Flexible seed input accepted by [`SeedSequence`] and BitGenerator constructors.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SeedInput {
    /// Draw unpredictable entropy from the operating system.
    #[default]
    None,
    /// Pre-converted little-endian 32-bit entropy words (empty yields empty run entropy).
    Words(Vec<u32>),
    /// Invalid seed input deferred as an error so constructors can remain fallible.
    Invalid(String),
}

fn push_u64_words(dst: &mut Vec<u32>, v: u64) {
    if v == 0 {
        dst.push(0);
    } else if v <= u32::MAX as u64 {
        dst.push(v as u32);
    } else {
        dst.push(v as u32);
        dst.push((v >> 32) as u32);
    }
}

fn push_u128_words(dst: &mut Vec<u32>, mut v: u128) {
    if v == 0 {
        dst.push(0);
        return;
    }
    while v > 0 {
        dst.push((v & 0xffff_ffff) as u32);
        v >>= 32;
    }
}

impl From<()> for SeedInput {
    fn from(_: ()) -> Self {
        SeedInput::None
    }
}

impl From<u32> for SeedInput {
    fn from(v: u32) -> Self {
        SeedInput::Words(vec![v])
    }
}

impl From<i32> for SeedInput {
    fn from(v: i32) -> Self {
        if v < 0 {
            SeedInput::Invalid(format!("Seed must be non-negative, got {v}"))
        } else {
            SeedInput::Words(vec![v as u32])
        }
    }
}

impl From<u64> for SeedInput {
    fn from(v: u64) -> Self {
        let mut w = Vec::with_capacity(2);
        push_u64_words(&mut w, v);
        SeedInput::Words(w)
    }
}

impl From<i64> for SeedInput {
    fn from(v: i64) -> Self {
        if v < 0 {
            SeedInput::Invalid(format!("Seed must be non-negative, got {v}"))
        } else {
            SeedInput::from(v as u64)
        }
    }
}

impl From<u128> for SeedInput {
    fn from(v: u128) -> Self {
        let mut w = Vec::with_capacity(4);
        push_u128_words(&mut w, v);
        SeedInput::Words(w)
    }
}

impl From<i128> for SeedInput {
    fn from(v: i128) -> Self {
        if v < 0 {
            SeedInput::Invalid(format!("Seed must be non-negative, got {v}"))
        } else {
            SeedInput::from(v as u128)
        }
    }
}

impl From<usize> for SeedInput {
    fn from(v: usize) -> Self {
        SeedInput::from(v as u64)
    }
}

impl From<isize> for SeedInput {
    fn from(v: isize) -> Self {
        if v < 0 {
            SeedInput::Invalid(format!("Seed must be non-negative, got {v}"))
        } else {
            SeedInput::from(v as u64)
        }
    }
}

impl<T: Into<SeedInput>> From<Option<T>> for SeedInput {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(inner) => inner.into(),
            Option::None => SeedInput::None,
        }
    }
}

impl From<&[u32]> for SeedInput {
    fn from(s: &[u32]) -> Self {
        SeedInput::Words(s.to_vec())
    }
}

impl From<Vec<u32>> for SeedInput {
    fn from(v: Vec<u32>) -> Self {
        SeedInput::Words(v)
    }
}

impl<const N: usize> From<[u32; N]> for SeedInput {
    fn from(arr: [u32; N]) -> Self {
        SeedInput::Words(arr.to_vec())
    }
}

impl From<&[u64]> for SeedInput {
    fn from(s: &[u64]) -> Self {
        let mut w = Vec::with_capacity(s.len() * 2);
        for &x in s {
            push_u64_words(&mut w, x);
        }
        SeedInput::Words(w)
    }
}

impl From<Vec<u64>> for SeedInput {
    fn from(v: Vec<u64>) -> Self {
        SeedInput::from(v.as_slice())
    }
}

impl<const N: usize> From<[u64; N]> for SeedInput {
    fn from(arr: [u64; N]) -> Self {
        SeedInput::from(arr.as_slice())
    }
}

impl From<&[i64]> for SeedInput {
    fn from(s: &[i64]) -> Self {
        let mut w = Vec::with_capacity(s.len() * 2);
        for &x in s {
            if x < 0 {
                return SeedInput::Invalid(format!("Seed sequence must be non-negative, got {x}"));
            }
            push_u64_words(&mut w, x as u64);
        }
        SeedInput::Words(w)
    }
}

impl From<Vec<i64>> for SeedInput {
    fn from(v: Vec<i64>) -> Self {
        SeedInput::from(v.as_slice())
    }
}

impl<const N: usize> From<[i64; N]> for SeedInput {
    fn from(arr: [i64; N]) -> Self {
        SeedInput::from(arr.as_slice())
    }
}

impl From<&[i32]> for SeedInput {
    fn from(s: &[i32]) -> Self {
        let mut w = Vec::with_capacity(s.len());
        for &x in s {
            if x < 0 {
                return SeedInput::Invalid(format!("Seed sequence must be non-negative, got {x}"));
            }
            w.push(x as u32);
        }
        SeedInput::Words(w)
    }
}

impl From<Vec<i32>> for SeedInput {
    fn from(v: Vec<i32>) -> Self {
        SeedInput::from(v.as_slice())
    }
}

impl<const N: usize> From<[i32; N]> for SeedInput {
    fn from(arr: [i32; N]) -> Self {
        SeedInput::from(arr.as_slice())
    }
}

impl From<&[u128]> for SeedInput {
    fn from(s: &[u128]) -> Self {
        let mut w = Vec::with_capacity(s.len() * 4);
        for &x in s {
            push_u128_words(&mut w, x);
        }
        SeedInput::Words(w)
    }
}

impl From<Vec<u128>> for SeedInput {
    fn from(v: Vec<u128>) -> Self {
        SeedInput::from(v.as_slice())
    }
}

#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Generate 128 bits of OS-backed entropy using std's OS-seeded `RandomState`
/// combined with monotonic counter, process id, stack address, and high-res clock.
fn os_entropy_words() -> Vec<u32> {
    let mut hasher1 = RandomState::new().build_hasher();
    let mut hasher2 = RandomState::new().build_hasher();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let cnt = ENTROPY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id() as u64;
    let stack_addr = (&nanos as *const u64) as usize as u64;

    hasher1.write_u64(nanos);
    hasher1.write_u64(cnt);
    hasher2.write_u64(pid);
    hasher2.write_u64(stack_addr);

    let mut s1 = hasher1.finish() ^ cnt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut s2 = hasher2.finish() ^ nanos;
    let lo = splitmix64_next(&mut s1);
    let hi = splitmix64_next(&mut s2);
    vec![
        lo as u32,
        (lo >> 32) as u32,
        hi as u32,
        (hi >> 32) as u32,
    ]
}

/// NumPy `SeedSequence`: mixes arbitrary-size entropy into a fixed-size pool
/// using Melissa O'Neill's hash-and-mix algorithm and supports independent
/// cryptographic-style `spawn()` streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequence {
    entropy: Vec<u32>,
    spawn_key: Vec<u32>,
    pool_size: usize,
    n_children_spawned: u32,
    pool: Vec<u32>,
}

impl SeedSequence {
    /// Create a new `SeedSequence` seeded from OS entropy.
    pub fn new() -> Self {
        Self::with_options(SeedInput::None, Vec::new(), DEFAULT_POOL_SIZE, 0)
            .expect("default SeedSequence options are valid")
    }

    /// Create a new `SeedSequence` from any seed input (`u64`, `u128`, `&[u32]`, etc.).
    pub fn from_seed(entropy: impl Into<SeedInput>) -> Result<Self> {
        Self::with_options(entropy, Vec::new(), DEFAULT_POOL_SIZE, 0)
    }

    /// Create a `SeedSequence` with explicit `spawn_key`, `pool_size`, and `n_children_spawned`.
    pub fn with_options(
        entropy: impl Into<SeedInput>,
        spawn_key: Vec<u32>,
        pool_size: usize,
        n_children_spawned: u32,
    ) -> Result<Self> {
        if pool_size < DEFAULT_POOL_SIZE {
            return Err(NumPyError::InvalidValue(format!(
                "The size of the entropy pool should be at least {DEFAULT_POOL_SIZE}, got {pool_size}"
            )));
        }
        let entropy_words = match entropy.into() {
            SeedInput::None => os_entropy_words(),
            SeedInput::Words(w) => w,
            SeedInput::Invalid(msg) => return Err(NumPyError::InvalidValue(msg)),
        };

        let mut seq = Self {
            entropy: entropy_words,
            spawn_key,
            pool_size,
            n_children_spawned,
            pool: Vec::new(),
        };
        let assembled = seq.get_assembled_entropy();
        seq.pool = Self::mix_entropy(&assembled, seq.pool_size);
        Ok(seq)
    }

    /// Original base 32-bit entropy words.
    pub fn entropy(&self) -> &[u32] {
        &self.entropy
    }

    /// Spawn path identifying this child stream.
    pub fn spawn_key(&self) -> &[u32] {
        &self.spawn_key
    }

    /// Pool size in 32-bit words.
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Current mixed entropy pool.
    pub fn pool(&self) -> &[u32] {
        &self.pool
    }

    /// Number of child sequences already spawned from this instance.
    pub fn n_children_spawned(&self) -> u32 {
        self.n_children_spawned
    }

    fn get_assembled_entropy(&self) -> Vec<u32> {
        let mut run_entropy = self.entropy.clone();
        if !self.spawn_key.is_empty() && run_entropy.len() < self.pool_size {
            run_entropy.resize(self.pool_size, 0);
        }
        run_entropy.extend_from_slice(&self.spawn_key);
        run_entropy
    }

    fn mix_entropy(entropy_array: &[u32], pool_size: usize) -> Vec<u32> {
        let mut mixer = vec![0u32; pool_size];
        let mut hash_const = INIT_A;

        let hashmix = |value: u32, hc: &mut u32| -> u32 {
            let mut v = value ^ *hc;
            *hc = hc.wrapping_mul(MULT_A);
            v = v.wrapping_mul(*hc);
            v ^= v >> XSHIFT;
            v
        };

        let mix = |x: u32, y: u32| -> u32 {
            let mut res = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
            res ^= res >> XSHIFT;
            res
        };

        for (i, slot) in mixer.iter_mut().enumerate() {
            let val = if i < entropy_array.len() {
                entropy_array[i]
            } else {
                0
            };
            *slot = hashmix(val, &mut hash_const);
        }

        for i_src in 0..pool_size {
            for i_dst in 0..pool_size {
                if i_src != i_dst {
                    let hashed = hashmix(mixer[i_src], &mut hash_const);
                    mixer[i_dst] = mix(mixer[i_dst], hashed);
                }
            }
        }

        for &ent in entropy_array.iter().skip(pool_size) {
            for dst in &mut mixer {
                let hashed = hashmix(ent, &mut hash_const);
                *dst = mix(*dst, hashed);
            }
        }

        mixer
    }

    /// Generate `n_words` reproducible 32-bit state words from the mixed entropy pool.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut state = Vec::with_capacity(n_words);
        let mut hash_const = INIT_B;
        let plen = self.pool.len();
        for i in 0..n_words {
            let mut data_val = self.pool[i % plen];
            data_val ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data_val = data_val.wrapping_mul(hash_const);
            data_val ^= data_val >> XSHIFT;
            state.push(data_val);
        }
        state
    }

    /// Generate `n_words` reproducible 64-bit state words by pairing little-endian 32-bit words.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let words32 = self.generate_state_u32(n_words * 2);
        let mut out = Vec::with_capacity(n_words);
        for chunk in words32.chunks_exact(2) {
            let lo = chunk[0] as u64;
            let hi = chunk[1] as u64;
            out.push(lo | (hi << 32));
        }
        out
    }

    /// Create `n_children` independent child `SeedSequence` objects.
    pub fn spawn(&mut self, n_children: usize) -> Result<Vec<SeedSequence>> {
        let mut children = Vec::with_capacity(n_children);
        for _ in 0..n_children {
            let child_idx = self.n_children_spawned;
            self.n_children_spawned = self
                .n_children_spawned
                .checked_add(1)
                .ok_or_else(|| {
                    NumPyError::InvalidValue(
                        "Maximum number of SeedSequence children exceeded".into(),
                    )
                })?;
            let mut child_key = Vec::with_capacity(self.spawn_key.len() + 1);
            child_key.extend_from_slice(&self.spawn_key);
            child_key.push(child_idx);
            let child = SeedSequence::with_options(
                SeedInput::Words(self.entropy.clone()),
                child_key,
                self.pool_size,
                0,
            )?;
            children.push(child);
        }
        Ok(children)
    }
}

impl Default for SeedSequence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_sequence_deadbeaf_matches_numpy() {
        let ss = SeedSequence::from_seed(0xdeadbeafu64).unwrap();
        let s64 = ss.generate_state_u64(4);
        assert_eq!(
            s64,
            vec![
                0x5031_2a63_d516_e24a,
                0x4063_bc8b_ca44_dc77,
                0xb780_f022_ab4e_c9e2,
                0x2056_086e_8249_885a,
            ]
        );
    }

    #[test]
    fn seed_sequence_rejects_negative() {
        assert!(SeedSequence::from_seed(-1i64).is_err());
        assert!(SeedSequence::from_seed([-5i32, 1]).is_err());
    }
}
