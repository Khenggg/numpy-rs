use crate::bit_generator::{BitGenerator, INV_2_POW_53};
use crate::mt19937_constants::POLY_COEF;
use crate::seed_sequence::{SeedInput, SeedSequence};
use numpy_types::{NumPyError, Result};

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// Internal state snapshot for [`MT19937`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mt19937State {
    pub key: [u32; N],
    pub pos: usize,
}

/// NumPy `MT19937` BitGenerator (Mersenne Twister 19937 with polynomial jump ahead).
#[derive(Debug, Clone)]
pub struct MT19937 {
    key: [u32; N],
    pos: usize,
    seed_seq: Option<SeedSequence>,
}

impl MT19937 {
    /// Construct an `MT19937` generator seeded from OS entropy via [`SeedSequence`].
    pub fn new() -> Self {
        Self::from_seed(SeedInput::None).expect("OS-seeded MT19937 cannot fail")
    }

    /// Construct an `MT19937` generator from any [`SeedInput`] via [`SeedSequence`].
    pub fn from_seed(seed: impl Into<SeedInput>) -> Result<Self> {
        let seed_seq = SeedSequence::from_seed(seed)?;
        Self::from_seed_seq(seed_seq)
    }

    /// Construct an `MT19937` generator using NumPy's legacy seeding algorithm (`RandomState`).
    pub fn from_legacy_seed(seed: u32) -> Self {
        let mut rng = Self {
            key: [0u32; N],
            pos: N,
            seed_seq: None,
        };
        rng.legacy_seed(seed);
        rng
    }

    /// Construct an `MT19937` generator using legacy array seeding (`mt19937_init_by_array`).
    pub fn from_legacy_key(init_key: &[u32]) -> Self {
        let mut rng = Self {
            key: [0u32; N],
            pos: N,
            seed_seq: None,
        };
        rng.legacy_seed_array(init_key);
        rng
    }

    /// Construct an `MT19937` generator directly from its state buffer and position.
    pub fn from_raw_state(key: [u32; N], pos: usize) -> Result<Self> {
        if pos > N {
            return Err(NumPyError::InvalidValue(format!(
                "MT19937 pos must be in 0..={N}, got {pos}"
            )));
        }
        Ok(Self {
            key,
            pos,
            seed_seq: None,
        })
    }

    /// Re-seed in place with legacy scalar algorithm (`RandomState.seed(n)`).
    pub fn legacy_seed(&mut self, mut seed: u32) {
        for pos in 0..N {
            seed &= 0xffff_ffff;
            self.key[pos] = seed;
            seed = 1812433253u32
                .wrapping_mul(seed ^ (seed >> 30))
                .wrapping_add((pos as u32) + 1);
        }
        self.pos = N;
        self.seed_seq = None;
    }

    /// Re-seed in place with legacy array algorithm (`init_by_array`).
    pub fn legacy_seed_array(&mut self, init_key: &[u32]) {
        self.legacy_seed(19650218);
        if init_key.is_empty() {
            return;
        }
        let key_length = init_key.len();
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k = N.max(key_length);
        while k > 0 {
            let prev = self.key[i - 1];
            self.key[i] = (self.key[i]
                ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
            .wrapping_add(init_key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.key[0] = self.key[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
            k -= 1;
        }
        for _ in 0..(N - 1) {
            let prev = self.key[i - 1];
            self.key[i] = (self.key[i]
                ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.key[0] = self.key[N - 1];
                i = 1;
            }
        }
        self.key[0] = 0x8000_0000;
        self.pos = N;
        self.seed_seq = None;
    }

    /// Return a copy of the generator's internal state.
    pub fn state(&self) -> Mt19937State {
        Mt19937State {
            key: self.key,
            pos: self.pos,
        }
    }

    /// Restore the generator's internal state.
    pub fn set_state(&mut self, state: Mt19937State) -> Result<()> {
        if state.pos > N {
            return Err(NumPyError::InvalidValue(format!(
                "MT19937 pos must be in 0..={N}, got {}",
                state.pos
            )));
        }
        self.key = state.key;
        self.pos = state.pos;
        Ok(())
    }

    fn twist(&mut self) {
        let mut y: u32;
        for i in 0..(N - M) {
            y = (self.key[i] & UPPER_MASK) | (self.key[i + 1] & LOWER_MASK);
            self.key[i] = self.key[i + M] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
        }
        for i in (N - M)..(N - 1) {
            y = (self.key[i] & UPPER_MASK) | (self.key[i + 1] & LOWER_MASK);
            self.key[i] =
                self.key[i + M - N] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
        }
        y = (self.key[N - 1] & UPPER_MASK) | (self.key[0] & LOWER_MASK);
        self.key[N - 1] = self.key[M - 1] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
        self.pos = 0;
    }

    #[inline]
    pub(crate) fn next_u32_raw(&mut self) -> u32 {
        if self.pos == N {
            self.twist();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    #[inline]
    fn get_coef(deg: usize) -> bool {
        (POLY_COEF[deg >> 5] & (1u32 << (deg & 0x1f))) != 0
    }

    fn gen_next_single(key: &mut [u32; N], pos: &mut usize) {
        let num = *pos;
        if num < N - M {
            let y = (key[num] & UPPER_MASK) | (key[num + 1] & LOWER_MASK);
            key[num] = key[num + M] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
            *pos += 1;
        } else if num < N - 1 {
            let y = (key[num] & UPPER_MASK) | (key[num + 1] & LOWER_MASK);
            key[num] = key[num + M - N] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
            *pos += 1;
        } else {
            let y = (key[N - 1] & UPPER_MASK) | (key[0] & LOWER_MASK);
            key[N - 1] = key[M - 1] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
            *pos = 0;
        }
    }

    fn add_state(dst_key: &mut [u32; N], dst_pos: usize, src_key: &[u32; N], src_pos: usize) {
        let pt1 = dst_pos;
        let pt2 = src_pos;
        if pt2 >= pt1 {
            let mut i = 0;
            while i < N - pt2 {
                dst_key[i + pt1] ^= src_key[i + pt2];
                i += 1;
            }
            while i < N - pt1 {
                dst_key[i + pt1] ^= src_key[i + pt2 - N];
                i += 1;
            }
            while i < N {
                dst_key[i + pt1 - N] ^= src_key[i + pt2 - N];
                i += 1;
            }
        } else {
            let mut i = 0;
            while i < N - pt1 {
                dst_key[i + pt1] ^= src_key[i + pt2];
                i += 1;
            }
            while i < N - pt2 {
                dst_key[i + pt1 - N] ^= src_key[i + pt2];
                i += 1;
            }
            while i < N {
                dst_key[i + pt1 - N] ^= src_key[i + pt2 - N];
                i += 1;
            }
        }
    }

    /// Jump this generator ahead by `2^128` steps using the precomputed GF(2) jump polynomial.
    pub fn jump_in_place(&mut self) {
        if self.pos >= N {
            self.pos = 0;
        }
        let mut temp_key = [0u32; N];
        let mut temp_pos = 0usize;
        let mut i: isize = 19937 - 1;
        while i >= 0 && !Self::get_coef(i as usize) {
            i -= 1;
        }
        if i > 0 {
            temp_key = self.key;
            temp_pos = self.pos;
            Self::gen_next_single(&mut temp_key, &mut temp_pos);
            i -= 1;
            while i > 0 {
                if Self::get_coef(i as usize) {
                    Self::add_state(&mut temp_key, temp_pos, &self.key, self.pos);
                }
                Self::gen_next_single(&mut temp_key, &mut temp_pos);
                i -= 1;
            }
            if Self::get_coef(0) {
                Self::add_state(&mut temp_key, temp_pos, &self.key, self.pos);
            }
        } else if i == 0 {
            temp_key = self.key;
            temp_pos = self.pos;
        }
        self.key = temp_key;
        self.pos = temp_pos;
        self.seed_seq = None;
    }

    /// Return a new `MT19937` jumped ahead by `jumps * 2^128` steps.
    pub fn jumped(&self, jumps: u64) -> Self {
        let mut out = self.clone();
        out.seed_seq = None;
        for _ in 0..jumps {
            out.jump_in_place();
        }
        out
    }
}

impl Default for MT19937 {
    fn default() -> Self {
        Self::new()
    }
}

impl BitGenerator for MT19937 {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.next_u32_raw()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        ((self.next_u32_raw() as u64) << 32) | (self.next_u32_raw() as u64)
    }

    #[inline]
    fn next_double(&mut self) -> f64 {
        let a = (self.next_u32_raw() >> 5) as f64;
        let b = (self.next_u32_raw() >> 6) as f64;
        (a * 67108864.0 + b) * INV_2_POW_53
    }

    #[inline]
    fn next_raw(&mut self) -> u64 {
        self.next_u32_raw() as u64
    }

    fn seed_seq(&self) -> Option<&SeedSequence> {
        self.seed_seq.as_ref()
    }

    fn seed_seq_mut(&mut self) -> Option<&mut SeedSequence> {
        self.seed_seq.as_mut()
    }

    fn from_seed_seq(seed_seq: SeedSequence) -> Result<Self> {
        let val = seed_seq.generate_state_u32(N);
        let mut key = [0u32; N];
        key.copy_from_slice(&val);
        key[0] = 0x8000_0000;
        Ok(Self {
            key,
            pos: N - 1,
            seed_seq: Some(seed_seq),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_deadbeaf_testset() {
        let mut bg = MT19937::from_seed(0xdeadbeafu64).unwrap();
        let vals = bg.random_raw(4);
        assert_eq!(vals, vec![0xc816_921f, 0xb362_3c6d, 0x5fa3_91bb, 0x0401_78d9]);
    }
}
