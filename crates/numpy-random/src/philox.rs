use crate::bit_generator::BitGenerator;
use crate::seed_sequence::{SeedInput, SeedSequence};
use numpy_types::{NumPyError, Result};

const PHILOX_M4X64_0: u64 = 0xD2E7_470E_E14C_6C93;
const PHILOX_M4X64_1: u64 = 0xCA5A_8263_9512_1157;
const PHILOX_W64_0: u64 = 0x9E37_79B9_7F4A_7C15;
const PHILOX_W64_1: u64 = 0xBB67_AE85_84CA_A73B;

/// Internal state snapshot for [`Philox`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiloxState {
    pub counter: [u64; 4],
    pub key: [u64; 2],
    pub buffer: [u64; 4],
    pub buffer_pos: usize,
    pub has_uint32: bool,
    pub uinteger: u32,
}

/// NumPy `Philox` BitGenerator (Philox4x64-10 counter-based PRNG).
#[derive(Debug, Clone)]
pub struct Philox {
    ctr: [u64; 4],
    key: [u64; 2],
    buffer: [u64; 4],
    buffer_pos: usize,
    has_uint32: bool,
    uinteger: u32,
    seed_seq: Option<SeedSequence>,
}

impl Philox {
    /// Construct a `Philox` generator seeded from OS entropy.
    pub fn new() -> Self {
        Self::from_seed(SeedInput::None).expect("OS-seeded Philox cannot fail")
    }

    /// Construct a `Philox` generator from any [`SeedInput`].
    pub fn from_seed(seed: impl Into<SeedInput>) -> Result<Self> {
        let seed_seq = SeedSequence::from_seed(seed)?;
        Self::from_seed_seq(seed_seq)
    }

    /// Construct a `Philox` generator directly from explicit key and optional counter.
    pub fn from_key_counter(key: [u64; 2], counter: Option<[u64; 4]>) -> Self {
        Self {
            ctr: counter.unwrap_or([0; 4]),
            key,
            buffer: [0; 4],
            buffer_pos: 4,
            has_uint32: false,
            uinteger: 0,
            seed_seq: None,
        }
    }

    /// Return a copy of the generator's internal state.
    pub fn state(&self) -> PhiloxState {
        PhiloxState {
            counter: self.ctr,
            key: self.key,
            buffer: self.buffer,
            buffer_pos: self.buffer_pos,
            has_uint32: self.has_uint32,
            uinteger: self.uinteger,
        }
    }

    /// Restore the generator's internal state.
    pub fn set_state(&mut self, state: PhiloxState) -> Result<()> {
        if state.buffer_pos > 4 {
            return Err(NumPyError::InvalidValue(format!(
                "Philox buffer_pos must be in 0..=4, got {}",
                state.buffer_pos
            )));
        }
        self.ctr = state.counter;
        self.key = state.key;
        self.buffer = state.buffer;
        self.buffer_pos = state.buffer_pos;
        self.has_uint32 = state.has_uint32;
        self.uinteger = state.uinteger;
        Ok(())
    }

    #[inline]
    fn mulhilo64(a: u64, b: u64) -> (u64, u64) {
        let prod = (a as u128) * (b as u128);
        ((prod >> 64) as u64, prod as u64)
    }

    #[inline]
    fn round(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
        let (hi0, lo0) = Self::mulhilo64(PHILOX_M4X64_0, ctr[0]);
        let (hi1, lo1) = Self::mulhilo64(PHILOX_M4X64_1, ctr[2]);
        [
            hi1 ^ ctr[1] ^ key[0],
            lo1,
            hi0 ^ ctr[3] ^ key[1],
            lo0,
        ]
    }

    #[inline]
    fn bumpkey(key: &mut [u64; 2]) {
        key[0] = key[0].wrapping_add(PHILOX_W64_0);
        key[1] = key[1].wrapping_add(PHILOX_W64_1);
    }

    #[inline]
    fn philox4x64_10(mut ctr: [u64; 4], mut key: [u64; 2]) -> [u64; 4] {
        ctr = Self::round(ctr, key);
        for _ in 1..10 {
            Self::bumpkey(&mut key);
            ctr = Self::round(ctr, key);
        }
        ctr
    }

    #[inline]
    fn increment_counter(&mut self) {
        self.ctr[0] = self.ctr[0].wrapping_add(1);
        if self.ctr[0] == 0 {
            self.ctr[1] = self.ctr[1].wrapping_add(1);
            if self.ctr[1] == 0 {
                self.ctr[2] = self.ctr[2].wrapping_add(1);
                if self.ctr[2] == 0 {
                    self.ctr[3] = self.ctr[3].wrapping_add(1);
                }
            }
        }
    }

    /// Add a 256-bit counter delta `[d0, d1, d2, d3]` (little-endian 64-bit limbs)
    /// to the internal counter and reset cached output buffers.
    pub fn advance(&mut self, delta: [u64; 4]) -> &mut Self {
        let mut carry: u128 = 0;
        for i in 0..4 {
            let sum = (self.ctr[i] as u128) + (delta[i] as u128) + carry;
            self.ctr[i] = sum as u64;
            carry = sum >> 64;
        }
        self.buffer_pos = 4;
        self.buffer = [0; 4];
        self.has_uint32 = false;
        self.uinteger = 0;
        self
    }

    /// Convenience helper for advancing the counter by a scalar `u128` step.
    pub fn advance_u128(&mut self, delta: u128) -> &mut Self {
        self.advance([delta as u64, (delta >> 64) as u64, 0, 0])
    }

    /// Jump this generator forward in place by `2^128` counter blocks (NumPy `philox_jump`).
    pub fn jump_in_place(&mut self) {
        self.ctr[2] = self.ctr[2].wrapping_add(1);
        if self.ctr[2] == 0 {
            self.ctr[3] = self.ctr[3].wrapping_add(1);
        }
        self.buffer_pos = 4;
        self.buffer = [0; 4];
        self.has_uint32 = false;
        self.uinteger = 0;
        self.seed_seq = None;
    }

    /// Return a new `Philox` jumped ahead by `jumps * 2^128` counter blocks.
    pub fn jumped(&self, jumps: u64) -> Self {
        let mut out = self.clone();
        out.seed_seq = None;
        let sum = (out.ctr[2] as u128) + (jumps as u128);
        out.ctr[2] = sum as u64;
        out.ctr[3] = out.ctr[3].wrapping_add((sum >> 64) as u64);
        out.buffer_pos = 4;
        out.buffer = [0; 4];
        out.has_uint32 = false;
        out.uinteger = 0;
        out
    }
}

impl Default for Philox {
    fn default() -> Self {
        Self::new()
    }
}

impl BitGenerator for Philox {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        if self.buffer_pos >= 4 {
            self.increment_counter();
            self.buffer = Self::philox4x64_10(self.ctr, self.key);
            self.buffer_pos = 0;
        }
        let val = self.buffer[self.buffer_pos];
        self.buffer_pos += 1;
        val
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            self.uinteger
        } else {
            let next = self.next_u64();
            self.has_uint32 = true;
            self.uinteger = (next >> 32) as u32;
            next as u32
        }
    }

    fn seed_seq(&self) -> Option<&SeedSequence> {
        self.seed_seq.as_ref()
    }

    fn seed_seq_mut(&mut self) -> Option<&mut SeedSequence> {
        self.seed_seq.as_mut()
    }

    fn from_seed_seq(seed_seq: SeedSequence) -> Result<Self> {
        let k = seed_seq.generate_state_u64(2);
        Ok(Self {
            ctr: [0; 4],
            key: [k[0], k[1]],
            buffer: [0; 4],
            buffer_pos: 4,
            has_uint32: false,
            uinteger: 0,
            seed_seq: Some(seed_seq),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_deadbeaf_matches_numpy() {
        let mut bg = Philox::from_seed(0xdeadbeafu64).unwrap();
        let vals = bg.random_raw(4);
        assert_eq!(
            vals,
            vec![
                0xedc9_5200_e2bd_66a5,
                0x581d_4e43_b768_2352,
                0x4be7_278f_5e37_3eab,
                0xee47_f179_91a9_e7ea,
            ]
        );
    }
}
