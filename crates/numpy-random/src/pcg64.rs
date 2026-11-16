use crate::bit_generator::BitGenerator;
use crate::seed_sequence::{SeedInput, SeedSequence};
use numpy_types::Result;

const PCG_DEFAULT_MULTIPLIER_128: u128 =
    (2549297995355413924u128 << 64) | 4865540595714422341u128;
const PCG_CHEAP_MULTIPLIER_128: u64 = 0xda94_2042_e4dd_58b5;
const PCG_JUMP_STEP: u128 = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835u128;

#[inline]
fn advance_lcg_128(state: u128, mult: u128, inc: u128, mut delta: u128) -> u128 {
    let mut cur_mult = mult;
    let mut cur_plus = inc;
    let mut acc_mult: u128 = 1;
    let mut acc_plus: u128 = 0;
    while delta > 0 {
        if (delta & 1) != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }
    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}

/// Internal state snapshot for [`PCG64`] and [`PCG64DXSM`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg64State {
    pub state: u128,
    pub inc: u128,
    pub has_uint32: bool,
    pub uinteger: u32,
}

/// NumPy `PCG64` BitGenerator (PCG-XSL-RR-128/64).
#[derive(Debug, Clone)]
pub struct PCG64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
    seed_seq: Option<SeedSequence>,
}

impl PCG64 {
    /// Construct a `PCG64` generator seeded from OS entropy.
    pub fn new() -> Self {
        Self::from_seed(SeedInput::None).expect("OS-seeded PCG64 cannot fail")
    }

    /// Construct a `PCG64` generator from any [`SeedInput`].
    pub fn from_seed(seed: impl Into<SeedInput>) -> Result<Self> {
        let seed_seq = SeedSequence::from_seed(seed)?;
        Self::from_seed_seq(seed_seq)
    }

    /// Construct a `PCG64` generator directly from 128-bit state and sequence stream id.
    pub fn from_raw_state(state: u128, inc: u128, has_uint32: bool, uinteger: u32) -> Self {
        Self {
            state,
            inc: inc | 1,
            has_uint32,
            uinteger,
            seed_seq: None,
        }
    }

    /// Return a copy of the generator's internal state.
    pub fn state(&self) -> Pcg64State {
        Pcg64State {
            state: self.state,
            inc: self.inc,
            has_uint32: self.has_uint32,
            uinteger: self.uinteger,
        }
    }

    /// Restore the generator's internal state.
    pub fn set_state(&mut self, state: Pcg64State) {
        self.state = state.state;
        self.inc = state.inc | 1;
        self.has_uint32 = state.has_uint32;
        self.uinteger = state.uinteger;
    }

    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_DEFAULT_MULTIPLIER_128)
            .wrapping_add(self.inc);
    }

    #[inline]
    fn output_xsl_rr(s: u128) -> u64 {
        let xored = ((s >> 64) as u64) ^ (s as u64);
        xored.rotate_right((s >> 122) as u32)
    }

    /// Advance the underlying LCG by `delta` steps in O(log delta) time.
    pub fn advance(&mut self, delta: u128) -> &mut Self {
        self.state = advance_lcg_128(self.state, PCG_DEFAULT_MULTIPLIER_128, self.inc, delta);
        self.has_uint32 = false;
        self.uinteger = 0;
        self
    }

    /// Return a new `PCG64` jumped ahead by `jumps` golden-ratio 128-bit steps.
    pub fn jumped(&self, jumps: u64) -> Self {
        let delta = PCG_JUMP_STEP.wrapping_mul(jumps as u128);
        let mut out = Self {
            state: self.state,
            inc: self.inc,
            has_uint32: false,
            uinteger: 0,
            seed_seq: None,
        };
        out.advance(delta);
        out
    }
}

impl Default for PCG64 {
    fn default() -> Self {
        Self::new()
    }
}

impl BitGenerator for PCG64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.step();
        Self::output_xsl_rr(self.state)
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
        let val = seed_seq.generate_state_u64(4);
        let initstate = ((val[0] as u128) << 64) | (val[1] as u128);
        let initseq = ((val[2] as u128) << 64) | (val[3] as u128);
        let inc = (initseq << 1) | 1;
        let mut rng = Self {
            state: 0,
            inc,
            has_uint32: false,
            uinteger: 0,
            seed_seq: Some(seed_seq),
        };
        // Upstream pcg64_cm_set_seed uses pcg_setseq_128_step_r (DEFAULT_MULTIPLIER_128) during seeding.
        let seed_step = |s: &mut u128, inc: u128| {
            *s = s
                .wrapping_mul(PCG_DEFAULT_MULTIPLIER_128)
                .wrapping_add(inc);
        };
        seed_step(&mut rng.state, rng.inc);
        rng.state = rng.state.wrapping_add(initstate);
        seed_step(&mut rng.state, rng.inc);
        Ok(rng)
    }
}

/// NumPy `PCG64DXSM` BitGenerator (PCG-DXSM-128/64 with cheap 64-bit multiplier).
#[derive(Debug, Clone)]
pub struct PCG64DXSM {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
    seed_seq: Option<SeedSequence>,
}

impl PCG64DXSM {
    /// Construct a `PCG64DXSM` generator seeded from OS entropy.
    pub fn new() -> Self {
        Self::from_seed(SeedInput::None).expect("OS-seeded PCG64DXSM cannot fail")
    }

    /// Construct a `PCG64DXSM` generator from any [`SeedInput`].
    pub fn from_seed(seed: impl Into<SeedInput>) -> Result<Self> {
        let seed_seq = SeedSequence::from_seed(seed)?;
        Self::from_seed_seq(seed_seq)
    }

    /// Construct a `PCG64DXSM` generator directly from internal state.
    pub fn from_raw_state(state: u128, inc: u128, has_uint32: bool, uinteger: u32) -> Self {
        Self {
            state,
            inc: inc | 1,
            has_uint32,
            uinteger,
            seed_seq: None,
        }
    }

    /// Return a copy of the generator's internal state.
    pub fn state(&self) -> Pcg64State {
        Pcg64State {
            state: self.state,
            inc: self.inc,
            has_uint32: self.has_uint32,
            uinteger: self.uinteger,
        }
    }

    /// Restore the generator's internal state.
    pub fn set_state(&mut self, state: Pcg64State) {
        self.state = state.state;
        self.inc = state.inc | 1;
        self.has_uint32 = state.has_uint32;
        self.uinteger = state.uinteger;
    }

    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_CHEAP_MULTIPLIER_128 as u128)
            .wrapping_add(self.inc);
    }

    #[inline]
    fn output_dxsm(s: u128) -> u64 {
        let mut hi = (s >> 64) as u64;
        let lo = (s as u64) | 1;
        hi ^= hi >> 32;
        hi = hi.wrapping_mul(PCG_CHEAP_MULTIPLIER_128);
        hi ^= hi >> 48;
        hi.wrapping_mul(lo)
    }

    /// Advance the underlying LCG by `delta` steps in O(log delta) time.
    pub fn advance(&mut self, delta: u128) -> &mut Self {
        self.state = advance_lcg_128(
            self.state,
            PCG_CHEAP_MULTIPLIER_128 as u128,
            self.inc,
            delta,
        );
        self.has_uint32 = false;
        self.uinteger = 0;
        self
    }

    /// Return a new `PCG64DXSM` jumped ahead by `jumps` golden-ratio steps.
    pub fn jumped(&self, jumps: u64) -> Self {
        let delta = PCG_JUMP_STEP.wrapping_mul(jumps as u128);
        let mut out = Self {
            state: self.state,
            inc: self.inc,
            has_uint32: false,
            uinteger: 0,
            seed_seq: None,
        };
        out.advance(delta);
        out
    }
}

impl Default for PCG64DXSM {
    fn default() -> Self {
        Self::new()
    }
}

impl BitGenerator for PCG64DXSM {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let out = Self::output_dxsm(self.state);
        self.step();
        out
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
        let val = seed_seq.generate_state_u64(4);
        let initstate = ((val[0] as u128) << 64) | (val[1] as u128);
        let initseq = ((val[2] as u128) << 64) | (val[3] as u128);
        let inc = (initseq << 1) | 1;
        let mut rng = Self {
            state: 0,
            inc,
            has_uint32: false,
            uinteger: 0,
            seed_seq: Some(seed_seq),
        };
        // Upstream pcg64_cm_set_seed uses pcg_setseq_128_step_r (DEFAULT_MULTIPLIER_128) during seeding.
        let seed_step = |s: &mut u128, inc: u128| {
            *s = s
                .wrapping_mul(PCG_DEFAULT_MULTIPLIER_128)
                .wrapping_add(inc);
        };
        seed_step(&mut rng.state, rng.inc);
        rng.state = rng.state.wrapping_add(initstate);
        seed_step(&mut rng.state, rng.inc);
        Ok(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg64_deadbeaf_testset() {
        let mut bg = PCG64::from_seed(0xdeadbeafu64).unwrap();
        let vals = bg.random_raw(4);
        assert_eq!(
            vals,
            vec![
                0x60d2_4054_e17a_0698,
                0xd5e7_9d89_856e_4f12,
                0xd254_972f_e64b_d782,
                0xf1e3_072a_53c7_2571,
            ]
        );
    }

    #[test]
    fn pcg64dxsm_deadbeaf_testset() {
        let mut bg = PCG64DXSM::from_seed(0xdeadbeafu64).unwrap();
        let vals = bg.random_raw(4);
        assert_eq!(
            vals,
            vec![
                0xdf1d_dcf1_e225_21fe,
                0xc71b_2f9c_706c_f151,
                0x6922_a8cc_24ad_96b2,
                0x8273_8c54_9bec_cc30,
            ]
        );
    }
}
