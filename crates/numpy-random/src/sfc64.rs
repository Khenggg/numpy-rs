use crate::bit_generator::BitGenerator;
use crate::seed_sequence::{SeedInput, SeedSequence};
use numpy_types::Result;

/// Internal state snapshot for [`SFC64`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sfc64State {
    pub state: [u64; 4],
    pub has_uint32: bool,
    pub uinteger: u32,
}

/// NumPy `SFC64` BitGenerator (Chris Doty-Humphrey's Small Fast Chaotic 64-bit PRNG).
#[derive(Debug, Clone)]
pub struct SFC64 {
    s: [u64; 4],
    has_uint32: bool,
    uinteger: u32,
    seed_seq: Option<SeedSequence>,
}

impl SFC64 {
    /// Construct an `SFC64` generator seeded from OS entropy.
    pub fn new() -> Self {
        Self::from_seed(SeedInput::None).expect("OS-seeded SFC64 cannot fail")
    }

    /// Construct an `SFC64` generator from any [`SeedInput`].
    pub fn from_seed(seed: impl Into<SeedInput>) -> Result<Self> {
        let seed_seq = SeedSequence::from_seed(seed)?;
        Self::from_seed_seq(seed_seq)
    }

    /// Construct an `SFC64` generator directly from its 4x64-bit internal state.
    pub fn from_raw_state(state: [u64; 4], has_uint32: bool, uinteger: u32) -> Self {
        Self {
            s: state,
            has_uint32,
            uinteger,
            seed_seq: None,
        }
    }

    /// Return a copy of the internal state.
    pub fn state(&self) -> Sfc64State {
        Sfc64State {
            state: self.s,
            has_uint32: self.has_uint32,
            uinteger: self.uinteger,
        }
    }

    /// Restore the internal state.
    pub fn set_state(&mut self, state: Sfc64State) {
        self.s = state.state;
        self.has_uint32 = state.has_uint32;
        self.uinteger = state.uinteger;
    }

    #[inline]
    fn step_next(&mut self) -> u64 {
        let tmp = self.s[0].wrapping_add(self.s[1]).wrapping_add(self.s[3]);
        self.s[3] = self.s[3].wrapping_add(1);
        self.s[0] = self.s[1] ^ (self.s[1] >> 11);
        self.s[1] = self.s[2].wrapping_add(self.s[2] << 3);
        self.s[2] = self.s[2].rotate_left(24).wrapping_add(tmp);
        tmp
    }
}

impl Default for SFC64 {
    fn default() -> Self {
        Self::new()
    }
}

impl BitGenerator for SFC64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.step_next()
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            self.uinteger
        } else {
            let next = self.step_next();
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
        let val = seed_seq.generate_state_u64(3);
        let mut rng = Self {
            s: [val[0], val[1], val[2], 1],
            has_uint32: false,
            uinteger: 0,
            seed_seq: Some(seed_seq),
        };
        for _ in 0..12 {
            rng.step_next();
        }
        Ok(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfc64_deadbeaf_testset() {
        let mut bg = SFC64::from_seed(0xdeadbeafu64).unwrap();
        let vals = bg.random_raw(4);
        assert_eq!(
            vals,
            vec![
                0xa475_f55f_bb6b_c638,
                0xb2d5_94b6_c29d_971c,
                0x275b_c4ec_e448_4fb1,
                0x569b_e72d_9b34_92fb,
            ]
        );
    }
}
