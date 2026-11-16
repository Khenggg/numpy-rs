use crate::mt19937::MT19937;
use crate::pcg64::{PCG64DXSM, PCG64};
use crate::philox::Philox;
use crate::seed_sequence::SeedSequence;
use crate::sfc64::SFC64;
use numpy_types::{NumPyError, Result};

/// Reciprocal of 2^53 (`9007199254740992.0`) used to map top 53 bits to `[0.0, 1.0)`.
pub(crate) const INV_2_POW_53: f64 = 1.0 / 9007199254740992.0;

/// Reciprocal of 2^24 (`16777216.0`) used by NumPy `distributions.c` for `next_float`.
pub(crate) const INV_2_POW_24_F32: f32 = 1.0 / 16777216.0;

/// Core interface implemented by all NumPy-compatible BitGenerators.
pub trait BitGenerator: Clone + Send {
    /// Generate the next uniform unsigned 32-bit integer.
    fn next_u32(&mut self) -> u32;

    /// Generate the next uniform unsigned 64-bit integer.
    fn next_u64(&mut self) -> u64;

    /// Generate the next double-precision float in `[0.0, 1.0)`.
    ///
    /// Default matches NumPy's `uint64_to_double(next_u64()) = (rnd >> 11) * 2^-53`.
    /// Note that `MT19937` overrides this to compose from two 32-bit draws.
    #[inline]
    fn next_double(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * INV_2_POW_53
    }

    /// Generate the next single-precision float in `[0.0, 1.0)`.
    ///
    /// Matches NumPy `distributions.c`: `(next_u32() >> 8) * (1.0f / 16777216.0f)`.
    #[inline]
    fn next_float(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * INV_2_POW_24_F32
    }

    /// Generate one native raw word (`u32 as u64` for 32-bit generators like MT19937,
    /// `next_u64()` for 64-bit generators).
    #[inline]
    fn next_raw(&mut self) -> u64 {
        self.next_u64()
    }

    /// Fill a byte buffer with uniform random bytes (little-endian word order).
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let tail = self.next_u64().to_le_bytes();
            rem.copy_from_slice(&tail[..rem.len()]);
        }
    }

    /// Return `size` raw outputs from this BitGenerator.
    fn random_raw(&mut self, size: usize) -> Vec<u64> {
        let mut out = Vec::with_capacity(size);
        for _ in 0..size {
            out.push(self.next_raw());
        }
        out
    }

    /// Reference to the underlying [`SeedSequence`], if constructed from one.
    fn seed_seq(&self) -> Option<&SeedSequence> {
        None
    }

    /// Mutable reference to the underlying [`SeedSequence`], if present.
    fn seed_seq_mut(&mut self) -> Option<&mut SeedSequence> {
        None
    }

    /// Construct an instance from an initialized [`SeedSequence`].
    fn from_seed_seq(seed_seq: SeedSequence) -> Result<Self>
    where
        Self: Sized;

    /// Spawn `n_children` independent BitGenerators from this BitGenerator's `SeedSequence`.
    fn spawn(&mut self, n_children: usize) -> Result<Vec<Self>>
    where
        Self: Sized,
    {
        let child_seqs = self
            .seed_seq_mut()
            .ok_or_else(|| {
                NumPyError::InvalidValue(
                    "Cannot spawn: BitGenerator has no attached SeedSequence".into(),
                )
            })?
            .spawn(n_children)?;
        let mut out = Vec::with_capacity(n_children);
        for seq in child_seqs {
            out.push(Self::from_seed_seq(seq)?);
        }
        Ok(out)
    }

    /// Spawn `n_children` child BitGenerators for parallel generation, falling back to
    /// seeding a fresh `SeedSequence` from random draws if no `SeedSequence` is attached.
    fn spawn_or_fork(&mut self, n_children: usize) -> Result<Vec<Self>>
    where
        Self: Sized,
    {
        if self.seed_seq().is_some() {
            self.spawn(n_children)
        } else {
            let s0 = self.next_u64();
            let s1 = self.next_u64();
            let mut seq = SeedSequence::from_seed([s0, s1])?;
            let child_seqs = seq.spawn(n_children)?;
            let mut out = Vec::with_capacity(n_children);
            for cseq in child_seqs {
                out.push(Self::from_seed_seq(cseq)?);
            }
            Ok(out)
        }
    }
}

/// Runtime-polymorphic wrapper over all 5 NumPy BitGenerators.
#[derive(Debug, Clone)]
pub enum AnyBitGenerator {
    PCG64(PCG64),
    PCG64DXSM(PCG64DXSM),
    MT19937(MT19937),
    Philox(Philox),
    SFC64(SFC64),
}

impl BitGenerator for AnyBitGenerator {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        match self {
            Self::PCG64(b) => b.next_u32(),
            Self::PCG64DXSM(b) => b.next_u32(),
            Self::MT19937(b) => b.next_u32(),
            Self::Philox(b) => b.next_u32(),
            Self::SFC64(b) => b.next_u32(),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        match self {
            Self::PCG64(b) => b.next_u64(),
            Self::PCG64DXSM(b) => b.next_u64(),
            Self::MT19937(b) => b.next_u64(),
            Self::Philox(b) => b.next_u64(),
            Self::SFC64(b) => b.next_u64(),
        }
    }

    #[inline]
    fn next_double(&mut self) -> f64 {
        match self {
            Self::PCG64(b) => b.next_double(),
            Self::PCG64DXSM(b) => b.next_double(),
            Self::MT19937(b) => b.next_double(),
            Self::Philox(b) => b.next_double(),
            Self::SFC64(b) => b.next_double(),
        }
    }

    #[inline]
    fn next_float(&mut self) -> f32 {
        match self {
            Self::PCG64(b) => b.next_float(),
            Self::PCG64DXSM(b) => b.next_float(),
            Self::MT19937(b) => b.next_float(),
            Self::Philox(b) => b.next_float(),
            Self::SFC64(b) => b.next_float(),
        }
    }

    #[inline]
    fn next_raw(&mut self) -> u64 {
        match self {
            Self::PCG64(b) => b.next_raw(),
            Self::PCG64DXSM(b) => b.next_raw(),
            Self::MT19937(b) => b.next_raw(),
            Self::Philox(b) => b.next_raw(),
            Self::SFC64(b) => b.next_raw(),
        }
    }

    fn seed_seq(&self) -> Option<&SeedSequence> {
        match self {
            Self::PCG64(b) => b.seed_seq(),
            Self::PCG64DXSM(b) => b.seed_seq(),
            Self::MT19937(b) => b.seed_seq(),
            Self::Philox(b) => b.seed_seq(),
            Self::SFC64(b) => b.seed_seq(),
        }
    }

    fn seed_seq_mut(&mut self) -> Option<&mut SeedSequence> {
        match self {
            Self::PCG64(b) => b.seed_seq_mut(),
            Self::PCG64DXSM(b) => b.seed_seq_mut(),
            Self::MT19937(b) => b.seed_seq_mut(),
            Self::Philox(b) => b.seed_seq_mut(),
            Self::SFC64(b) => b.seed_seq_mut(),
        }
    }

    fn from_seed_seq(seed_seq: SeedSequence) -> Result<Self> {
        Ok(Self::PCG64(PCG64::from_seed_seq(seed_seq)?))
    }

    fn spawn(&mut self, n_children: usize) -> Result<Vec<Self>> {
        Ok(match self {
            Self::PCG64(b) => b.spawn(n_children)?.into_iter().map(Self::PCG64).collect(),
            Self::PCG64DXSM(b) => b
                .spawn(n_children)?
                .into_iter()
                .map(Self::PCG64DXSM)
                .collect(),
            Self::MT19937(b) => b.spawn(n_children)?.into_iter().map(Self::MT19937).collect(),
            Self::Philox(b) => b.spawn(n_children)?.into_iter().map(Self::Philox).collect(),
            Self::SFC64(b) => b.spawn(n_children)?.into_iter().map(Self::SFC64).collect(),
        })
    }
}
