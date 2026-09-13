// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The device memory budget of a Groth16 proof, as a closed-form function of
//! the circuit's dimensions — the written budget the bbstark harvest
//! (`prover-expert` DEF-PRV-007) recommends, so resident-vs-streaming is a
//! COMPUTED decision against the device's free memory, not a hand-set flag.
//!
//! Every figure is derived from the same element sizes the ABI pins
//! ([`crate::abi`]) and the same buffers [`CudaProver::prepare`] uploads
//! (mirrored from `ResidentZkey::device_bytes`), so the numbers here track the
//! code. Residency follows the three-tier taxonomy: **persistent** (uploaded
//! once, lives the whole proof — the resident zkey), **ephemeral** (one
//! phase's working buffers), and the per-proof **scratch** peak on top.

use core::mem::size_of;

use risc0_groth16_core::{
    ec::{Affine, Jacobian},
    field::{Fp, Fr},
    fp2::Fp2,
    zkey::Zkey,
};

use crate::pipeline::WINDOW_BITS;

/// One element's device footprint, in bytes, taken from the concrete types so
/// the budget can never drift from the ABI records.
const FR: usize = size_of::<Fr>(); // 32
const G1: usize = size_of::<Affine<Fp>>(); // 72
const G2: usize = size_of::<Affine<Fp2>>(); // 136
const JAC_G1: usize = size_of::<Jacobian<Fp>>(); // 96
const JAC_G2: usize = size_of::<Jacobian<Fp2>>(); // 192
const COEFF: usize = size_of::<risc0_groth16_core::coeff::GroupedCoeff>(); // 40
const U32: usize = size_of::<u32>(); // 4

/// The residency class of a buffer (the bbstark taxonomy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Uploaded once, lives the whole proof (the resident zkey).
    Persistent,
    /// One phase's working buffers, freed before the next.
    Ephemeral,
    /// Kernel-local scratch within a phase.
    Scratch,
}

/// Which proof path fits the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The zkey uploaded once per process; every later proof moves only the
    /// witness in and the proof out. The fast path on a device that holds it.
    Resident,
    /// Every phase uploads what it reads and frees it before the next; the
    /// smallest footprint, for a shared or small device.
    Streaming,
}

/// The circuit dimensions the budget is a function of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    /// `num_vars` — the A/B/C witness length (G1 point-set size for a, b1, b2).
    pub num_vars: usize,
    /// Public inputs (the `c` MSM is over `num_vars - num_public - 1`).
    pub num_public: usize,
    /// The evaluation domain (a power of two; the `h` MSM and the transforms).
    pub domain: usize,
    /// The total grouped-coefficient count (the scatter input).
    pub coefficients: usize,
}

impl Dims {
    /// Read the dimensions off a parsed zkey (before any upload).
    pub fn from_zkey(zkey: &Zkey) -> Self {
        Self {
            num_vars: zkey.num_vars,
            num_public: zkey.num_public,
            domain: zkey.domain_size,
            coefficients: zkey.coefficients.len(),
        }
    }
}

/// The device memory budget of a proof over [`Dims`].
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    dims: Dims,
    windows: usize,
    buckets: usize,
}

impl Budget {
    /// The budget for these dimensions.
    pub fn new(dims: Dims) -> Self {
        let w = WINDOW_BITS as usize;
        Self {
            dims,
            windows: 256usize.div_ceil(w),
            buckets: (1usize << w) - 1,
        }
    }

    /// The NTT tables (twiddles forward + inverse over `domain/2`, shift powers
    /// over `domain`): `2 · domain · FR`. Persistent.
    pub fn tables_bytes(&self) -> usize {
        (self.dims.domain / 2 + self.dims.domain / 2 + self.dims.domain) * FR
    }

    /// The grouped coefficients and their per-group starts. Persistent.
    pub fn coefficients_bytes(&self) -> usize {
        // ca + cb hold every coefficient once; sa + sb are one u32 per group,
        // bounded above by the domain (one group per output index).
        self.dims.coefficients * COEFF + 2 * self.dims.domain * U32
    }

    /// The five point sets (a, b1, c on G1 over `num_vars`; h on G1 over
    /// `domain`; b2 on G2 over `num_vars`). Persistent.
    pub fn points_bytes(&self) -> usize {
        let c = self.dims.num_vars.saturating_sub(self.dims.num_public + 1);
        (self.dims.num_vars + self.dims.num_vars + c + self.dims.domain) * G1
            + self.dims.num_vars * G2
    }

    /// The persistent resident set: tables + coefficients + point sets. This is
    /// what `ResidentZkey::device_bytes` reports once uploaded.
    pub fn resident_bytes(&self) -> usize {
        self.tables_bytes() + self.coefficients_bytes() + self.points_bytes()
    }

    /// The transform phase's ephemeral working set: four polynomial buffers of
    /// `domain` field elements (the C19 four-buffer scheme; the tables are
    /// already resident or streamed alongside).
    pub fn transform_scratch_bytes(&self) -> usize {
        4 * self.dims.domain * FR
    }

    /// One MSM's scratch, excluding the point set: the canonical scalars, the
    /// window-major digits, the sorted `order`, the flat `starts`, and the flat
    /// bucket sums. `g2` picks the wider Jacobian for the sums.
    pub fn msm_scratch_bytes(&self, g2: bool) -> usize {
        let n = self.dims.num_vars.max(self.dims.domain); // the largest MSM (h over the domain)
        let jac = if g2 { JAC_G2 } else { JAC_G1 };
        n * FR                              // canonical scalars
            + self.windows * n * U32        // digits (window-major)
            + self.windows * n * U32        // order (<= digits)
            + self.windows * self.buckets * U32 // starts (flat over window,bucket)
            + self.windows * self.buckets * jac // level-0 bucket sums (upper bound)
    }

    /// The largest single point set, uploaded per MSM on the streaming path
    /// (h on G1 over the domain, or b2 on G2 over `num_vars`).
    pub fn largest_point_set_bytes(&self) -> usize {
        (self.dims.domain * G1).max(self.dims.num_vars * G2)
    }

    /// Peak device bytes on the RESIDENT path: the persistent set plus the
    /// largest per-proof phase scratch (the point sets are already resident).
    pub fn resident_peak_bytes(&self) -> usize {
        let scratch = self
            .transform_scratch_bytes()
            .max(self.msm_scratch_bytes(true))
            + self.dims.num_vars * FR; // the witness
        self.resident_bytes() + scratch
    }

    /// Peak device bytes on the STREAMING path: nothing persistent; the peak is
    /// the largest phase's own uploads (the tables plus either the four-buffer
    /// transform, or one point set with its MSM scratch).
    pub fn streaming_peak_bytes(&self) -> usize {
        let transform = self.transform_scratch_bytes() + self.tables_bytes();
        let msm = self.largest_point_set_bytes() + self.msm_scratch_bytes(true);
        transform.max(msm)
    }

    /// Choose the path that fits `free_bytes` with `margin` (0..1) of it left
    /// as headroom for the allocator and a co-tenant: [`Mode::Resident`] if its
    /// peak fits, else [`Mode::Streaming`] if it fits, else `None` (the proof
    /// does not fit this device even streaming).
    pub fn choose(&self, free_bytes: usize, margin: f64) -> Option<Mode> {
        let usable = (free_bytes as f64 * margin) as usize;
        if self.resident_peak_bytes() <= usable {
            Some(Mode::Resident)
        } else if self.streaming_peak_bytes() <= usable {
            Some(Mode::Streaming)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The production circuit's measured dimensions (BOUNDARY §4).
    fn production() -> Budget {
        Budget::new(Dims {
            num_vars: 5_635_930,
            num_public: 5,
            domain: 1 << 23,
            coefficients: 29_098_147,
        })
    }

    const GIB: f64 = (1usize << 30) as f64;

    #[test]
    fn resident_set_matches_the_measured_footprint() {
        // BOUNDARY §7: resident set ≈ 4.6 GB (point sets ≈ 2.6, coeffs ≈ 1.2,
        // tables ≈ 0.8). The closed form must land in that neighbourhood.
        let b = production();
        let gib = b.resident_bytes() as f64 / GIB;
        assert!(
            (3.8..5.2).contains(&gib),
            "resident set {gib:.2} GiB out of range"
        );
        let points = b.points_bytes() as f64 / GIB;
        assert!((2.2..3.0).contains(&points), "point sets {points:.2} GiB");
    }

    #[test]
    fn streaming_peak_is_about_two_gib() {
        // The measured streaming footprint fits beside a tenant's peak (≈ 2 GB,
        // C18/C19).
        let gib = production().streaming_peak_bytes() as f64 / GIB;
        assert!(
            (1.3..2.8).contains(&gib),
            "streaming peak {gib:.2} GiB out of range"
        );
    }

    #[test]
    fn resident_peak_exceeds_streaming_peak() {
        let b = production();
        assert!(b.resident_peak_bytes() > b.streaming_peak_bytes());
    }

    #[test]
    fn choose_picks_resident_on_a_big_card_and_streaming_on_a_shared_one() {
        let b = production();
        // A 24 GB RTX 4090, mostly free: resident.
        assert_eq!(b.choose(24 * (1 << 30), 0.9), Some(Mode::Resident));
        // Only 3 GiB free (a co-tenant holds the rest): streaming.
        assert_eq!(b.choose(3 * (1 << 30), 0.9), Some(Mode::Streaming));
        // 1 GiB free: nothing fits.
        assert_eq!(b.choose(1 << 30, 0.9), None);
    }

    #[test]
    fn element_sizes_are_the_abi_records() {
        assert_eq!(
            (FR, G1, G2, JAC_G1, JAC_G2, COEFF),
            (32, 72, 136, 96, 192, 40)
        );
    }
}
