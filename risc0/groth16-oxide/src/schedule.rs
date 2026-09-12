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

//! The launch schedule of the coset transform (`h_to_coset`: inverse NTT,
//! scale by `n⁻¹·shift^i`, forward NTT) as DATA: which kernel, reading which
//! of two ping-pong buffers, writing which, with which arguments — and which
//! buffer holds the result at the end.
//!
//! Three executors run the same schedule: the CPU pipeline with slices, the
//! Metal prover with `MTLBuffer`s, the CUDA prover with device buffers. The
//! ping-pong bookkeeping therefore exists once, here, and is covered by the
//! CPU pipeline's byte-identical-proof tests; an executor cannot read the
//! stale buffer, because it never chooses a buffer — the schedule does.
//! (The first Metal draft chose its own and read the pre-scale buffer into
//! the forward NTT — the bug this module removes.)

extern crate alloc;

/// One of the two ping-pong buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buf {
    /// The buffer that holds the input on entry.
    A,
    /// The scratch buffer.
    B,
}

impl Buf {
    /// The other buffer.
    pub const fn other(self) -> Self {
        match self {
            Buf::A => Buf::B,
            Buf::B => Buf::A,
        }
    }
}

/// Which twiddle table a stage reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Twiddles {
    /// `Tables::inverse` (the inverse NTT).
    Inverse,
    /// `Tables::forward` (the forward NTT).
    Forward,
}

/// One launch of the coset transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// `dst[i] = src[rev(i)]` (`kernels::bit_reverse`).
    BitReverse {
        /// Read.
        src: Buf,
        /// Written.
        dst: Buf,
    },
    /// One radix-2 stage (`kernels::ntt_stage`) of length `len` with twiddle
    /// `stride`.
    NttStage {
        /// Read.
        src: Buf,
        /// Written.
        dst: Buf,
        /// Stage length (a power of two, `2..=n`).
        len: u32,
        /// Twiddle stride: `n / len`.
        stride: u32,
        /// Table.
        twiddles: Twiddles,
    },
    /// `dst[i] = src[i]·shift_powers[i]·n_inv` (`kernels::pointwise_scale`
    /// then `kernels::scale`).
    Scale {
        /// Read.
        src: Buf,
        /// Written.
        dst: Buf,
    },
}

impl Step {
    /// The buffer this step reads.
    pub const fn src(&self) -> Buf {
        match *self {
            Step::BitReverse { src, .. } | Step::NttStage { src, .. } | Step::Scale { src, .. } => {
                src
            }
        }
    }

    /// The buffer this step writes.
    pub const fn dst(&self) -> Buf {
        match *self {
            Step::BitReverse { dst, .. } | Step::NttStage { dst, .. } | Step::Scale { dst, .. } => {
                dst
            }
        }
    }
}

/// The schedule for a domain of `n = 2^lg_n` points, and the buffer holding
/// the result once every step has run in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Launches, in order.
    pub steps: alloc::vec::Vec<Step>,
    /// Where the coset evaluations are at the end.
    pub result: Buf,
}

/// Bit-reverse `src` into `dst`, then `lg_n` stages ping-ponging from `dst`;
/// returns the buffer holding the transform.
fn ntt(steps: &mut alloc::vec::Vec<Step>, input: Buf, n: u32, twiddles: Twiddles) -> Buf {
    let mut src = input;
    let mut dst = input.other();
    steps.push(Step::BitReverse { src, dst });
    let mut len = 2u32;
    while len <= n {
        (src, dst) = (dst, src);
        steps.push(Step::NttStage {
            src,
            dst,
            len,
            stride: n / len,
            twiddles,
        });
        len <<= 1;
    }
    dst
}

/// The coset transform's schedule for `n` points (`n` a power of two, `>= 2`),
/// input in [`Buf::A`].
pub fn coset(n: u32) -> Schedule {
    assert!(
        n >= 2 && n.is_power_of_two(),
        "domain must be a power of two"
    );
    let mut steps = alloc::vec::Vec::with_capacity(2 * (n.trailing_zeros() as usize + 1) + 1);
    let evals = ntt(&mut steps, Buf::A, n, Twiddles::Inverse);
    steps.push(Step::Scale {
        src: evals,
        dst: evals.other(),
    });
    let result = ntt(&mut steps, evals.other(), n, Twiddles::Forward);
    Schedule { steps, result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_step_reads_what_the_previous_wrote() {
        for lg in 1..8 {
            let n = 1u32 << lg;
            let s = coset(n);
            assert_eq!(s.steps.len(), 2 * (lg as usize + 1) + 1);
            let mut live = Buf::A; // the buffer holding the current data
            for step in &s.steps {
                let (src, dst) = (step.src(), step.dst());
                assert_eq!(src, live, "{step:?} reads a stale buffer");
                assert_ne!(src, dst, "{step:?} reads and writes one buffer");
                live = dst;
            }
            assert_eq!(s.result, live);
            // stages: lengths 2, 4, …, n with stride n/len, first inverse then forward
            let stages: alloc::vec::Vec<_> = s
                .steps
                .iter()
                .filter_map(|st| match *st {
                    Step::NttStage {
                        len,
                        stride,
                        twiddles,
                        ..
                    } => Some((len, stride, twiddles)),
                    _ => None,
                })
                .collect();
            assert_eq!(stages.len(), 2 * lg as usize);
            for (k, (len, stride, tw)) in stages.iter().enumerate() {
                let j = k % lg as usize;
                assert_eq!(*len, 2u32 << j);
                assert_eq!(*stride, n / len);
                assert_eq!(
                    *tw,
                    if k < lg as usize {
                        Twiddles::Inverse
                    } else {
                        Twiddles::Forward
                    }
                );
            }
        }
    }
}
