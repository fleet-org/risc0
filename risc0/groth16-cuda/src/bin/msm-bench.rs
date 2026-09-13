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

//! `groth16-cuda-msm-bench`: the bucket-sum kernel's throughput under the
//! production layout and under finer partitions of the same work — the
//! experiment behind W-14. Synthetic data: `--points` distinct multiples of
//! the G1 generator tiled to `--n` points, `--n` uniform full-width scalars,
//! digits and the per-window counting sort exactly as the prover does them.
//!
//! `--chunk C` splits every bucket's run into pieces of at most `C` points
//! (the kernel's contract is only "`starts[b]..starts[b+1]` is my run", so
//! the same launch computes partial sums); `--tiny` indexes only the distinct
//! points, so the gather stays in cache. `--check` compares the production
//! layout's first ranges with the Rust body.

use std::time::Instant;

use anyhow::{bail, Result};
use risc0_groth16_core::{
    ec::{g1_generator, Affine, Jacobian},
    field::Fp,
    scalar::digit,
};
use risc0_groth16_cuda::{CudaProver, ModuleSource};
use risc0_groth16_oxide::{
    kernels,
    pipeline::{sort_all_windows, WINDOW_BITS},
};

struct Opts {
    points: usize,
    n: usize,
    chunk: usize,
    tiny: bool,
    check: bool,
    repeats: usize,
}

fn parse() -> Result<Opts> {
    let mut o = Opts {
        points: 1 << 16,
        n: 1 << 20,
        chunk: 0,
        tiny: false,
        check: false,
        repeats: 3,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |what: &str| -> Result<usize> {
            match args.next() {
                Some(v) => Ok(v.parse()?),
                None => bail!("{what} needs a value"),
            }
        };
        match a.as_str() {
            "--points" => o.points = val("--points")?,
            "--n" => o.n = val("--n")?,
            "--chunk" => o.chunk = val("--chunk")?,
            "--repeats" => o.repeats = val("--repeats")?,
            "--tiny" => o.tiny = true,
            "--check" => o.check = true,
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(o)
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// `starts` with every range cut into pieces of at most `chunk` entries.
fn chunked(starts: &[u32], chunk: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(starts.len());
    for w in starts.windows(2) {
        let (from, to) = (w[0], w[1]);
        let mut at = from;
        loop {
            out.push(at);
            if to.saturating_sub(at) <= chunk {
                break;
            }
            at += chunk;
        }
    }
    out.push(*starts.last().expect("starts has a terminator"));
    out
}

fn main() -> Result<()> {
    let o = parse()?;
    let prover = CudaProver::new(ModuleSource::from_env()?)?;
    println!(
        "device: {} · module: {} · free {}",
        prover.device_name(),
        prover.module_source(),
        prover.device_free()
    );

    let t = Instant::now();
    let g = g1_generator();
    let mut acc = Jacobian::<Fp>::INFINITY;
    let distinct: Vec<Affine<Fp>> = (0..o.points)
        .map(|_| {
            acc = acc.add_affine(&g);
            acc.to_affine()
        })
        .collect();
    let points: Vec<Affine<Fp>> = if o.tiny {
        distinct.clone()
    } else {
        (0..o.n).map(|i| distinct[i % o.points]).collect()
    };
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let scalars: Vec<[u64; 4]> = (0..o.n)
        .map(|_| {
            [
                xorshift(&mut seed),
                xorshift(&mut seed),
                xorshift(&mut seed),
                xorshift(&mut seed) >> 3,
            ]
        })
        .collect();
    let w = WINDOW_BITS;
    let windows = 256u32.div_ceil(w) as usize;
    let buckets = (1usize << w) - 1;
    let mut digits = vec![0u32; windows * o.n];
    for (win, chunk) in digits.chunks_exact_mut(o.n).enumerate() {
        for (i, d) in chunk.iter_mut().enumerate() {
            *d = digit(&scalars[i], win as u32, w) as u32;
        }
    }
    let (mut order, starts) = sort_all_windows(&digits, o.n, buckets);
    if o.tiny {
        for x in &mut order {
            *x %= o.points as u32;
        }
    }
    let starts = if o.chunk > 0 {
        chunked(&starts, o.chunk as u32)
    } else {
        starts
    };
    let ranges = starts.len() - 1;
    println!(
        "data: {} distinct points, n = {}, {} windows, {} adds, {} ranges ({:.1} per range), host {:.1} s",
        o.points,
        o.n,
        windows,
        order.len(),
        ranges,
        order.len() as f64 / ranges as f64,
        t.elapsed().as_secs_f64()
    );

    let (secs, sums) = prover.bench_bucket_sum_g1(&points, &order, &starts, o.repeats)?;
    for (i, s) in secs.iter().enumerate() {
        println!(
            "launch {i}: {s:8.3} s  {:8.2} M adds/s  (free {})",
            order.len() as f64 / s / 1e6,
            prover.device_free()
        );
    }
    let calib = prover.bench_pointwise_mul(1 << 23, o.repeats)?;
    let best = calib.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "pointwise_mul over 2^23: best {:.4} s = {:.0} GB/s of traffic (768 MB per launch)",
        best,
        0.768 / best
    );

    if o.check {
        let bad = (0..ranges.min(4096))
            .filter(|&b| sums[b] != kernels::bucket_sum(b, &points, &order, &starts))
            .count();
        if bad > 0 {
            bail!(
                "{bad} of the first {} ranges differ from the Rust body",
                ranges.min(4096)
            );
        }
        println!(
            "check: the first {} ranges agree with the Rust body",
            ranges.min(4096)
        );
    }
    Ok(())
}
