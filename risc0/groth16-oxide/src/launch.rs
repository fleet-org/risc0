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

//! The launch seam. Every kernel in this crate is a *map*: output index `i`
//! is a pure function of the inputs and `i`, so a launcher only has to run
//! the body over an index space and hand each thread its own output slot —
//! the shape of `#[kernel]` + `DisjointSlice` in cuda-oxide, and the shape
//! a CPU can emulate with plain chunking.

/// Runs a per-index body over an output buffer.
pub trait Launcher {
    /// `out[i] = f(i)` for every `i < out.len()`.
    fn map<T, F>(&self, out: &mut [T], f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(usize) -> T + Sync;
}

/// A host launcher: the same bodies, chunked across OS threads. It proves
/// the pipeline on a CPU; it is not a registered backend.
#[derive(Clone, Copy, Debug)]
pub struct CpuLauncher {
    /// Threads to spread the index space over.
    pub threads: usize,
}

impl Default for CpuLauncher {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self { threads }
    }
}

impl Launcher for CpuLauncher {
    fn map<T, F>(&self, out: &mut [T], f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(usize) -> T + Sync,
    {
        let n = out.len();
        if n == 0 {
            return;
        }
        let threads = self.threads.clamp(1, n);
        let chunk = n.div_ceil(threads);
        let f = &f;
        std::thread::scope(|scope| {
            for (t, slice) in out.chunks_mut(chunk).enumerate() {
                let base = t * chunk;
                scope.spawn(move || {
                    for (j, slot) in slice.iter_mut().enumerate() {
                        *slot = f(base + j);
                    }
                });
            }
        });
    }
}

/// A single-threaded launcher, for deterministic small tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct SerialLauncher;

impl Launcher for SerialLauncher {
    fn map<T, F>(&self, out: &mut [T], f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(usize) -> T + Sync,
    {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = f(i);
        }
    }
}
