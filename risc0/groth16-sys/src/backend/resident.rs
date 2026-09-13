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

//! A process-wide cache of one prepared zkey per arm (DEF-G16-014): the
//! canonical path re-maps and re-uploads the zkey on every call; the arms
//! parse and upload it once and key the result by the file's identity.
//! `RISC0_GROTH16_RESIDENT=0` turns it off (every call uploads and frees).

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::SystemTime,
};

use anyhow::{Context as _, Result};

/// Environment variable: `0` disables the cache.
pub const RESIDENT_ENV: &str = "RISC0_GROTH16_RESIDENT";

/// Whether the cache is enabled (default: yes).
pub fn enabled() -> bool {
    !matches!(
        std::env::var(RESIDENT_ENV).as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// What `RISC0_GROTH16_RESIDENT` asks for. Unset means AUTO: the backend
/// chooses resident vs streaming from the memory budget against the device's
/// free memory (`budget::Budget::choose`). `0`/`false`/`off` forces streaming;
/// anything else forces resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// Decide from the budget and the device's free memory.
    Auto,
    /// The operator pinned the path (`true` = resident, `false` = streaming).
    Force(bool),
}

/// Read the tristate choice from the environment.
pub fn choice() -> Choice {
    match std::env::var(RESIDENT_ENV) {
        Err(_) => Choice::Auto,
        Ok(v) => Choice::Force(!matches!(v.as_str(), "0" | "false" | "off")),
    }
}

/// A zkey file's identity: path, length and modification time — enough to
/// notice a replaced file at the same path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key {
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
}

impl Key {
    /// Read the file's identity.
    pub fn of(path: &Path) -> Result<Self> {
        let meta = std::fs::metadata(path)
            .with_context(|| format!("zkey {}: metadata", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }
}

/// One cached value, keyed; `get_or_prepare` returns the cached value for
/// the same key and rebuilds it for a different one.
pub struct Cache<T> {
    slot: Mutex<Option<(Key, Arc<T>)>>,
}

impl<T> Cache<T> {
    /// An empty cache (a `static`).
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// The value for `key`, building it with `prepare` when absent or stale.
    /// The lock is held while building (one upload at a time) and released
    /// before the caller proves, so proofs never serialise on it.
    /// The cached value for `key`, if it is the one currently held — a
    /// non-preparing peek, so an AUTO caller can skip its probe once a resident
    /// zkey is already up.
    pub fn peek(&self, key: &Key) -> Option<Arc<T>> {
        let slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        slot.as_ref()
            .and_then(|(k, v)| (k == key).then(|| v.clone()))
    }

    pub fn get_or_prepare(&self, key: Key, prepare: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((k, v)) = slot.as_ref() {
            if *k == key {
                return Ok(v.clone());
            }
        }
        *slot = None; // free the stale zkey before the new one goes up
        let v = Arc::new(prepare()?);
        *slot = Some((key, v.clone()));
        Ok(v)
    }
}

impl<T> Default for Cache<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-step wall-clock on stderr when `RISC0_GROTH16_TIMING` is set: the
/// backend's own steps (read, parse, group, upload) around the prover's
/// phases, so a profile covers the whole call behind the boundary.
pub struct Stamp {
    enabled: bool,
    start: std::time::Instant,
    last: std::time::Instant,
}

impl Stamp {
    /// Start the clock.
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            enabled: std::env::var_os("RISC0_GROTH16_TIMING").is_some(),
            start: now,
            last: now,
        }
    }

    /// Print `what` with the time since the previous stamp and since the start.
    pub fn mark(&mut self, what: &str) {
        if self.enabled {
            let now = std::time::Instant::now();
            eprintln!(
                "[groth16-backend] {what:<28} {:>8.3} s  (t = {:.3} s)",
                (now - self.last).as_secs_f64(),
                (now - self.start).as_secs_f64()
            );
            self.last = now;
        }
    }
}

impl Default for Stamp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_reuses_and_a_changed_file_rebuilds() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("k.zkey");
        std::fs::write(&f, b"one").unwrap();
        let cache: Cache<String> = Cache::new();
        let mut builds = 0;
        let mut build = |s: &str| {
            builds += 1;
            Ok(s.to_string())
        };
        let a = cache
            .get_or_prepare(Key::of(&f).unwrap(), || build("a"))
            .unwrap();
        let b = cache
            .get_or_prepare(Key::of(&f).unwrap(), || build("b"))
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        std::fs::write(&f, b"three").unwrap(); // length changes → new key
        let c = cache
            .get_or_prepare(Key::of(&f).unwrap(), || build("c"))
            .unwrap();
        assert_eq!(*c, "c");
        assert_eq!(builds, 2);
    }

    #[test]
    fn env_switch() {
        // not set → enabled (the test does not touch the process environment)
        assert!(enabled() || std::env::var(RESIDENT_ENV).is_ok());
    }
}
