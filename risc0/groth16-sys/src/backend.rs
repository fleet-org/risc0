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

//! The Groth16 prover boundary (GROTH16 s01/2, decision DEF-G16-002).
//!
//! [`crate::prove`] is the single point at which the GPU prover is substituted.
//! Every caller — `risc0-groth16`'s `prove/cuda.rs` for the `stark_verify`
//! circuit and boundless's `blake3_groth16` for its own circuit — reaches it
//! with the same two parameter structs, so one selection covers both circuits.
//!
//! Three implementations can stand behind it: `canonical` (the upstream CUDA
//! C++ kernels behind the `risc0_groth16_cuda_prove` C ABI), `cuda-oxide`
//! (s01/4) and `metal` (s01/4b). The rules:
//!
//! * **Availability is decided by the build.** A backend is registered in
//!   [`compiled_in`] only under the `cfg` that compiles its implementation.
//! * **Selection is decided by configuration.** `RISC0_GROTH16_BACKEND`
//!   names the backend; when the variable is absent the default is
//!   `canonical`, the proven path.
//! * **An unknown name is an error, not a default; a known but unavailable
//!   backend is an error, not a fallback.** The outcomes *selected*,
//!   *unknown* and *unavailable* stay distinct, so a differential harness can
//!   never mistake "the canonical path answered" for "the rewrite answered".

use std::fmt;

use crate::{ProverParams, SetupParams};

/// The environment variable that selects the backend.
pub const BACKEND_ENV: &str = "RISC0_GROTH16_BACKEND";

/// One implementation of the Groth16 prover behind the boundary.
pub trait Groth16Backend {
    /// Which implementation this is.
    fn kind(&self) -> BackendKind;

    /// Run the prover: load the artifacts named by `setup`, consume the witness
    /// in `prover`, and write `proof.json` and `public.json` to the paths in
    /// `prover`. The data contract is documented in `groth16_s01/BOUNDARY.md`.
    fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()>;
}

impl fmt::Debug for dyn Groth16Backend + '_ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Groth16Backend({})", self.kind())
    }
}

/// The implementations the boundary admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// The upstream CUDA C++ kernels via `risc0_groth16_cuda_prove`.
    Canonical,
    /// The cuda-oxide (Rust → PTX) rewrite (GROTH16 s01/4).
    CudaOxide,
    /// The Metal (MSL) rewrite for Apple Silicon (GROTH16 s01/4b).
    Metal,
    /// The CPU reference prover (`risc0-groth16-core`): the arms' shared
    /// arithmetic run on the host. For tests and the differential harness —
    /// never the default, not a production path.
    Reference,
    /// The cuda-oxide arm's kernel bodies (`risc0-groth16-oxide`) run by the
    /// host launcher: the exact kernel code, without a GPU. A testing kind
    /// like `reference` — never the default, not a production path.
    OxideCpu,
    /// The Metal arm's shaders compiled as C++ and run on the CPU (every target
    /// but macOS): the Metal counterpart of `OxideCpu`, a testing kind.
    MetalCpu,
}

impl BackendKind {
    /// Every kind, in a stable order.
    pub const ALL: [BackendKind; 6] = [
        BackendKind::Canonical,
        BackendKind::CudaOxide,
        BackendKind::Metal,
        BackendKind::Reference,
        BackendKind::OxideCpu,
        BackendKind::MetalCpu,
    ];

    /// The kind used when [`BACKEND_ENV`] is absent.
    pub const DEFAULT: BackendKind = BackendKind::Canonical;

    /// The configured name of this kind.
    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Canonical => "canonical",
            BackendKind::CudaOxide => "cuda-oxide",
            BackendKind::Metal => "metal",
            BackendKind::Reference => "reference",
            BackendKind::OxideCpu => "oxide-cpu",
            BackendKind::MetalCpu => "metal-cpu",
        }
    }

    /// Parse a configured name. Surrounding whitespace is ignored and so is
    /// ASCII case; anything else — including the empty string — is
    /// [`SelectError::Unknown`], never a default.
    pub fn parse(value: &str) -> Result<Self, SelectError> {
        let wanted = value.trim().to_ascii_lowercase();
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.name() == wanted)
            .ok_or_else(|| SelectError::Unknown {
                value: value.to_string(),
            })
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Why a backend could not be selected. Each variant is a different fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectError {
    /// The configured value names no backend this crate knows.
    Unknown {
        /// The value as configured.
        value: String,
    },
    /// The configured value names a backend this build does not contain.
    Unavailable {
        /// What was asked for.
        kind: BackendKind,
        /// What this build contains.
        available: Vec<BackendKind>,
    },
    /// [`BACKEND_ENV`] is present but is not valid Unicode.
    NotUnicode,
}

impl fmt::Display for SelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectError::Unknown { value } => write!(
                f,
                "{BACKEND_ENV}={value:?} names no Groth16 backend (known: {})",
                BackendKind::ALL.map(BackendKind::name).join(", ")
            ),
            SelectError::Unavailable { kind, available } => {
                let list = if available.is_empty() {
                    "none".to_string()
                } else {
                    available
                        .iter()
                        .map(|k| k.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                write!(
                    f,
                    "Groth16 backend `{kind}` is not compiled into this build (available: {list})"
                )
            }
            SelectError::NotUnicode => write!(f, "{BACKEND_ENV} is not valid Unicode"),
        }
    }
}

impl std::error::Error for SelectError {}

/// The backends compiled into a build, keyed by kind.
#[derive(Default)]
pub struct Registry {
    backends: Vec<Box<dyn Groth16Backend>>,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a backend. Registering a kind twice is a construction error.
    pub fn with(mut self, backend: Box<dyn Groth16Backend>) -> Self {
        let kind = backend.kind();
        assert!(
            self.backends.iter().all(|b| b.kind() != kind),
            "Groth16 backend `{kind}` registered twice"
        );
        self.backends.push(backend);
        self
    }

    /// The kinds this registry contains, in registration order.
    pub fn available(&self) -> Vec<BackendKind> {
        self.backends.iter().map(|b| b.kind()).collect()
    }

    /// The backend of a given kind, or [`SelectError::Unavailable`].
    pub fn get(&self, kind: BackendKind) -> Result<&dyn Groth16Backend, SelectError> {
        self.backends
            .iter()
            .find(|b| b.kind() == kind)
            .map(|b| b.as_ref())
            .ok_or_else(|| SelectError::Unavailable {
                kind,
                available: self.available(),
            })
    }

    /// Select by a configured value: `None` (the variable is absent) means
    /// [`BackendKind::DEFAULT`]; `Some` must parse and must be available.
    pub fn select(&self, configured: Option<&str>) -> Result<&dyn Groth16Backend, SelectError> {
        let kind = match configured {
            None => BackendKind::DEFAULT,
            Some(value) => BackendKind::parse(value)?,
        };
        self.get(kind)
    }

    /// Select from the process environment ([`BACKEND_ENV`]).
    pub fn select_from_env(&self) -> Result<&dyn Groth16Backend, SelectError> {
        match std::env::var_os(BACKEND_ENV) {
            None => self.select(None),
            Some(value) => {
                let value = value.to_str().ok_or(SelectError::NotUnicode)?;
                self.select(Some(value))
            }
        }
    }

    /// The whole boundary in one call: select by `configured`, then prove.
    /// A selection error or a backend error is returned as-is; nothing
    /// answers in place of the selected backend.
    pub fn prove(
        &self,
        configured: Option<&str>,
        prover: &ProverParams,
        setup: &SetupParams,
    ) -> anyhow::Result<()> {
        let selected = self.select(configured)?;
        selected.prove(prover, setup)
    }

    /// [`Registry::prove`] with the selection read from [`BACKEND_ENV`].
    pub fn prove_from_env(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
        let selected = self.select_from_env()?;
        selected.prove(prover, setup)
    }
}

/// The registry of this build: exactly the backends whose implementation is
/// compiled in. Nothing is registered under a feature it does not implement.
pub fn compiled_in() -> Registry {
    let registry = Registry::new();
    #[cfg(feature = "cuda")]
    let registry = registry.with(Box::new(canonical::Canonical));
    #[cfg(feature = "reference")]
    let registry = registry.with(Box::new(reference::Reference));
    #[cfg(feature = "oxide-cpu")]
    let registry = registry.with(Box::new(oxide_cpu::OxideCpu));
    #[cfg(feature = "cuda-oxide")]
    let registry = registry.with(Box::new(cuda_oxide::CudaOxide));
    #[cfg(all(
        feature = "metal-cpu",
        not(all(target_os = "macos", target_arch = "aarch64"))
    ))]
    let registry = registry.with(Box::new(metal_cpu::MetalCpu));
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    let registry = registry.with(Box::new(metal::Metal));
    registry
}

#[cfg(any(
    feature = "reference",
    feature = "oxide-cpu",
    feature = "cuda-oxide",
    feature = "metal",
    feature = "metal-cpu"
))]
pub mod reference;

#[cfg(feature = "oxide-cpu")]
pub mod oxide_cpu;

#[cfg(any(feature = "cuda-oxide", feature = "metal", feature = "metal-cpu"))]
pub mod resident;

#[cfg(feature = "cuda-oxide")]
pub mod cuda_oxide;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod metal;

#[cfg(all(
    feature = "metal-cpu",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
pub mod metal_cpu;

#[cfg(test)]
pub(crate) mod fixture;

#[cfg(feature = "cuda")]
mod canonical {
    use super::{BackendKind, Groth16Backend};
    use crate::{ProverParams, SetupParams};

    /// The upstream CUDA C++ prover behind the `risc0_groth16_cuda_prove` C ABI.
    pub struct Canonical;

    impl Groth16Backend for Canonical {
        fn kind(&self) -> BackendKind {
            BackendKind::Canonical
        }

        fn prove(&self, prover: &ProverParams, setup: &SetupParams) -> anyhow::Result<()> {
            crate::ffi_prove(prover, setup)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, path::Path, rc::Rc};

    use super::*;

    /// A backend that only records whether it ran.
    struct Mock {
        kind: BackendKind,
        calls: Rc<Cell<u32>>,
        fail: bool,
    }

    impl Groth16Backend for Mock {
        fn kind(&self) -> BackendKind {
            self.kind
        }

        fn prove(&self, _: &ProverParams, _: &SetupParams) -> anyhow::Result<()> {
            self.calls.set(self.calls.get() + 1);
            if self.fail {
                anyhow::bail!("mock failure in {}", self.kind)
            }
            Ok(())
        }
    }

    fn mock(kind: BackendKind, fail: bool) -> (Box<dyn Groth16Backend>, Rc<Cell<u32>>) {
        let calls = Rc::new(Cell::new(0));
        let backend = Mock {
            kind,
            calls: Rc::clone(&calls),
            fail,
        };
        (Box::new(backend), calls)
    }

    /// Parameter structs with no I/O behind them (paths are never opened here).
    fn with_params(f: impl FnOnce(&ProverParams, &SetupParams)) {
        let witness = [0u8; 32];
        let root = Path::new("unused");
        let prover = ProverParams::new(root, witness.as_ptr()).unwrap();
        let setup = SetupParams::new(root).unwrap();
        f(&prover, &setup)
    }

    #[test]
    fn absent_configuration_selects_the_default_canonical() {
        let (canonical, _) = mock(BackendKind::Canonical, false);
        let (metal, _) = mock(BackendKind::Metal, false);
        let registry = Registry::new().with(canonical).with(metal);
        assert_eq!(
            registry.select(None).unwrap().kind(),
            BackendKind::Canonical
        );
        assert_eq!(BackendKind::DEFAULT, BackendKind::Canonical);
    }

    #[test]
    fn unknown_name_is_an_error_even_when_canonical_is_available() {
        let (canonical, _) = mock(BackendKind::Canonical, false);
        let registry = Registry::new().with(canonical);
        for value in ["rapidsnark", "", "   ", "cuda", "metal2"] {
            let err = registry.select(Some(value)).unwrap_err();
            assert_eq!(
                err,
                SelectError::Unknown {
                    value: value.to_string()
                },
                "{value:?} must not resolve"
            );
        }
    }

    #[test]
    fn documented_names_parse_ignoring_case_and_whitespace() {
        assert_eq!(
            BackendKind::parse("canonical").unwrap(),
            BackendKind::Canonical
        );
        assert_eq!(
            BackendKind::parse(" CUDA-OXIDE ").unwrap(),
            BackendKind::CudaOxide
        );
        assert_eq!(BackendKind::parse("Metal\n").unwrap(), BackendKind::Metal);
        assert_eq!(
            BackendKind::parse("reference").unwrap(),
            BackendKind::Reference
        );
        for kind in BackendKind::ALL {
            assert_eq!(BackendKind::parse(kind.name()).unwrap(), kind);
            assert_eq!(kind.to_string(), kind.name());
        }
    }

    #[test]
    fn unavailable_kind_is_an_error_listing_what_this_build_has() {
        let (canonical, _) = mock(BackendKind::Canonical, false);
        let registry = Registry::new().with(canonical);
        let err = registry.select(Some("metal")).unwrap_err();
        assert_eq!(
            err,
            SelectError::Unavailable {
                kind: BackendKind::Metal,
                available: vec![BackendKind::Canonical],
            }
        );
        let text = err.to_string();
        assert!(
            text.contains("`metal`") && text.contains("canonical"),
            "{text}"
        );
    }

    #[test]
    fn dispatch_reaches_the_selected_backend_and_no_other() {
        let (canonical, canonical_calls) = mock(BackendKind::Canonical, false);
        let (metal, metal_calls) = mock(BackendKind::Metal, false);
        let registry = Registry::new().with(canonical).with(metal);
        with_params(|prover, setup| {
            assert_eq!(
                registry.select(Some("metal")).unwrap().kind(),
                BackendKind::Metal
            );
            registry.prove(Some("metal"), prover, setup).unwrap();
        });
        assert_eq!(metal_calls.get(), 1, "the selected backend must run");
        assert_eq!(canonical_calls.get(), 0, "no other backend may run");
    }

    #[test]
    fn a_failing_backend_propagates_its_error_and_nothing_answers_in_its_place() {
        let (canonical, canonical_calls) = mock(BackendKind::Canonical, false);
        let (oxide, oxide_calls) = mock(BackendKind::CudaOxide, true);
        let registry = Registry::new().with(canonical).with(oxide);
        with_params(|prover, setup| {
            let err = registry
                .prove(Some("cuda-oxide"), prover, setup)
                .unwrap_err();
            assert!(
                err.to_string().contains("mock failure in cuda-oxide"),
                "{err:#}"
            );
        });
        assert_eq!(oxide_calls.get(), 1);
        assert_eq!(
            canonical_calls.get(),
            0,
            "an error must not fall back to canonical"
        );
    }

    #[test]
    #[should_panic(expected = "registered twice")]
    fn registering_a_kind_twice_is_a_construction_error() {
        let (a, _) = mock(BackendKind::Metal, false);
        let (b, _) = mock(BackendKind::Metal, false);
        let _ = Registry::new().with(a).with(b);
    }

    #[test]
    fn compiled_in_registers_exactly_the_cfg_backends() {
        let mut expected: Vec<BackendKind> = vec![];
        if cfg!(feature = "cuda") {
            expected.push(BackendKind::Canonical);
        }
        if cfg!(feature = "reference") {
            expected.push(BackendKind::Reference);
        }
        if cfg!(feature = "oxide-cpu") {
            expected.push(BackendKind::OxideCpu);
        }
        if cfg!(feature = "cuda-oxide") {
            expected.push(BackendKind::CudaOxide);
        }
        if cfg!(all(
            feature = "metal-cpu",
            not(all(target_os = "macos", target_arch = "aarch64"))
        )) {
            expected.push(BackendKind::MetalCpu);
        }
        assert_eq!(compiled_in().available(), expected);
    }

    #[cfg(not(any(feature = "cuda", feature = "reference", feature = "oxide-cpu")))]
    #[test]
    fn a_build_without_backends_reports_unavailable_rather_than_answering() {
        let err = compiled_in().select(None).unwrap_err();
        assert_eq!(
            err,
            SelectError::Unavailable {
                kind: BackendKind::Canonical,
                available: vec![],
            }
        );
        assert!(err.to_string().contains("available: none"), "{err}");
    }

    #[test]
    #[allow(unused_unsafe)]
    fn environment_present_selects_and_absent_defaults() {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap();
        let (canonical, _) = mock(BackendKind::Canonical, false);
        let (metal, _) = mock(BackendKind::Metal, false);
        let registry = Registry::new().with(canonical).with(metal);
        unsafe { std::env::set_var(BACKEND_ENV, "metal") };
        assert_eq!(
            registry.select_from_env().unwrap().kind(),
            BackendKind::Metal
        );
        unsafe { std::env::set_var(BACKEND_ENV, "no-such-backend") };
        assert!(matches!(
            registry.select_from_env().unwrap_err(),
            SelectError::Unknown { .. }
        ));
        unsafe { std::env::remove_var(BACKEND_ENV) };
        assert_eq!(
            registry.select_from_env().unwrap().kind(),
            BackendKind::Canonical
        );
    }
}
