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

//! Where the device module comes from, and how it is loaded.

use std::{path::Path, sync::Arc};

use anyhow::{anyhow, bail, Context as _, Result};
use cuda_core::simt::{
    embedded::{
        artifact_bundles_from_binary_path, embedded_modules_from_current_exe, EmbeddedModule,
    },
    CudaContext, CudaModule,
};

/// The device module, in one of the forms `cargo oxide` produces.
pub enum ModuleSource {
    /// PTX text (`cargo oxide build` output, or `cargo oxide emit-…`); the
    /// driver JIT-compiles it for the present GPU at load time.
    Ptx(String),
    /// A cubin or fatbin image.
    Image(Vec<u8>),
    /// An artifact bundle read out of a `cargo oxide` build product (an
    /// executable or library carrying the `.oxart` section).
    Bundle(Box<EmbeddedModule>),
    /// The bundle embedded in the running executable — the shape when the
    /// host binary itself is built by `cargo oxide` with the device crate
    /// linked in.
    CurrentExe,
}

impl ModuleSource {
    /// Environment variable naming the module file.
    pub const ENV: &'static str = "RISC0_GROTH16_CUDA_MODULE";

    /// [`Self::ENV`] when set, else [`Self::CurrentExe`].
    pub fn from_env() -> Result<Self> {
        match std::env::var_os(Self::ENV) {
            Some(p) => Self::from_path(Path::new(&p)),
            None => Ok(Self::CurrentExe),
        }
    }

    /// Classify a file by extension: `.ptx` is text, `.cubin`/`.fatbin` an
    /// image, anything else a `cargo oxide` build product with a bundle.
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        let ctx = || format!("{}: reading the device module", path.display());
        Ok(match ext.as_str() {
            "ptx" => Self::Ptx(std::fs::read_to_string(path).with_context(ctx)?),
            "cubin" | "fatbin" => Self::Image(std::fs::read(path).with_context(ctx)?),
            _ => {
                let mut bundles = artifact_bundles_from_binary_path(path)
                    .map_err(|e| anyhow!("{}: {e:?}", path.display()))?;
                if bundles.is_empty() {
                    bail!(
                        "{}: no device artifact bundle in this file (not a `cargo oxide` build product?)",
                        path.display()
                    );
                }
                let module = EmbeddedModule::new(bundles.remove(0)).ok_or_else(|| {
                    anyhow!("{}: bundle carries no loadable payload", path.display())
                })?;
                Self::Bundle(Box::new(module))
            }
        })
    }

    /// Load into `ctx`.
    pub fn load(&self, ctx: &Arc<CudaContext>) -> Result<Arc<CudaModule>> {
        match self {
            Self::Ptx(src) => ctx
                .load_module_from_ptx_src(src)
                .map_err(|e| anyhow!("loading PTX: {e:?}")),
            Self::Image(bytes) => ctx
                .load_module_from_image(bytes)
                .map_err(|e| anyhow!("loading cubin/fatbin: {e:?}")),
            Self::Bundle(module) => module
                .load(ctx)
                .map_err(|e| anyhow!("loading the artifact bundle: {e:?}")),
            Self::CurrentExe => {
                let mut modules = embedded_modules_from_current_exe()
                    .map_err(|e| anyhow!("reading the running executable's bundles: {e:?}"))?;
                if modules.is_empty() {
                    bail!(
                        "no device module: set {} to a .ptx/.cubin/.fatbin or a `cargo oxide` build product \
                         (this executable carries no embedded bundle)",
                        Self::ENV
                    );
                }
                modules
                    .remove(0)
                    .load(ctx)
                    .map_err(|e| anyhow!("loading the embedded bundle: {e:?}"))
            }
        }
    }

    /// A short description for logs.
    pub fn describe(&self) -> String {
        match self {
            Self::Ptx(s) => format!("ptx ({} bytes)", s.len()),
            Self::Image(b) => format!("image ({} bytes)", b.len()),
            Self::Bundle(m) => format!("bundle {} for {}", m.name(), m.target()),
            Self::CurrentExe => "embedded in the running executable".into(),
        }
    }
}
