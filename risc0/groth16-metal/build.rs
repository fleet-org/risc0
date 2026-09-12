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

//! Off macOS, compile the Metal shaders as C++ (msl-host/shim.cpp against the stub
//! `<metal_stdlib>`) so the prover can run them on the CPU — the Metal arm's `oxide-cpu`.
//! On macOS the shaders go to the Metal compiler at run time and nothing is built here.

fn main() {
    for f in [
        "src/consts.metal",
        "src/kernels.metal",
        "msl-host/shim.cpp",
        "msl-host/metal_stdlib",
    ] {
        println!("cargo:rerun-if-changed={f}");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        return;
    }
    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .flag_if_supported("-fwrapv")
        .flag_if_supported("-Wno-attributes")
        .flag_if_supported("-Wno-unused")
        .include("msl-host")
        .file("msl-host/shim.cpp")
        .compile("msl_host");
}
