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

//! The coefficient record a scatter kernel consumes — `no_std`, because the
//! kernels read it on the device.

use crate::field::Fr;

/// One A- or B-matrix coefficient, grouped by constraint: the scatter unit
/// of every arm (and the record of `preprocessed_coeffs.bin` minus its
/// matrix/constraint fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct GroupedCoeff {
    /// Witness index.
    pub signal: u32,
    /// The coefficient value.
    pub value: Fr,
}
