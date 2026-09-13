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
// The Metal shaders, compiled as C++ and run on the CPU, one thread index at a time — the Metal arm's
// `oxide-cpu`: the same source a Mac's Metal compiler consumes (consts.metal + kernels.metal,
// against the stub <metal_stdlib> beside this file), so its arithmetic, layouts and kernels are
// testable where there is no Metal device. What this cannot check is the Metal compiler itself and
// the device (address spaces, scheduling, the real intrinsics); the Mac run does that.
//
// One entry point, `msl_run`: the kernel by name (the parameter cannot be called `kernel`: the stub
// defines that word away, as MSL's function-kind keyword), its arguments in [[buffer(i)]] order as raw
// pointers (a by-value `constant T&` argument is a pointer to the bytes), the thread count.
// System headers first: the stub poisons MSL's built-in type names (`char` among them) for every
// token after it, and <cstring> declares its functions with `char`. A spelling for the shim's own
// use is declared before the poison takes effect.
#include <cstring>
typedef char msl_char;

#include "metal_stdlib"
#include "../src/consts.metal"
#include "../src/kernels.metal"

namespace {

template <typename T> const T* in(const void* const* args, unsigned i) {
    return static_cast<const T*>(args[i]);
}
template <typename T> T* out(const void* const* args, unsigned i) {
    return static_cast<T*>(const_cast<void*>(args[i]));
}
template <typename T> T val(const void* const* args, unsigned i) {
    T v;
    std::memcpy(&v, args[i], sizeof v);
    return v;
}

} // namespace

extern "C" {

// Returns 0 on success, 1 for an unknown kernel name.
int msl_run(const msl_char* name, const void* const* args, unsigned nargs, unsigned n) {
    (void)nargs;
    if (!std::strcmp(name, "scatter_group")) {
        for (uint g = 0; g < n; g++) {
            scatter_group(in<GroupedCoeff>(args, 0), in<uint>(args, 1), in<Fr>(args, 2), out<Fr>(args, 3),
                          g);
        }
    } else if (!std::strcmp(name, "pointwise_mul")) {
        for (uint i = 0; i < n; i++) {
            pointwise_mul(in<Fr>(args, 0), in<Fr>(args, 1), out<Fr>(args, 2), i);
        }
    } else if (!std::strcmp(name, "pointwise_mul_sub")) {
        for (uint i = 0; i < n; i++) {
            pointwise_mul_sub(in<Fr>(args, 0), in<Fr>(args, 1), in<Fr>(args, 2), out<Fr>(args, 3), i);
        }
    } else if (!std::strcmp(name, "pointwise_scale")) {
        const Fr k = val<Fr>(args, 2);
        for (uint i = 0; i < n; i++) {
            pointwise_scale(in<Fr>(args, 0), in<Fr>(args, 1), k, out<Fr>(args, 3), i);
        }
    } else if (!std::strcmp(name, "bit_reverse")) {
        const uint lg_n = val<uint>(args, 1);
        for (uint i = 0; i < n; i++) {
            bit_reverse(in<Fr>(args, 0), lg_n, out<Fr>(args, 2), i);
        }
    } else if (!std::strcmp(name, "ntt_stage")) {
        const uint len = val<uint>(args, 1);
        const uint stride = val<uint>(args, 3);
        for (uint i = 0; i < n; i++) {
            ntt_stage(in<Fr>(args, 0), len, in<Fr>(args, 2), stride, out<Fr>(args, 4), i);
        }
    } else if (!std::strcmp(name, "digits")) {
        const uint window = val<uint>(args, 1);
        const uint w = val<uint>(args, 2);
        for (uint i = 0; i < n; i++) {
            digits(in<uint>(args, 0), window, w, out<uint>(args, 3), i);
        }
    } else if (!std::strcmp(name, "digits_all")) {
        const uint count = val<uint>(args, 1);
        const uint w = val<uint>(args, 2);
        for (uint i = 0; i < n; i++) {
            digits_all(in<uint>(args, 0), count, w, out<uint>(args, 3), i);
        }
    } else if (!std::strcmp(name, "bucket_sum_g1")) {
        for (uint b = 0; b < n; b++) {
            bucket_sum_g1(in<Aff<Fp>>(args, 0), in<uint>(args, 1), in<uint>(args, 2), out<Jac<Fp>>(args, 3),
                          b);
        }
    } else if (!std::strcmp(name, "bucket_sum_g2")) {
        for (uint b = 0; b < n; b++) {
            bucket_sum_g2(in<Aff<Fp2>>(args, 0), in<uint>(args, 1), in<uint>(args, 2),
                          out<Jac<Fp2>>(args, 3), b);
        }
    } else if (!std::strcmp(name, "jacobian_sum_g1")) {
        for (uint b = 0; b < n; b++) {
            jacobian_sum_g1(in<Jac<Fp>>(args, 0), in<uint>(args, 1), out<Jac<Fp>>(args, 2), b);
        }
    } else if (!std::strcmp(name, "jacobian_sum_g2")) {
        for (uint b = 0; b < n; b++) {
            jacobian_sum_g2(in<Jac<Fp2>>(args, 0), in<uint>(args, 1), out<Jac<Fp2>>(args, 2), b);
        }
    } else {
        return 1;
    }
    return 0;
}

} // extern "C"
