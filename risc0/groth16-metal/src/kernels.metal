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

// GROTH16 s01/4b — the Metal arm's kernels. One kernel per step of the
// canonical sequence (BOUNDARY.md §2), each thread computing one output
// index from the inputs, mirroring `risc0-groth16-oxide`'s kernel bodies
// one to one so the two arms stay comparable. Elements are 8 × 32-bit
// little-endian limbs in Montgomery form (R = 2^256) — the same bytes as
// the Rust `Fr`/`Fp` limbs on a little-endian host. `consts.metal` is
// prepended by the host before compilation.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- 256-bit Montgomery arithmetic

struct fe {
    uint v[8];
};

static inline bool fe_is_zero(fe a) {
    uint acc = 0;
    for (int i = 0; i < 8; ++i)
        acc |= a.v[i];
    return acc == 0;
}

static inline bool fe_geq(thread const uint* a, constant uint* m) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] != m[i])
            return a[i] > m[i];
    }
    return true;
}

static inline void fe_sub_mod(thread uint* a, constant uint* m) {
    long borrow = 0;
    for (int i = 0; i < 8; ++i) {
        long t = (long)a[i] - (long)m[i] - borrow;
        a[i] = (uint)t;
        borrow = t < 0 ? 1 : 0;
    }
}

static inline fe fe_add(fe a, fe b, constant uint* m) {
    fe r;
    ulong carry = 0;
    for (int i = 0; i < 8; ++i) {
        ulong t = (ulong)a.v[i] + (ulong)b.v[i] + carry;
        r.v[i] = (uint)t;
        carry = t >> 32;
    }
    if (carry != 0 || fe_geq(r.v, m))
        fe_sub_mod(r.v, m);
    return r;
}

static inline fe fe_sub(fe a, fe b, constant uint* m) {
    fe r;
    long borrow = 0;
    for (int i = 0; i < 8; ++i) {
        long t = (long)a.v[i] - (long)b.v[i] - borrow;
        r.v[i] = (uint)t;
        borrow = t < 0 ? 1 : 0;
    }
    if (borrow != 0) {
        ulong carry = 0;
        for (int i = 0; i < 8; ++i) {
            ulong t = (ulong)r.v[i] + (ulong)m[i] + carry;
            r.v[i] = (uint)t;
            carry = t >> 32;
        }
    }
    return r;
}

static inline fe fe_neg(fe a, constant uint* m) {
    if (fe_is_zero(a))
        return a;
    fe r;
    long borrow = 0;
    for (int i = 0; i < 8; ++i) {
        long t = (long)m[i] - (long)a.v[i] - borrow;
        r.v[i] = (uint)t;
        borrow = t < 0 ? 1 : 0;
    }
    return r;
}

// Montgomery multiplication, CIOS with 32-bit limbs: a·b·R⁻¹ mod m, reduced.
static inline fe fe_mul(fe a, fe b, constant uint* m, uint inv) {
    uint t[10];
    for (int i = 0; i < 10; ++i)
        t[i] = 0;
    for (int i = 0; i < 8; ++i) {
        ulong carry = 0;
        for (int j = 0; j < 8; ++j) {
            ulong x = (ulong)a.v[j] * (ulong)b.v[i] + (ulong)t[j] + carry;
            t[j] = (uint)x;
            carry = x >> 32;
        }
        {
            ulong x = (ulong)t[8] + carry;
            t[8] = (uint)x;
            t[9] = (uint)(x >> 32);
        }
        uint k = t[0] * inv;
        {
            ulong x = (ulong)k * (ulong)m[0] + (ulong)t[0];
            carry = x >> 32;
        }
        for (int j = 1; j < 8; ++j) {
            ulong x = (ulong)k * (ulong)m[j] + (ulong)t[j] + carry;
            t[j - 1] = (uint)x;
            carry = x >> 32;
        }
        {
            ulong x = (ulong)t[8] + carry;
            t[7] = (uint)x;
            t[8] = t[9] + (uint)(x >> 32);
        }
    }
    fe r;
    for (int i = 0; i < 8; ++i)
        r.v[i] = t[i];
    if (t[8] != 0 || fe_geq(r.v, m))
        fe_sub_mod(r.v, m);
    return r;
}

// ---------------------------------------------------------------- Fp, Fr, Fp2

struct Fp {
    fe v;
};
struct Fr {
    fe v;
};
struct Fp2 {
    Fp c0, c1;
};

static inline Fp fp_zero() {
    Fp r;
    for (int i = 0; i < 8; ++i)
        r.v.v[i] = 0;
    return r;
}
static inline Fp fp_one() {
    Fp r;
    for (int i = 0; i < 8; ++i)
        r.v.v[i] = FP_ONE[i];
    return r;
}
static inline bool fp_is_zero(Fp a) {
    return fe_is_zero(a.v);
}
static inline Fp fp_add(Fp a, Fp b) {
    Fp r;
    r.v = fe_add(a.v, b.v, FP_MOD);
    return r;
}
static inline Fp fp_sub(Fp a, Fp b) {
    Fp r;
    r.v = fe_sub(a.v, b.v, FP_MOD);
    return r;
}
static inline Fp fp_mul(Fp a, Fp b) {
    Fp r;
    r.v = fe_mul(a.v, b.v, FP_MOD, FP_INV);
    return r;
}
static inline Fp fp_sqr(Fp a) {
    return fp_mul(a, a);
}
static inline Fp fp_dbl(Fp a) {
    return fp_add(a, a);
}
static inline Fp fp_neg(Fp a) {
    Fp r;
    r.v = fe_neg(a.v, FP_MOD);
    return r;
}

static inline Fr fr_zero() {
    Fr r;
    for (int i = 0; i < 8; ++i)
        r.v.v[i] = 0;
    return r;
}
static inline Fr fr_add(Fr a, Fr b) {
    Fr r;
    r.v = fe_add(a.v, b.v, FR_MOD);
    return r;
}
static inline Fr fr_sub(Fr a, Fr b) {
    Fr r;
    r.v = fe_sub(a.v, b.v, FR_MOD);
    return r;
}
static inline Fr fr_mul(Fr a, Fr b) {
    Fr r;
    r.v = fe_mul(a.v, b.v, FR_MOD, FR_INV);
    return r;
}

static inline Fp2 fp2_zero() {
    Fp2 r;
    r.c0 = fp_zero();
    r.c1 = fp_zero();
    return r;
}
static inline Fp2 fp2_one() {
    Fp2 r;
    r.c0 = fp_one();
    r.c1 = fp_zero();
    return r;
}
static inline bool fp2_is_zero(Fp2 a) {
    return fp_is_zero(a.c0) && fp_is_zero(a.c1);
}
static inline Fp2 fp2_add(Fp2 a, Fp2 b) {
    Fp2 r;
    r.c0 = fp_add(a.c0, b.c0);
    r.c1 = fp_add(a.c1, b.c1);
    return r;
}
static inline Fp2 fp2_sub(Fp2 a, Fp2 b) {
    Fp2 r;
    r.c0 = fp_sub(a.c0, b.c0);
    r.c1 = fp_sub(a.c1, b.c1);
    return r;
}
static inline Fp2 fp2_dbl(Fp2 a) {
    return fp2_add(a, a);
}
static inline Fp2 fp2_neg(Fp2 a) {
    Fp2 r;
    r.c0 = fp_neg(a.c0);
    r.c1 = fp_neg(a.c1);
    return r;
}
// (a0 + a1 u)(b0 + b1 u) = (a0 b0 − a1 b1) + (a0 b1 + a1 b0) u, u² = −1
static inline Fp2 fp2_mul(Fp2 a, Fp2 b) {
    Fp2 r;
    r.c0 = fp_sub(fp_mul(a.c0, b.c0), fp_mul(a.c1, b.c1));
    r.c1 = fp_add(fp_mul(a.c0, b.c1), fp_mul(a.c1, b.c0));
    return r;
}
static inline Fp2 fp2_sqr(Fp2 a) {
    Fp2 r;
    r.c0 = fp_mul(fp_sub(a.c0, a.c1), fp_add(a.c0, a.c1));
    r.c1 = fp_dbl(fp_mul(a.c0, a.c1));
    return r;
}

// Field operation tables, so the curve code is written once for G1 and G2.
template <typename F> struct Ops;
template <> struct Ops<Fp> {
    static Fp zero() { return fp_zero(); }
    static Fp one() { return fp_one(); }
    static bool is_zero(Fp a) { return fp_is_zero(a); }
    static Fp add(Fp a, Fp b) { return fp_add(a, b); }
    static Fp sub(Fp a, Fp b) { return fp_sub(a, b); }
    static Fp mul(Fp a, Fp b) { return fp_mul(a, b); }
    static Fp sqr(Fp a) { return fp_sqr(a); }
    static Fp dbl(Fp a) { return fp_dbl(a); }
    static Fp neg(Fp a) { return fp_neg(a); }
};
template <> struct Ops<Fp2> {
    static Fp2 zero() { return fp2_zero(); }
    static Fp2 one() { return fp2_one(); }
    static bool is_zero(Fp2 a) { return fp2_is_zero(a); }
    static Fp2 add(Fp2 a, Fp2 b) { return fp2_add(a, b); }
    static Fp2 sub(Fp2 a, Fp2 b) { return fp2_sub(a, b); }
    static Fp2 mul(Fp2 a, Fp2 b) { return fp2_mul(a, b); }
    static Fp2 sqr(Fp2 a) { return fp2_sqr(a); }
    static Fp2 dbl(Fp2 a) { return fp2_dbl(a); }
    static Fp2 neg(Fp2 a) { return fp2_neg(a); }
};

// ---------------------------------------------------------------- curve (y² = x³ + b, a = 0)

// Affine points as stored by the zkey (Montgomery coordinates); the host
// encodes the point at infinity as x = y = 0, which is off every curve here.
template <typename F> struct Aff {
    F x, y;
};
template <typename F> struct Jac {
    F x, y, z;
}; // infinity ⇔ z = 0

template <typename F> static inline bool aff_is_inf(Aff<F> p) {
    return Ops<F>::is_zero(p.x) && Ops<F>::is_zero(p.y);
}
template <typename F> static inline bool jac_is_inf(Jac<F> p) {
    return Ops<F>::is_zero(p.z);
}
template <typename F> static inline Jac<F> jac_inf() {
    Jac<F> r;
    r.x = Ops<F>::one();
    r.y = Ops<F>::one();
    r.z = Ops<F>::zero();
    return r;
}
template <typename F> static inline Jac<F> jac_from_aff(Aff<F> p) {
    Jac<F> r;
    r.x = p.x;
    r.y = p.y;
    r.z = Ops<F>::one();
    return r;
}

// dbl-2009-l
template <typename F> static inline Jac<F> jac_dbl(Jac<F> p) {
    if (jac_is_inf(p))
        return p;
    F a = Ops<F>::sqr(p.x);
    F b = Ops<F>::sqr(p.y);
    F c = Ops<F>::sqr(b);
    F d = Ops<F>::dbl(Ops<F>::sub(Ops<F>::sub(Ops<F>::sqr(Ops<F>::add(p.x, b)), a), c));
    F e = Ops<F>::add(Ops<F>::dbl(a), a);
    F f = Ops<F>::sqr(e);
    Jac<F> r;
    r.x = Ops<F>::sub(f, Ops<F>::dbl(d));
    F eight_c = Ops<F>::dbl(Ops<F>::dbl(Ops<F>::dbl(c)));
    r.y = Ops<F>::sub(Ops<F>::mul(e, Ops<F>::sub(d, r.x)), eight_c);
    r.z = Ops<F>::dbl(Ops<F>::mul(p.y, p.z));
    return r;
}

// madd-2007-bl: Jacobian + affine (the bucket step)
template <typename F> static inline Jac<F> jac_madd(Jac<F> p, Aff<F> q) {
    if (aff_is_inf(q))
        return p;
    if (jac_is_inf(p))
        return jac_from_aff(q);
    F z1z1 = Ops<F>::sqr(p.z);
    F u2 = Ops<F>::mul(q.x, z1z1);
    F s2 = Ops<F>::mul(Ops<F>::mul(q.y, p.z), z1z1);
    F h = Ops<F>::sub(u2, p.x);
    F rr = Ops<F>::dbl(Ops<F>::sub(s2, p.y));
    if (Ops<F>::is_zero(h)) {
        if (Ops<F>::is_zero(rr))
            return jac_dbl(p);
        return jac_inf<F>();
    }
    F hh = Ops<F>::sqr(h);
    F i = Ops<F>::dbl(Ops<F>::dbl(hh));
    F j = Ops<F>::mul(h, i);
    F v = Ops<F>::mul(p.x, i);
    Jac<F> r;
    r.x = Ops<F>::sub(Ops<F>::sub(Ops<F>::sqr(rr), j), Ops<F>::dbl(v));
    r.y = Ops<F>::sub(Ops<F>::mul(rr, Ops<F>::sub(v, r.x)), Ops<F>::dbl(Ops<F>::mul(p.y, j)));
    r.z = Ops<F>::sub(Ops<F>::sub(Ops<F>::sqr(Ops<F>::add(p.z, h)), z1z1), hh);
    return r;
}

// add-2007-bl: Jacobian + Jacobian (the reduction above the bucket sums), the
// same branches as `Jacobian::add` in risc0-groth16-core: either side at
// infinity returns the other; h = 0 doubles when rr = 0 and is infinity otherwise.
template <typename F> static inline Jac<F> jac_add(Jac<F> p, Jac<F> q) {
    if (jac_is_inf(p))
        return q;
    if (jac_is_inf(q))
        return p;
    F z1z1 = Ops<F>::sqr(p.z);
    F z2z2 = Ops<F>::sqr(q.z);
    F u1 = Ops<F>::mul(p.x, z2z2);
    F u2 = Ops<F>::mul(q.x, z1z1);
    F s1 = Ops<F>::mul(Ops<F>::mul(p.y, q.z), z2z2);
    F s2 = Ops<F>::mul(Ops<F>::mul(q.y, p.z), z1z1);
    F h = Ops<F>::sub(u2, u1);
    F rr = Ops<F>::dbl(Ops<F>::sub(s2, s1));
    if (Ops<F>::is_zero(h)) {
        if (Ops<F>::is_zero(rr))
            return jac_dbl(p);
        return jac_inf<F>();
    }
    F i = Ops<F>::sqr(Ops<F>::dbl(h));
    F j = Ops<F>::mul(h, i);
    F v = Ops<F>::mul(u1, i);
    Jac<F> r;
    r.x = Ops<F>::sub(Ops<F>::sub(Ops<F>::sqr(rr), j), Ops<F>::dbl(v));
    r.y = Ops<F>::sub(Ops<F>::mul(rr, Ops<F>::sub(v, r.x)), Ops<F>::dbl(Ops<F>::mul(s1, j)));
    r.z = Ops<F>::mul(Ops<F>::sub(Ops<F>::sub(Ops<F>::sqr(Ops<F>::add(p.z, q.z)), z1z1), z2z2), h);
    return r;
}

// ---------------------------------------------------------------- records

// A grouped coefficient as the host packs it (and as preprocessed_coeffs.bin
// stores it): signal at 0, value at 16, 48 bytes.
struct GroupedCoeff {
    uint signal;
    uint pad[3];
    Fr value;
};

// ---------------------------------------------------------------- kernels (one output per thread)

kernel void scatter_group(device const GroupedCoeff* coeffs [[buffer(0)]],
                          device const uint* starts [[buffer(1)]],
                          device const Fr* witness [[buffer(2)]],
                          device Fr* out [[buffer(3)]],
                          uint g [[thread_position_in_grid]]) {
    uint from = starts[g], to = starts[g + 1];
    Fr sum = fr_zero();
    for (uint k = from; k < to; ++k) {
        GroupedCoeff c = coeffs[k];
        sum = fr_add(sum, fr_mul(witness[c.signal], c.value));
    }
    out[g] = sum;
}

kernel void pointwise_mul(device const Fr* a [[buffer(0)]],
                          device const Fr* b [[buffer(1)]],
                          device Fr* out [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    out[i] = fr_mul(a[i], b[i]);
}

kernel void pointwise_mul_sub(device const Fr* a [[buffer(0)]],
                              device const Fr* b [[buffer(1)]],
                              device const Fr* c [[buffer(2)]],
                              device Fr* out [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    out[i] = fr_sub(fr_mul(a[i], b[i]), c[i]);
}

// out[i] = a[i] · table[i] · k   (coset shift powers and the 1/n of an inverse NTT in one pass)
kernel void pointwise_scale(device const Fr* a [[buffer(0)]],
                            device const Fr* table [[buffer(1)]],
                            constant Fr& k [[buffer(2)]],
                            device Fr* out [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    out[i] = fr_mul(fr_mul(a[i], table[i]), k);
}

kernel void bit_reverse(device const Fr* a [[buffer(0)]],
                        constant uint& lg_n [[buffer(1)]],
                        device Fr* out [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    uint j = reverse_bits(i) >> (32u - lg_n);
    out[i] = a[j];
}

// One radix-2 Cooley–Tukey stage, out of place (see the Rust twin).
kernel void ntt_stage(device const Fr* a [[buffer(0)]],
                      constant uint& len [[buffer(1)]],
                      device const Fr* tw [[buffer(2)]],
                      constant uint& stride [[buffer(3)]],
                      device Fr* out [[buffer(4)]],
                      uint i [[thread_position_in_grid]]) {
    uint hl = len / 2;
    uint k = i % len;
    if (k < hl) {
        Fr u = a[i];
        Fr v = fr_mul(a[i + hl], tw[k * stride]);
        out[i] = fr_add(u, v);
    } else {
        Fr u = a[i - hl];
        Fr v = fr_mul(a[i], tw[(k - hl) * stride]);
        out[i] = fr_sub(u, v);
    }
}

// The w-bit digit `window` of canonical scalar i (8 little-endian 32-bit limbs).
kernel void digits(device const uint* scalars [[buffer(0)]],
                   constant uint& window [[buffer(1)]],
                   constant uint& w [[buffer(2)]],
                   device uint* out [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    uint bit = window * w;
    uint limb = bit / 32u;
    uint shift = bit % 32u;
    uint d = 0;
    if (limb < 8u) {
        d = scalars[i * 8u + limb] >> shift;
        if (shift + w > 32u && limb + 1u < 8u) {
            d |= scalars[i * 8u + limb + 1u] << (32u - shift);
        }
    }
    out[i] = d & ((1u << w) - 1u);
}

kernel void digits_all(device const uint* scalars [[buffer(0)]],
                       constant uint& n [[buffer(1)]],
                       constant uint& w [[buffer(2)]],
                       device uint* out [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    // every window in one launch: output i is scalar i % n, window i / n
    uint s = i % n;
    uint window = i / n;
    uint bit = window * w;
    uint limb = bit / 32u;
    uint shift = bit % 32u;
    uint d = 0;
    if (limb < 8u) {
        d = scalars[s * 8u + limb] >> shift;
        if (shift + w > 32u && limb + 1u < 8u) {
            d |= scalars[s * 8u + limb + 1u] << (32u - shift);
        }
    }
    out[i] = d & ((1u << w) - 1u);
}

kernel void bucket_sum_g1(device const Aff<Fp>* points [[buffer(0)]],
                          device const uint* order [[buffer(1)]],
                          device const uint* starts [[buffer(2)]],
                          device Jac<Fp>* out [[buffer(3)]],
                          uint b [[thread_position_in_grid]]) {
    Jac<Fp> acc = jac_inf<Fp>();
    for (uint k = starts[b]; k < starts[b + 1]; ++k)
        acc = jac_madd(acc, points[order[k]]);
    out[b] = acc;
}

kernel void bucket_sum_g2(device const Aff<Fp2>* points [[buffer(0)]],
                          device const uint* order [[buffer(1)]],
                          device const uint* starts [[buffer(2)]],
                          device Jac<Fp2>* out [[buffer(3)]],
                          uint b [[thread_position_in_grid]]) {
    Jac<Fp2> acc = jac_inf<Fp2>();
    for (uint k = starts[b]; k < starts[b + 1]; ++k)
        acc = jac_madd(acc, points[order[k]]);
    out[b] = acc;
}

// One level of the bounded-chain reduction above the bucket sums
// (`pipeline::plan_ranges`): out[b] = Σ sums[k] for k in starts[b]..starts[b+1].
kernel void jacobian_sum_g1(device const Jac<Fp>* sums [[buffer(0)]],
                            device const uint* starts [[buffer(1)]],
                            device Jac<Fp>* out [[buffer(2)]],
                            uint b [[thread_position_in_grid]]) {
    Jac<Fp> acc = jac_inf<Fp>();
    for (uint k = starts[b]; k < starts[b + 1]; ++k)
        acc = jac_add(acc, sums[k]);
    out[b] = acc;
}

kernel void jacobian_sum_g2(device const Jac<Fp2>* sums [[buffer(0)]],
                            device const uint* starts [[buffer(1)]],
                            device Jac<Fp2>* out [[buffer(2)]],
                            uint b [[thread_position_in_grid]]) {
    Jac<Fp2> acc = jac_inf<Fp2>();
    for (uint k = starts[b]; k < starts[b + 1]; ++k)
        acc = jac_add(acc, sums[k]);
    out[b] = acc;
}
