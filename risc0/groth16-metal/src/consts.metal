// ---- BN254 constants: 8 x 32-bit little-endian limbs, Montgomery R = 2^256 ----
constant uint FP_MOD[8] = {0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u, 0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u};
constant uint FP_INV = 0xe4866389u;           // -p^-1 mod 2^32
constant uint FP_ONE[8] = {0xc58f0d9du, 0xd35d438du, 0xf5c70b3du, 0x0a78eb28u, 0x7879462cu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u};      // R mod p
constant uint FR_MOD[8] = {0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u, 0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u};
constant uint FR_INV = 0xefffffffu;
constant uint FR_ONE[8] = {0x4ffffffbu, 0xac96341cu, 0x9f60cd29u, 0x36fc7695u, 0x7879462eu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u};
