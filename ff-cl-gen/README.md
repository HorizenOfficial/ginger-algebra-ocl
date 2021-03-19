# ff-cl-gen

> This is a fork of the filecoin [bellperson](https://github.com/filecoin-project/ff-cl-gen) library.

OpenCL code generator for finite-field arithmetic over prime fields constructed with Rust [ginger-lib](https://github.com/HorizenOfficial/ginger-lib) library.

Notes:
 - Limbs are 32/64-bit long, by your choice.
 - The library assumes that the most significant bit of your prime-field is unset. This allows for cheap reductions.

## Usage

Generating OpenCL codes for BN-382 Fr elements:

```rust
let src = ff_cl_gen::field::<algebra::fields::bn_382::Fr, Limb64>("Fr")
```

Generated interface (`FIELD` is substituted with `Fr`):

```c
#define FIELD_LIMB_BITS ... // 32/64
#define FIELD_limb ... // uint/ulong, based on FIELD_LIMB_BITS
#define FIELD_LIMBS ... // Number of limbs for this field
#define FIELD_P ... // Normal form of field modulus
#define FIELD_ONE ... // Montomery form of one
#define FIELD_ZERO ... // Montomery/normal form of zero
#define FIELD_BITS (FIELD_LIMBS * FIELD_LIMB_BITS)

typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;

bool FIELD_gte(FIELD a, FIELD b); // Greater than or equal
bool FIELD_eq(FIELD a, FIELD b); // Equal
FIELD FIELD_sub(FIELD a, FIELD b); // Modular subtraction
FIELD FIELD_add(FIELD a, FIELD b); // Modular addition
FIELD FIELD_mul(FIELD a, FIELD b); // Modular multiplication
FIELD FIELD_sqr(FIELD a); // Modular squaring
FIELD FIELD_double(FIELD a); // Modular doubling
FIELD FIELD_pow(FIELD base, uint exponent); // Modular power
FIELD FIELD_pow_lookup(global FIELD *bases, uint exponent); // Modular power with lookup table for bases
FIELD FIELD_mont(FIELD a); // To montgomery form
FIELD FIELD_unmont(FIELD a); // To regular form
bool FIELD_get_bit(FIELD l, uint i); // Get `i`th bit (From most significant digit)
uint FIELD_get_bits(FIELD l, uint skip, uint window); // Get `window` consecutive bits, (Starting from `skip`th bit from most significant digit)
