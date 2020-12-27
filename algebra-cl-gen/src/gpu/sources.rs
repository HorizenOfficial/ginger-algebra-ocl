use algebra::{PrimeField, AffineCurve, Fp2Parameters};
use itertools::*;
use ff_cl_gen as ffgen;
use ffgen::*;

use std::any::TypeId;
use std::collections::HashMap;

// Instead of having a very large OpenCL program written for a specific curve, with a lot of
// rudandant codes (As OpenCL doesn't have generic types or templates), this module will dynamically
// generate OpenCL codes given different PrimeFields and curves.

static FFT_SRC: &str = include_str!("fft/fft.cl");
static FIELD2_SRC: &str = include_str!("multiexp/field2.cl");
static EC_SRC: &str = include_str!("multiexp/ec.cl");
static MULTIEXP_SRC: &str = include_str!("multiexp/multiexp.cl");
static POLYCOMMIT_ROUND_REDUCE_SRC: &str = include_str!("polycommit/round_reduce.cl");

fn params2<P, L: Limb>(field2: &str, field: &str) -> String
where
    P: Fp2Parameters
{
    let nonresidue = P::NONRESIDUE;
    let nonresidue_def = define_field("FIELD_NONRESIDUE", L::limbs_of(nonresidue));
    join(
        &[
            nonresidue_def,
        ],
        "\n",
    )
        .replace("FIELD2", field2)
        .replace("FIELD", field)
}

fn field2(field2: &str, field: &str) -> String {
    String::from(FIELD2_SRC)
        .replace("FIELD2", field2)
        .replace("FIELD", field)
}

fn fft(field: &str) -> String {
    String::from(FFT_SRC).replace("FIELD", field)
}


fn polycommit_round_reduce(point: &str, exp: &str) -> String {
    String::from(POLYCOMMIT_ROUND_REDUCE_SRC)
        .replace("EXPONENT", exp)
        .replace("POINT", point)
}


#[cfg(not(feature = "blstrs"))]
const BLSTRS_DEF: &str = "";
#[cfg(feature = "blstrs")]
const BLSTRS_DEF: &str = "#define BLSTRS";

fn ec(field: &str, point: &str, exp: &str) -> String {
    String::from(EC_SRC)
        .replace("FIELD", field)
        .replace("POINT", point)
        .replace("EXPONENT", exp)
        .replace("__BLSTRS__", BLSTRS_DEF)
}

fn multiexp(point: &str, exp: &str) -> String {
    String::from(MULTIEXP_SRC)
        .replace("POINT", point)
        .replace("EXPONENT", exp)
}

pub fn get_prefix_map<G>() -> HashMap<TypeId, String> 
where
    G: AffineCurve
{
    let mut prefix_map = HashMap::<TypeId, String>::new();

    #[cfg(feature = "bn_382")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::g::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::G2Affine>()
    {
        prefix_map.insert(TypeId::of::<algebra::curves::bn_382::g::Affine>(), String::from("G_"));
        prefix_map.insert(TypeId::of::<algebra::curves::bn_382::G1Affine>(), String::from("G1_"));
        prefix_map.insert(TypeId::of::<algebra::curves::bn_382::G2Affine>(), String::from("G2_"));
    }

    #[cfg(feature = "bls12_381")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_381::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_381::G2Affine>()
    {
        prefix_map.insert(TypeId::of::<algebra::curves::bls12_381::G1Affine>(), String::from("G1_"));
        prefix_map.insert(TypeId::of::<algebra::curves::bls12_381::G2Affine>(), String::from("G2_"));
    }
    
    #[cfg(feature = "bls12_377")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_377::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_377::G2Affine>()
    {    
        prefix_map.insert(TypeId::of::<algebra::curves::bls12_377::G1Affine>(), String::from("G1_"));
        prefix_map.insert(TypeId::of::<algebra::curves::bls12_377::G2Affine>(), String::from("G2_"));
    }
    
    #[cfg(feature = "bn254")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bn254::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn254::G2Affine>()
    {
        prefix_map.insert(TypeId::of::<algebra::curves::bn254::G1Affine>(), String::from("G1_"));
        prefix_map.insert(TypeId::of::<algebra::curves::bn254::G2Affine>(), String::from("G2_"));
    }
    
    #[cfg(feature = "tweedle")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::tweedle::dee::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::tweedle::dum::Affine>()
    {
        prefix_map.insert(TypeId::of::<algebra::curves::tweedle::dee::Affine>(), String::from("Dee_"));
        prefix_map.insert(TypeId::of::<algebra::curves::tweedle::dum::Affine>(), String::from("Dum_"));
    }

    prefix_map
}

pub fn kernel_fft<F: PrimeField>(limb64: bool) -> String 
{
    vec![
        if limb64 {
            field::<F, Limb64>("Fp")
        } else {
            field::<F, Limb32>("Fp")
        },
        fft("Fp"),
    ]
    .join("\n\n")
}

pub fn kernel_multiexp<G>(limb64: bool) -> String
where
    G: AffineCurve
{   
    let mut src = String::from("");

    #[cfg(feature = "bn_382")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::g::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::G2Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::bn_382::Fr, Limb64>("Fp")
            } else {
                ffgen::field::<algebra::fields::bn_382::Fr, Limb32>("Fp")
            },
            if limb64 {
                ffgen::field::<algebra::fields::bn_382::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::bn_382::Fq, Limb32>("Fq")
            },
            if limb64 {
                params2::<algebra::fields::bn_382::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::bn_382::Fq2Parameters, Limb32>("Fq2", "Fq")
            },            
            field2("Fq2", "Fq"),
            ec("Fp", "G", "Fp"),
            multiexp("G", "Fp"),
            ec("Fq", "G1", "Fp"),
            multiexp("G1", "Fp"),
            ec("Fq2", "G2", "Fp"),
            multiexp("G2", "Fp"),
        ]
        .join("\n\n");
    }

    #[cfg(feature = "bls12_381")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_381::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_381::G2Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::bls12_381::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::bls12_381::Fr, Limb32>("Fr")
            },
            if limb64 {
                ffgen::field::<algebra::fields::bls12_381::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::bls12_381::Fq, Limb32>("Fq")
            },
            if limb64 {
                params2::<algebra::fields::bls12_381::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::bls12_381::Fq2Parameters, Limb32>("Fq2", "Fq")
            },
            field2("Fq2", "Fq"),
            ec("Fq", "G1", "Fr"),
            multiexp("G1", "Fr"),
            ec("Fq2", "G2", "Fr"),
            multiexp("G2", "Fr"),
        ]
        .join("\n\n");
    }
    
    #[cfg(feature = "bls12_377")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_377::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_377::G2Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::bls12_377::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::bls12_377::Fr, Limb32>("Fr")
            },
            if limb64 {
                ffgen::field::<algebra::fields::bls12_377::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::bls12_377::Fq, Limb32>("Fq")
            },
            if limb64 {
                params2::<algebra::fields::bls12_377::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::bls12_377::Fq2Parameters, Limb32>("Fq2", "Fq")
            },
            field2("Fq2", "Fq"),
            ec("Fq", "G1", "Fr"),
            multiexp("G1", "Fr"),
            ec("Fq2", "G2", "Fr"),
            multiexp("G2", "Fr"),
        ]
        .join("\n\n");
    }
    
    #[cfg(feature = "bn254")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bn254::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn254::G2Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::bn254::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::bn254::Fr, Limb32>("Fr")
            },
            if limb64 {
                ffgen::field::<algebra::fields::bn254::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::bn254::Fq, Limb32>("Fq")
            },
            if limb64 {
                params2::<algebra::fields::bn254::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::bn254::Fq2Parameters, Limb32>("Fq2", "Fq")
            },
            field2("Fq2", "Fq"),
            ec("Fq", "G1", "Fr"),
            multiexp("G1", "Fr"),
            ec("Fq2", "G2", "Fr"),
            multiexp("G2", "Fr"),
        ]
        .join("\n\n");
    }
    
    #[cfg(feature = "tweedle")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::tweedle::dee::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::tweedle::dum::Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::tweedle::Fr, Limb64>("Fp")
            } else {
                ffgen::field::<algebra::fields::tweedle::Fr, Limb32>("Fp")
            },
            if limb64 {
                ffgen::field::<algebra::fields::tweedle::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::tweedle::Fq, Limb32>("Fq")
            },
            ec("Fq", "Dee", "Fp"),
            multiexp("Dee", "Fp"),
            ec("Fp", "Dum", "Fp"),
            multiexp("Dum", "Fp"),
        ]
        .join("\n\n");
    }

    src
}

pub fn kernel_polycommit<G>(limb64: bool) -> String
where
    G: AffineCurve
{
    let mut src = String::from("");

    #[cfg(feature = "bn_382")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::g::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn_382::G2Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::bn_382::Fr, Limb64>("Fp")
            } else {
                ffgen::field::<algebra::fields::bn_382::Fr, Limb32>("Fp")
            },
            if limb64 {
                ffgen::field::<algebra::fields::bn_382::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::bn_382::Fq, Limb32>("Fq")
            },
            if limb64 {
                params2::<algebra::fields::bn_382::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::bn_382::Fq2Parameters, Limb32>("Fq2", "Fq")
            },            
            field2("Fq2", "Fq"),
            ec("Fp", "G", "Fp"),
            polycommit_round_reduce("G", "Fp"),
            ec("Fq", "G1", "Fp"),
            polycommit_round_reduce("G1", "Fp"),
            ec("Fq2", "G2", "Fp"),
            polycommit_round_reduce("G2", "Fp"),
        ]
        .join("\n\n");
    }

    #[cfg(feature = "bls12_381")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_381::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_381::G2Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::bls12_381::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::bls12_381::Fr, Limb32>("Fr")
            },
            if limb64 {
                ffgen::field::<algebra::fields::bls12_381::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::bls12_381::Fq, Limb32>("Fq")
            },
            if limb64 {
                params2::<algebra::fields::bls12_381::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::bls12_381::Fq2Parameters, Limb32>("Fq2", "Fq")
            },
            field2("Fq2", "Fq"),
            ec("Fq", "G1", "Fr"),
            polycommit_round_reduce("G1", "Fr"),
            ec("Fq2", "G2", "Fr"),
            polycommit_round_reduce("G2", "Fr"),
        ]
        .join("\n\n");
    }
    
    #[cfg(feature = "bls12_377")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_377::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bls12_377::G2Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::bls12_377::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::bls12_377::Fr, Limb32>("Fr")
            },
            if limb64 {
                ffgen::field::<algebra::fields::bls12_377::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::bls12_377::Fq, Limb32>("Fq")
            },
            if limb64 {
                params2::<algebra::fields::bls12_377::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::bls12_377::Fq2Parameters, Limb32>("Fq2", "Fq")
            },
            field2("Fq2", "Fq"),
            ec("Fq", "G1", "Fr"),
            polycommit_round_reduce("G1", "Fr"),
            ec("Fq2", "G2", "Fr"),
            polycommit_round_reduce("G2", "Fr"),
        ]
        .join("\n\n");
    }
    
    #[cfg(feature = "bn254")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::bn254::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::bn254::G2Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::bn254::Fr, Limb64>("Fr")
            } else {
                ffgen::field::<algebra::fields::bn254::Fr, Limb32>("Fr")
            },
            if limb64 {
                ffgen::field::<algebra::fields::bn254::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::bn254::Fq, Limb32>("Fq")
            },
            if limb64 {
                params2::<algebra::fields::bn254::Fq2Parameters, Limb64>("Fq2", "Fq")
            } else {
                params2::<algebra::fields::bn254::Fq2Parameters, Limb32>("Fq2", "Fq")
            },
            field2("Fq2", "Fq"),
            ec("Fq", "G1", "Fr"),
            polycommit_round_reduce("G1", "Fr"),
            ec("Fq2", "G2", "Fr"),
            polycommit_round_reduce("G2", "Fr"),
        ]
        .join("\n\n");
    }

    #[cfg(feature = "tweedle")]
    if TypeId::of::<G>() == TypeId::of::<algebra::curves::tweedle::dee::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<algebra::curves::tweedle::dum::Affine>()
    {
        src = vec![
            if limb64 {
                ffgen::field::<algebra::fields::tweedle::Fr, Limb64>("Fp")
            } else {
                ffgen::field::<algebra::fields::tweedle::Fr, Limb32>("Fp")
            },
            if limb64 {
                ffgen::field::<algebra::fields::tweedle::Fq, Limb64>("Fq")
            } else {
                ffgen::field::<algebra::fields::tweedle::Fq, Limb32>("Fq")
            },
            ec("Fq", "Dee", "Fp"),
            polycommit_round_reduce("Dee", "Fp"),
            ec("Fp", "Dum", "Fp"),
            polycommit_round_reduce("Dum", "Fp"),
        ]
        .join("\n\n");
    }

    src
}
