use algebra::{
    PrimeField, AffineCurve, Fp2Parameters, 
    curves::*
};
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


fn ec(field: &str, point: &str, exp: &str) -> String {
    String::from(EC_SRC)
        .replace("FIELD", field)
        .replace("POINT", point)
        .replace("EXPONENT", exp)        
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
    if TypeId::of::<G>() == TypeId::of::<bn_382::g::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn_382::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn_382::G2Affine>()
    {
        prefix_map.insert(TypeId::of::<bn_382::g::Affine>(), String::from("G_"));
        prefix_map.insert(TypeId::of::<bn_382::G1Affine>(), String::from("G1_"));
        prefix_map.insert(TypeId::of::<bn_382::G2Affine>(), String::from("G2_"));
    }

    #[cfg(feature = "bls12_381")]
    if TypeId::of::<G>() == TypeId::of::<bls12_381::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bls12_381::G2Affine>()
    {
        prefix_map.insert(TypeId::of::<bls12_381::G1Affine>(), String::from("G1_"));
        prefix_map.insert(TypeId::of::<bls12_381::G2Affine>(), String::from("G2_"));
    }
    
    #[cfg(feature = "bls12_377")]
    if TypeId::of::<G>() == TypeId::of::<bls12_377::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bls12_377::G2Affine>()
    {    
        prefix_map.insert(TypeId::of::<bls12_377::G1Affine>(), String::from("G1_"));
        prefix_map.insert(TypeId::of::<bls12_377::G2Affine>(), String::from("G2_"));
    }
    
    #[cfg(feature = "bn254")]
    if TypeId::of::<G>() == TypeId::of::<bn254::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn254::G2Affine>()
    {
        prefix_map.insert(TypeId::of::<bn254::G1Affine>(), String::from("G1_"));
        prefix_map.insert(TypeId::of::<bn254::G2Affine>(), String::from("G2_"));
    }
    
    #[cfg(feature = "tweedle")]
    if TypeId::of::<G>() == TypeId::of::<tweedle::dee::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<tweedle::dum::Affine>()
    {
        prefix_map.insert(TypeId::of::<tweedle::dee::Affine>(), String::from("Dee_"));
        prefix_map.insert(TypeId::of::<tweedle::dum::Affine>(), String::from("Dum_"));
    }

    prefix_map
}

pub fn kernel_fft<F: PrimeField>(limb64: bool) -> String 
{
    vec![
        if limb64 {
            field::<F, Limb64>("Fr")
        } else {
            field::<F, Limb32>("Fr")
        },
        fft("Fr"),
    ]
    .join("\n\n")
}

pub fn kernel_multiexp<G>(limb64: bool) -> String
where
    G: AffineCurve
{   
    let mut src = String::from("");

    #[cfg(feature = "bn_382")]
    if TypeId::of::<G>() == TypeId::of::<bn_382::g::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn_382::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn_382::G2Affine>()
    {
        src = multiexp_src!(
            limb64, bn_382, true,
            "G", "Fr", "Fr",
            "G1", "Fq", "Fr",
            "G2", "Fq2", "Fr"
        );
    }

    #[cfg(feature = "bls12_381")]
    if TypeId::of::<G>() == TypeId::of::<bls12_381::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bls12_381::G2Affine>()
    {
        src = multiexp_src!(
            limb64, bls12_381, true,
            "G1", "Fq", "Fr",
            "G2", "Fq2", "Fr"
        );        
    }
    
    #[cfg(feature = "bls12_377")]
    if TypeId::of::<G>() == TypeId::of::<bls12_377::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bls12_377::G2Affine>()
    {
        src = multiexp_src!(
            limb64, bls12_377, true,
            "G1", "Fq", "Fr",
            "G2", "Fq2", "Fr"
        );        
    }
    
    #[cfg(feature = "bn254")]
    if TypeId::of::<G>() == TypeId::of::<bn254::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn254::G2Affine>()
    {
        src = multiexp_src!(
            limb64, bn254, true,
            "G1", "Fq", "Fr",
            "G2", "Fq2", "Fr"
        );        
    }

    #[cfg(feature = "tweedle")]
    if TypeId::of::<G>() == TypeId::of::<tweedle::dee::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<tweedle::dum::Affine>()
    {
        src = multiexp_src!(
            limb64, tweedle, false,
            "Dee", "Fq", "Fr",
            "Dum", "Fr", "Fq"
        );        
    }

    src

}

pub fn kernel_polycommit<G>(limb64: bool) -> String
where
    G: AffineCurve
{
    let mut src = String::from("");

    #[cfg(feature = "bn_382")]
    if TypeId::of::<G>() == TypeId::of::<bn_382::g::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn_382::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn_382::G2Affine>()
    {
        src = polycommit_round_reduce_src!(
            limb64, bn_382, true,
            "G", "Fr", "Fr",
            "G1", "Fq", "Fr",
            "G2", "Fq2", "Fr"
        );
    }

    #[cfg(feature = "bls12_381")]
    if TypeId::of::<G>() == TypeId::of::<bls12_381::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bls12_381::G2Affine>()
    {
        src = polycommit_round_reduce_src!(
            limb64, bls12_381, true,
            "G1", "Fq", "Fr",
            "G2", "Fq2", "Fr"
        );        
    }
    
    #[cfg(feature = "bls12_377")]
    if TypeId::of::<G>() == TypeId::of::<bls12_377::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bls12_377::G2Affine>()
    {
        src = polycommit_round_reduce_src!(
            limb64, bls12_377, true,
            "G1", "Fq", "Fr",
            "G2", "Fq2", "Fr"
        );        
    }
    
    #[cfg(feature = "bn254")]
    if TypeId::of::<G>() == TypeId::of::<bn254::G1Affine>() ||
       TypeId::of::<G>() == TypeId::of::<bn254::G2Affine>()
    {
        src = polycommit_round_reduce_src!(
            limb64, bn254, true,
            "G1", "Fq", "Fr",
            "G2", "Fq2", "Fr"
        );        
    }

    #[cfg(feature = "tweedle")]
    if TypeId::of::<G>() == TypeId::of::<tweedle::dee::Affine>() ||
       TypeId::of::<G>() == TypeId::of::<tweedle::dum::Affine>()
    {
        src = polycommit_round_reduce_src!(
            limb64, tweedle, false,
            "Dee", "Fq", "Fr",
            "Dum", "Fr", "Fq"
        );        
    }

    src
}
