mod nvidia;
mod utils;

use algebra::{PrimeField, FpParameters};
use itertools::*;

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");

pub trait Limb: Sized + Clone + Copy {
    type LimbType: Clone + std::fmt::Display;
    fn zero() -> Self;
    fn new(val: Self::LimbType) -> Self;
    fn value(&self) -> Self::LimbType;
    fn bits() -> usize;
    fn ptx_info() -> (&'static str, &'static str);
    fn opencl_type() -> &'static str;
    fn limbs_of<T>(value: T) -> Vec<Self> {
        utils::limbs_of::<T, Self::LimbType>(value)
            .into_iter()
            .map(|l| Self::new(l))
            .collect()
    }
}

#[derive(Clone, Copy)]
pub struct Limb32(u32);
impl Limb for Limb32 {
    type LimbType = u32;
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn bits() -> usize {
        32
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u32", "r")
    }
    fn opencl_type() -> &'static str {
        "uint"
    }
}

#[derive(Clone, Copy)]
pub struct Limb64(u64);
impl Limb for Limb64 {
    type LimbType = u64;
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn bits() -> usize {
        64
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u64", "l")
    }
    fn opencl_type() -> &'static str {
        "ulong"
    }
}

fn define_field<L: Limb>(name: &str, limbs: Vec<L>) -> String {
    format!(
        "#define {} ((FIELD){{ {{ {} }} }})",
        name,
        join(limbs.iter().map(|l| l.value()), ", ")
    )
}

/// Generates OpenCL constants and type definitions of prime-field `F`
fn params<F, L: Limb>() -> String
where
    F: PrimeField,
{
    let one = L::limbs_of(F::one()); // Get Montgomery form of F::one()
    let p = L::limbs_of(F::Params::MODULUS); // Get regular form of field modulus
    let r2 = F::Params::R2;
    let limbs = one.len(); // Number of limbs
    let inv = F::Params::INV;
    let limb_def = format!("#define FIELD_limb {}", L::opencl_type());
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let limb_bits_def = format!("#define FIELD_LIMB_BITS {}", L::bits());
    let p_def = define_field("FIELD_P", p);
    let r2_def = define_field("FIELD_R2", L::limbs_of(r2));
    let one_def = define_field("FIELD_ONE", one);
    let zero_def = define_field("FIELD_ZERO", vec![L::zero(); limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv);
    let typedef = format!("typedef struct {{ FIELD_limb val[FIELD_LIMBS]; }} FIELD;");
    join(
        &[
            limb_def,
            limbs_def,
            limb_bits_def,
            one_def,
            p_def,
            r2_def,
            zero_def,
            inv_def,
            typedef,
        ],
        "\n",
    )
}

/// Returns OpenCL source-code of a ff::PrimeField with name `name`
/// Find details in README.md
pub fn field<F, L: Limb>(name: &str) -> String
where
    F: PrimeField,
{
    join(
        &[
            COMMON_SRC.to_string(),
            params::<F, L>(),
            nvidia::field_add_sub_nvidia::<F, L>(),
            String::from(FIELD_SRC),
        ],
        "\n",
    )
    .replace("FIELD", name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::{Field, UniformRand};
    use lazy_static::lazy_static;
    use ocl::{OclPrm, ProQue};
    use algebra::fields::bls12_381::Fr;
    use rand::{thread_rng, Rng};

    #[derive(PartialEq, Debug, Clone, Copy)]
    #[repr(transparent)]
    pub struct GpuFr(pub Fr);
    impl Default for GpuFr {
        fn default() -> Self {
            Self(Fr::zero())
        }
    }
    unsafe impl OclPrm for GpuFr {}

    lazy_static! {
        static ref PROQUE: ProQue = {
            static TEST_SRC: &str = include_str!("cl/test.cl");
            let src = format!(
                "{}\n{}\n{}",
                field::<Fr, Limb32>("Fr32"),
                field::<Fr, Limb64>("Fr64"),
                TEST_SRC
            );
            ProQue::builder().src(src).dims(1).build().unwrap()
        };
    }

    macro_rules! call_kernel {
        ($name:expr, $($arg:expr),*) => {{
            let mut cpu_buffer = vec![GpuFr::default()];
            let buffer = PROQUE.create_buffer::<GpuFr>().unwrap();
            buffer.write(&cpu_buffer).enq().unwrap();
            let kernel =
                PROQUE
                .kernel_builder($name)
                $(.arg($arg))*
                .arg(&buffer)
                .build().unwrap();
            unsafe {
                kernel.enq().unwrap();
            }
            buffer.read(&mut cpu_buffer).enq().unwrap();

            cpu_buffer[0].0
        }};
    }

    #[test]
    fn test_add() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::rand(&mut rng);
            let b = Fr::rand(&mut rng);
            let c = a + b;
            assert_eq!(call_kernel!("test_add_32", GpuFr(a), GpuFr(b)), c);
            assert_eq!(call_kernel!("test_add_64", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::rand(&mut rng);
            let b = Fr::rand(&mut rng);
            let c = a - b;
            assert_eq!(call_kernel!("test_sub_32", GpuFr(a), GpuFr(b)), c);
            assert_eq!(call_kernel!("test_sub_64", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::rand(&mut rng);
            let b = Fr::rand(&mut rng);
            let c = a * b;
            assert_eq!(call_kernel!("test_mul_32", GpuFr(a), GpuFr(b)), c);
            assert_eq!(call_kernel!("test_mul_64", GpuFr(a), GpuFr(b)), c);
        }
    }

    #[test]
    fn test_pow() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::rand(&mut rng);
            let b = rng.gen::<u32>();
            let c = a.pow([b as u64]);
            assert_eq!(call_kernel!("test_pow_32", GpuFr(a), b), c);
            assert_eq!(call_kernel!("test_pow_64", GpuFr(a), b), c);
        }
    }

    #[test]
    fn test_sqr() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::rand(&mut rng);
            let b = a.square();
            assert_eq!(call_kernel!("test_sqr_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_sqr_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::rand(&mut rng);
            let b = a.double();
            assert_eq!(call_kernel!("test_double_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_double_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_unmont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::rand(&mut rng);
            let mut b = a.clone();
            b.0 = a.into_repr();
            assert_eq!(call_kernel!("test_unmont_32", GpuFr(a)), b);
            assert_eq!(call_kernel!("test_unmont_64", GpuFr(a)), b);
        }
    }

    #[test]
    fn test_mont() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = Fr::rand(&mut rng);
            let mut b = a.clone();
            b.0 = a.into_repr();
            assert_eq!(call_kernel!("test_mont_32", GpuFr(b)), a);
            assert_eq!(call_kernel!("test_mont_64", GpuFr(b)), a);
        }
    }
}
