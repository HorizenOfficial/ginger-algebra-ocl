# algebra-kernels

This library contains ready-to-use GPU kernels for the [ginger-lib](https://github.com/HorizenOfficial/ginger-lib) library.

Currently the next algorithm are implemented:

* Multi-scalar multiplication
* Fast Fourier transform
* Reduce for third round of Marlin (polycommit)

Notes:
 - You should specify the curves, which are used in your algorithm to generate GPU-code for them
 - Needed curves can be specified as features
 - Currently supported curves are: `bn_382`, `bls12_381`, `bls12_377`, `bn254`, `tweedle`
 - All examples in this README will use `bn_382` curve

## Library connection

Add the following to the `dependencies` section of your `Cargo.toml`:

```toml
[dependencies]
# ... 
algebra-kernels = { git = "https://github.com/HorizenOfficial/ginger-algebra-ocl", branch = "ginger_ocl", features = ["bn_382"] }
# ...
```

## Multi scalar multiplication kernel

MSM kernel usage:

```Rust
use algebra::{AffineCurve, BigInteger, Field, FpParameters, PrimeField, ProjectiveCurve};
use algebra_kernels::msm::{get_kernels};
use crossbeam::thread;

fn gpu_msm<G>(
    bases: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt]
) -> G::Projective
where
    G: AffineCurve,
    G::Projective: ProjectiveCurve<Affine = G>
{
    let kernels = get_kernels().unwrap();
    let num_devices = kernels.len();
    let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;
    match thread::scope(|s| -> Result<G::Projective, GPUError> {
        let mut acc = G::Projective::zero();
        let mut threads = Vec::new();

        for ((bases, scalars), kern) in bases
            .chunks(chunk_size)
            .zip(scalars.chunks(chunk_size))
            .zip(kernels.iter())
        {
            threads.push(s.spawn(
                move |_| -> Result<G::Projective, GPUError> {
                    let mut acc = G::Projective::zero();
                    for (bases, scalars) in bases.chunks(kern.n).zip(scalars.chunks(kern.n)) {
                        let result = kern.msm(bases, scalars, bases.len())?;
                        acc.add_assign_mixed(&result.into_affine());
                    }
                    Ok(acc)
                },
            ));
        }

        let mut results = vec![];
        for t in threads {
            results.push(t.join());
        }
        for r in results {
            acc.add_assign_mixed(&r??.into_affine());
        }

        Ok(acc)
    }) {
        Ok(res) => res.unwrap(),
        Err(_) => zero
    }   
}
```

## Fast Fourier transform kernel

FFT kernel usage:

```Rust
use algebra::{FpParameters, PrimeField};
use algebra_kernels::fft::{get_kernels};

fn gpu_fft<F: PrimeField>(a: &mut [F], omega: F, log_n: u32) {
    kernels = get_kernels().unwrap();
    match kernels[0].radix_fft(a, &omega, log_n) {
        Ok(_) => {},
        Err(error) => { panic!("{}", error); }
    }
}
```

## Reduce for third round of Marlin (polycommit) kernel

Polycommit reduce kernel usage:

```Rust
use algebra::{Field, AffineCurve};
use algebra_kernels::polycommit::{get_kernels};

pub fn gpu_polycommit_round_reduce<G: AffineCurve>(
    round_challenge: G::ScalarField,
    round_challenge_inv: G::ScalarField,
    c_l: &mut [G::ScalarField],
    c_r: &[G::ScalarField],
    z_l: &mut [G::ScalarField],
    z_r: &[G::ScalarField],
    k_l: &mut [G::Projective],
    k_r: &[G],
) {
    kernels = get_kernels().unwrap();
    match kernels[0].polycommit_round_reduce(
        round_challenge,
        round_challenge_inv,
        c_l,
        c_r,
        z_l,
        z_r,
        k_l,
        k_r        
    ) {
        Ok(_) => {},
        Err(error) => { panic!("{}", error); }
    }
}
```