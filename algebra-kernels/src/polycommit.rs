use algebra_cl_gen::gpu::{GPUError, GPUResult, kernel_polycommit, get_prefix_map};
use algebra::{
    AffineCurve, PrimeField
};
use log::{error};
use rust_gpu_tools::*;

use std::any::TypeId;
use std::collections::HashMap;
use std::env;
use std::cmp;

use crate::SingleKernel;

pub fn get_gpu_min_length() -> usize {

    env::var("POLYCOMMIT_GPU_MIN_LENGTH")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid POLYCOMMIT_GPU_MIN_LENGTH! Defaulting to 1024...");
                Ok(1024)
            }
        })
        .unwrap_or(1024)
}

pub struct PolycommitSingleKernel<'a, G>
where
    G: AffineCurve
{
    pub program: &'a opencl::Program,
    pub prefix_map: HashMap<TypeId, String>,

    _phantom: std::marker::PhantomData<<G::ScalarField as PrimeField>::BigInt>,
}

impl<'a, G: AffineCurve> SingleKernel<'a> for PolycommitSingleKernel<'a, G> {

    fn get_program_src() -> String {        
        kernel_polycommit::<G>(true)
    }
}

impl<'a, G> PolycommitSingleKernel<'a, G>
where
    G: AffineCurve
{
    pub fn create(d: opencl::Device) -> GPUResult<PolycommitSingleKernel<'a, G>> {

        let prefix_map = get_prefix_map::<G>();
        let hash_key = TypeId::of::<PolycommitSingleKernel<G>>();
        let program=Self::get_program(&d, hash_key).unwrap();
        
        Ok(PolycommitSingleKernel {
            program,
            prefix_map,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn polycommit_round_reduce<F: PrimeField>(
        &self,
        round_challenge: F,
        round_challenge_inv: F,
        c_l: &mut [F],
        c_r: &[F],
        z_l: &mut [F],
        z_r: &[F],
        k_l: &mut [G::Projective],
        k_r: &[G],
    ) -> GPUResult<()> {

        let mut global_work_size;
        
        global_work_size = cmp::max(
            c_l.len(),
            z_l.len(),
        );

        global_work_size = cmp::max(
            global_work_size,
            k_l.len(),
        );

        let mut kernel_name = self.prefix_map[&TypeId::of::<G>()].clone();
        kernel_name.push_str("polycommit_round_reduce");

        let kernel = self.program.create_kernel(
            kernel_name.as_str(),
            global_work_size,
            None
        );

        let mut rch_buf = self.program.create_buffer::<F>(2)?;
        rch_buf.write_from(0, &[
            round_challenge,
            round_challenge_inv,
        ])?;

        let mut rch_repr_buf = self.program.create_buffer::<F::BigInt>(2)?;
        rch_repr_buf.write_from(0, &[
            round_challenge.into_repr(),
            round_challenge_inv.into_repr(),
        ])?;

        let mut c_l_buf = self.program.create_buffer::<F>(c_l.len())?;
        c_l_buf.write_from(0, c_l)?;

        let mut c_r_buf = self.program.create_buffer::<F>(c_r.len())?;
        c_r_buf.write_from(0, c_r)?;

        let mut z_l_buf = self.program.create_buffer::<F>(z_l.len())?;
        z_l_buf.write_from(0, z_l)?;

        let mut z_r_buf = self.program.create_buffer::<F>(z_r.len())?;
        z_r_buf.write_from(0, z_r)?;

        let mut k_l_buf = self.program.create_buffer::<G::Projective>(k_l.len())?;
        k_l_buf.write_from(0, k_l)?;

        let mut k_r_buf = self.program.create_buffer::<G>(k_r.len())?;
        k_r_buf.write_from(0, k_r)?;

        call_kernel!(
            kernel,
            &rch_buf,
            &rch_repr_buf,
            c_l.len() as u32,
            &c_l_buf,
            &c_r_buf,
            z_l.len() as u32,
            &z_l_buf,
            &z_r_buf,
            k_l.len() as u32,
            &k_l_buf,
            &k_r_buf
        )?;

        c_l_buf.read_into(0, c_l)?;
        z_l_buf.read_into(0, z_l)?;
        k_l_buf.read_into(0, k_l)?;

        Ok(())
    }
}

pub fn get_kernels<'a, G>() -> GPUResult<Vec<PolycommitSingleKernel<'a, G>>>
where
    G: AffineCurve
{
    let mut devices = opencl::Device::all();
    devices.truncate(1);

    let kernels: Vec<_> = devices
        .into_iter()
        .map(|d| (d.clone(), PolycommitSingleKernel::<G>::create(d.clone())))
        .filter_map(|(device, res)| {
            if let Err(ref e) = res {
                error!(
                    "Cannot initialize kernel for device '{}'! Error: {}",
                    device.name(),
                    e
                );
            }
            res.ok()
        })
        .collect();

    if kernels.is_empty() {
        return Err(GPUError::Simple("No working GPUs found!"));
    }

    Ok(kernels)
}
