use algebra_cl_gen::gpu::{GPUError, GPUResult, kernel_fft};
use algebra::{
    PrimeField
};
use log::{error};
use rust_gpu_tools::*;

use std::any::TypeId;
use std::cmp;

// use crate::utils::CACHED_PROGRAMS;
use std::collections::HashMap;

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 8; // Radix256
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7; // 128

lazy_mut! {
    static mut CACHED_PROGRAMS = HashMap::<opencl::Device, HashMap<TypeId, opencl::Program>>::new();
}

// Multiscalar kernel for a single GPU
pub struct SingleFftKernel<F>
where
    F: PrimeField
{
    pub program: &'static opencl::Program,
   _phantom: std::marker::PhantomData<<F as PrimeField>::BigInt>,
}

impl<F> SingleFftKernel<F>
where
    F: PrimeField
{
    pub fn create(d: opencl::Device) -> GPUResult<SingleFftKernel<F>> {

        let hash_key = TypeId::of::<F>();
        let program;

        unsafe {
            if !CACHED_PROGRAMS.contains_key(&d) {
                CACHED_PROGRAMS.insert(d.clone(), HashMap::<TypeId, opencl::Program>::new());
            }
            if !CACHED_PROGRAMS.get(&d).unwrap().contains_key(&hash_key) {
                CACHED_PROGRAMS.get_mut(&d).unwrap().insert(
                    hash_key.clone(), 
                    opencl::Program::from_opencl(d.clone(), &kernel_fft::<F>(true))?
                );
            }
            program = CACHED_PROGRAMS.get(&d).unwrap().get(&hash_key).unwrap();    
        }

        Ok(SingleFftKernel {
            program,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Peforms a FFT round
    /// * `log_n` - Specifies log2 of number of elements
    /// * `log_p` - Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    /// * `deg` - 1=>radix2, 2=>radix4, 3=>radix8, ...
    /// * `max_deg` - The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    fn radix_fft_round(
        &self,
        src_buffer: &opencl::Buffer<F>,
        dst_buffer: &opencl::Buffer<F>,
        pq_buffer: &opencl::Buffer<F>,
        omegas_buffer: &opencl::Buffer<F>,
        log_n: u32,
        log_p: u32,
        deg: u32,
        max_deg: u32,
    ) -> GPUResult<()> {

        let n = 1u32 << log_n;
        let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
        let global_work_size = (n >> deg) * local_work_size;
        let kernel = self.program.create_kernel(
            "radix_fft",
            global_work_size as usize,
            Some(local_work_size as usize),
        );
        call_kernel!(
            kernel,
            src_buffer,
            dst_buffer,
            pq_buffer,
            omegas_buffer,
            opencl::LocalBuffer::<F>::new(1 << deg),
            n,
            log_p,
            deg,
            max_deg
        )?;
        Ok(())
    }

    /// Performs FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(&self, a: &mut [F], omega: &F, log_n: u32) -> GPUResult<()> {

        let n = 1 << log_n;
        let mut src_buffer = self.program.create_buffer::<F>(n)?;
        let mut dst_buffer = self.program.create_buffer::<F>(n)?;

        let max_deg = cmp::max(cmp::min(MAX_LOG2_RADIX, log_n), 1);

        let mut pq_buffer = self.program.create_buffer::<F>(1 << MAX_LOG2_RADIX >> 1)?;
        let mut omegas_buffer = self.program.create_buffer::<F>(LOG2_MAX_ELEMENTS)?;

        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut pq = vec![F::zero(); 1 << max_deg >> 1];
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        pq[0] = F::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&twiddle);
            }
        }
        pq_buffer.write_from(0, &pq)?;

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![F::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }
        omegas_buffer.write_from(0, &omegas)?;

        src_buffer.write_from(0, &*a)?;
        let mut log_p = 0u32;
        while log_p < log_n {
            let deg = cmp::min(max_deg, log_n - log_p);
            self.radix_fft_round(&src_buffer, &dst_buffer, &pq_buffer, &omegas_buffer, log_n, log_p, deg, max_deg)?;
            log_p += deg;
            std::mem::swap(&mut src_buffer, &mut dst_buffer);
        }

        src_buffer.read_into(0, a)?;

        Ok(())
    }
}

pub fn get_kernels<F>() -> GPUResult<Vec<SingleFftKernel<F>>>
where
    F: PrimeField
{
    let mut devices = opencl::Device::all()?;
    devices.truncate(1);

    let kernels: Vec<_> = devices
        .into_iter()
        .map(|d| (d.clone(), SingleFftKernel::<F>::create(d)))
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
