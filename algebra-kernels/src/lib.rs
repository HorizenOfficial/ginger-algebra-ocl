
pub mod msm;
pub mod fft;
pub mod polycommit;

#[cfg(test)]
mod test;

use algebra_cl_gen::gpu::GPUResult;
use rust_gpu_tools::*;
use lazy_mut::LazyMut;
use std::any::TypeId;
use std::collections::HashMap;

type CachedProgramsMap = HashMap::<opencl::Device, HashMap<TypeId, opencl::Program>>;

static mut CACHED_PROGRAMS: LazyMut<CachedProgramsMap> = LazyMut::Init(CachedProgramsMap::new);

trait SingleKernel<'a> {

    fn get_program(d: &opencl::Device, hash_key: TypeId) -> GPUResult<&'a opencl::Program> {

        let programs = unsafe {
            CACHED_PROGRAMS.init();
            &mut CACHED_PROGRAMS
        };

        if !programs.contains_key(&d) {
            programs.insert(d.clone(), HashMap::<TypeId, opencl::Program>::new());
        }
        if !programs.get(&d).unwrap().contains_key(&hash_key) {
            programs.get_mut(&d).unwrap().insert(
                hash_key.clone(), 
                opencl::Program::from_opencl(d.clone(), &Self::get_program_src())?
            );
        }

        Ok(programs.get(&d).unwrap().get(&hash_key).unwrap().clone())
    }

    fn get_program_src() -> String;
}

