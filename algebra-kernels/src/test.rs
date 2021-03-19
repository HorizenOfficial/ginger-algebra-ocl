#[test]
fn test_kernels() {
 
    use algebra::{AffineCurve, curves::*};
    use crate::fft::get_kernels as fft_get_kernels;
    use crate::msm::get_kernels as msm_get_kernels;
    use crate::polycommit::get_kernels as polycommit_get_kernels;

    fn get_kernels<G: AffineCurve>() {

        let kernels = fft_get_kernels::<G::ScalarField>().unwrap();
        assert_ne!(kernels.len(), 0);

        let kernels = msm_get_kernels::<G>().unwrap();
        assert_ne!(kernels.len(), 0);

        let kernels = polycommit_get_kernels::<G>().unwrap();
        assert_ne!(kernels.len(), 0);
    }

    #[cfg(feature = "bn_382")]
    get_kernels::<bn_382::G1Affine>();

    #[cfg(feature = "bls12_381")]
    get_kernels::<bls12_381::G1Affine>();
    
    #[cfg(feature = "bls12_377")]
    get_kernels::<bls12_377::G1Affine>();
    
    #[cfg(feature = "bn254")]
    get_kernels::<bn254::G1Affine>();
    
    #[cfg(feature = "tweedle")]
    get_kernels::<tweedle::dee::Affine>();
}