// Requires nightly for aarch64
#![cfg_attr(target_arch = "aarch64", feature(stdsimd))]

pub mod gpu;
