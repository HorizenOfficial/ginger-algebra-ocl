# ginger-algebra-ocl

This repository contains GPU-powered alogrithms for the [ginger-lib](https://github.com/HorizenOfficial/ginger-lib) library

## Directory structure

The high-level structure of the repository is as follows:

* [`algebra-kernels`](algebra-kernels): GPU-kernels to run the concrete algorithm on GPU
* [`algebra-cl-gen`](algebra-cl-gen): OpenCL code generator for kernels
* [`ff-cl-gen`](ff-cl-gen): basic finite field OpenCL arithmetic code generator

## Build guide

The library compiles on the `stable` toolchain of the Rust compiler. To install the latest version of Rust, first install `rustup` by following the instructions [here](https://rustup.rs/), or via your platform's package manager. Once `rustup` is installed, install the Rust toolchain by invoking:

```bash
rustup install stable
```

GPU features require the OpenCL drivers to be installed in your system.

### Install OpenCL for Nvidia

First you should install the fresh Nvidia drivers into your system. You can find the latest drivers for your GPU [here](https://www.nvidia.com/Download/index.aspx). 

For Linux you can also install the drivers from the repositories of your vendor

After the drivers are installed you shoud install the OpenCL toolkit

#### Ubuntu / Debian

```bash
sudo apt install nvidia-opencl-icd nvidia-opencl-dev
```

### Check the installation

To check if everything works properly you can run the tests using this command:

```bash
cargo test --all-features
```

## Common usage

Most likely you will need the pre-defined GPU kernels, which are placed inside [`algebra-kernels`](algebra-kernels) submodule.

To get more information of how to build you code using this submodule check the [link](algebra-kernels)

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
