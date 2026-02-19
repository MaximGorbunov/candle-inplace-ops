pub(crate) mod dtype;

type LibraryName = &'static str;
type KernelName = &'static str;
type Code = &'static str;

#[cfg(feature = "metal")]
pub mod metal_kernel_manager;
#[cfg(feature = "metal")]
pub mod metal_kernels;

#[cfg(feature = "cuda")]
pub mod cuda_kernel_manager;
