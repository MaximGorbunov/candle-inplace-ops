pub mod dtype;

pub type LibraryName = &'static str;
pub type KernelName = &'static str;
pub type Code = &'static str;

#[cfg(feature = "metal")]
pub mod metal_kernel_manager;
#[cfg(feature = "metal")]
pub mod metal_kernels;

#[cfg(feature = "cuda")]
pub mod cuda_kernel_manager;
