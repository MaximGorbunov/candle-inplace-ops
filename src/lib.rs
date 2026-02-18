use std::{
    collections::HashMap,
    ffi::OsStr,
    sync::{PoisonError, RwLock},
};

use candle_core::{
    CpuStorage, DType, InplaceOp1, MetalDevice, WithDType, backend::BackendStorage, bail,
    metal_backend::buffer_o,
};
use candle_metal_kernels::{
    metal::{ComputeCommandEncoder, ComputePipeline, ConstantValues, Function, Library},
    set_params,
    utils::get_tile_size,
};
use objc2::{available, rc::Retained};
use objc2_metal::{
    MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLResourceUsage, MTLSize,
};
use thiserror::Error;

type LibraryName = &'static str;
type KernelName = &'static str;
type Code = &'static str;
type PipelineKey = (LibraryName, KernelName, Option<ConstantValues>);

pub const BINARY: Code = include_str!("metal_src/binary.metal");

pub struct KernelManager<'a> {
    libraries: RwLock<HashMap<LibraryName, Library>>,
    pipelines: RwLock<HashMap<PipelineKey, ComputePipeline>>,
    device: &'a MetalDevice,
}

#[derive(Error, Debug)]
pub enum KernelLoadingError {
    #[error("failed to acquire write lock for libraries, it was poisoned")]
    LockError(String),
    #[error("failed to load library")]
    LibraryLoadingError(String),
    #[error("failed to load function from library")]
    FunctionLoadingError(String),
    #[error("failed to create compute pipeline")]
    PipelineCreationError(String),
}

impl<T> From<PoisonError<T>> for KernelLoadingError {
    fn from(value: PoisonError<T>) -> Self {
        Self::LockError(value.to_string())
    }
}

impl<'a> KernelManager<'a> {
    fn new(device: &'a MetalDevice) -> Self {
        KernelManager {
            libraries: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            device,
        }
    }

    #[inline]
    fn get_library(&self, library_name: LibraryName) -> Option<Library> {
        let libraries = self.libraries.read().unwrap();
        libraries.get(library_name).cloned()
    }

    pub fn load_metal_library(
        &self,
        library_name: LibraryName,
        code: &'static str,
    ) -> Result<Library, KernelLoadingError> {
        match self.get_library(library_name) {
            Some(lib) => Ok(lib.clone()),
            None => {
                let mut libraries = self.libraries.write()?;
                match libraries.get(library_name) {
                    Some(lib) => Ok(lib.clone()),
                    None => {
                        let lib = self
                            .device
                            .new_library_with_source(code, Some(&get_compile_options()))
                            .map_err(|e| KernelLoadingError::LibraryLoadingError(e.to_string()))?;
                        libraries.insert(code, lib.clone());
                        Ok(lib)
                    }
                }
            }
        }
    }

    fn load_metal_function(
        &self,
        kernel_name: KernelName,
        library_name: LibraryName,
        code: Code,
        constants: Option<&ConstantValues>,
    ) -> Result<Function, KernelLoadingError> {
        self.load_metal_library(library_name, code)?
            .get_function(kernel_name, constants)
            .map_err(|e| KernelLoadingError::FunctionLoadingError(e.to_string()))
    }

    fn get_pipepine(&self, key: &PipelineKey) -> Option<ComputePipeline> {
        let pipelines = self.pipelines.read().unwrap();
        pipelines.get(key).cloned()
    }

    pub fn load_metal_pipeline_with_constants(
        &self,
        library_name: LibraryName,
        kernel_name: KernelName,
        code: Code,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipeline, KernelLoadingError> {
        let key = (library_name, kernel_name, constants);
        match self.get_pipepine(&key) {
            Some(pipeline) => Ok(pipeline.clone()),
            None => {
                let mut pipelines = self.pipelines.write()?;
                match pipelines.get(&key) {
                    Some(pipeline) => Ok(pipeline.clone()),
                    None => {
                        let (library_name, kernel_name, constants) = key;
                        let function = self.load_metal_function(
                            kernel_name,
                            library_name,
                            code,
                            constants.as_ref(),
                        )?;
                        let pipeline = self
                            .device
                            .new_compute_pipeline_state_with_function(&function)
                            .map_err(|e| {
                                KernelLoadingError::PipelineCreationError(e.to_string())
                            })?;
                        pipelines.insert((library_name, kernel_name, constants), pipeline.clone());
                        Ok(pipeline)
                    }
                }
            }
        }
    }

    pub fn load_metal_pipeline(
        &self,
        library_name: LibraryName,
        kernel_name: KernelName,
        code: Code,
    ) -> Result<ComputePipeline, KernelLoadingError> {
        self.load_metal_pipeline_with_constants(library_name, kernel_name, code, None)
    }
}

fn is_truthy(s: String) -> bool {
    matches!(s.as_str(), "true" | "t" | "yes" | "y" | "1")
}

fn get_env_bool<K: AsRef<OsStr>>(key: K, default: bool) -> bool {
    std::env::var(key).map(is_truthy).unwrap_or(default)
}

fn get_compile_options() -> Retained<MTLCompileOptions> {
    let compile_options = MTLCompileOptions::new();

    let fast_math_enabled = get_env_bool("CANDLE_METAL_ENABLE_FAST_MATH", true);
    // Ref availability:
    // https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode
    if available!(macos = 15, ios = 18) {
        if fast_math_enabled {
            compile_options.setMathMode(MTLMathMode::Fast);
            compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
        } else {
            compile_options.setMathMode(MTLMathMode::Relaxed);
            compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        }
    } else {
        // For older OS versions we use the old api
        #[allow(deprecated)]
        compile_options.setFastMathEnabled(fast_math_enabled);
    }
    compile_options
}

#[derive(Error, Debug)]
pub enum DTypeError {
    #[error("unexpected type")]
    UnexpectedDType,
}

pub trait MutWithDType: WithDType {
     fn to_mut_slice(storage: &mut CpuStorage) -> Result<&mut [Self], DTypeError>;
}

macro_rules! mut_with_dtype {
    ($ty: ty, $dtype: ident) => {
        impl MutWithDType for $ty {
            fn to_mut_slice(storage: &mut CpuStorage) -> Result<&mut [Self], DTypeError> {
                match storage {
                    CpuStorage::$dtype(items) => Ok(items.as_mut_slice()),
                    _ => Err(DTypeError::UnexpectedDType),
                }
            }
        }
    };
}

use float8::F8E4M3 as f8e4m3;
use half::{bf16, f16};

mut_with_dtype!(u8, U8);
mut_with_dtype!(u32, U32);
mut_with_dtype!(i16, I16);
mut_with_dtype!(i32, I32);
mut_with_dtype!(i64, I64);
mut_with_dtype!(f16, F16);
mut_with_dtype!(bf16, BF16);
mut_with_dtype!(f32, F32);
mut_with_dtype!(f64, F64);
mut_with_dtype!(f8e4m3, F8E4M3);

struct InplaceBroadcastAdd<'a, T: MutWithDType> {
    value: T,
    kernel_manager: &'a KernelManager<'a>,
}

impl<'a, T: MutWithDType + candle_metal_kernels::utils::EncoderParam> InplaceOp1 for InplaceBroadcastAdd<'a, T> {
    fn name(&self) -> &'static str {
        "inplace_broadcast_add"
    }

    fn cpu_fwd(
        &self,
        storage: &mut candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        if storage.dtype() != T::DTYPE {
            bail!(
                "Types not matches for tensor: {:?}, scalar: {:?}",
                storage.dtype(),
                T::DTYPE
            );
        }
        let shape = layout.shape();
        let dims = shape.dims();
        let strides = layout.stride();
        let buf = T::to_mut_slice(storage).unwrap();

        let buf = &mut buf[layout.start_offset()..];
        for i in 0..shape.elem_count() {
            let mut idx = 0;
            strides.iter().enumerate().for_each(|(stride_index, x)| {
                idx = idx + (i % dims[stride_index]) * x;
            });
            buf[idx] = buf[idx] + self.value;
        }
        Ok(())
    }

    fn metal_fwd(
        &self,
        storage: &mut candle_core::MetalStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        if layout.is_contiguous() {
            let pipeline = self
                .kernel_manager
                .load_metal_pipeline("example_name", "add_f32", BINARY)
                .unwrap();
            let device = storage.device();
            let encoder = device.command_encoder().unwrap();

            // let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);
            let length = layout.shape().elem_count();
            let dtype = storage.dtype();
            let lhs = buffer_o(storage.buffer(), layout, dtype);

            set_params!(encoder, (length, &lhs, self.value));

            let tile_size = get_tile_size(dtype.size_in_bytes());
            let tiles = length.div_ceil(tile_size);
            let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);

            encoder.use_resource(lhs.buffer, MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            Ok(())
        } else {
            let pipeline = self
                .kernel_manager
                .load_metal_pipeline("example_name", "add_f32_strided", BINARY)
                .unwrap();
            let device = storage.device();
            let encoder = device.command_encoder().unwrap();

            // let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);
            let length = layout.shape().elem_count();
            let dtype = storage.dtype();
            let lhs = buffer_o(storage.buffer(), layout, dtype);
            let shape = layout.dims();
            set_params!(
                encoder,
                (
                    length,
                    shape.len(),
                    shape,
                    layout.stride(),
                    &lhs,
                    self.value
                )
            );

            let tile_size = get_tile_size(dtype.size_in_bytes());
            let tiles = length.div_ceil(tile_size);
            let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);

            encoder.use_resource(lhs.buffer, MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            Ok(())
        }
    }
}

pub(crate) fn linear_split(pipeline: &ComputePipeline, length: usize) -> (MTLSize, MTLSize) {
    let size = length;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);
    let count = size.div_ceil(width);
    let thread_group_count = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    (thread_group_count, thread_group_size)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, IndexOp, Tensor};

    use super::*;

    #[test]
    fn test() {
        let device = Device::new_metal(0).unwrap();
        let metal_device = device.as_metal_device().unwrap();
        let mut kernel_manager = KernelManager::new(metal_device);
        let device = Device::Cpu;
        let op = InplaceBroadcastAdd {
            value: 3.0_f32,
            kernel_manager: &kernel_manager,
        };
        let t2 = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 0.0], 4, &device).unwrap();
        let t = t2.i(2).unwrap();
        println!("before {}", t);
        t.inplace_op1(&op);
        println!("after {}", t2);
    }
}
