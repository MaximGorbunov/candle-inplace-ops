use std::{
    collections::HashMap,
    ffi::OsStr,
    sync::{PoisonError, RwLock},
};

use candle_core::MetalDevice;
use candle_metal_kernels::metal::{ComputePipeline, ConstantValues, Function, Library};
use objc2::{available, rc::Retained};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
use thiserror::Error;

use crate::{Code, KernelName, LibraryName};
type PipelineKey = (LibraryName, KernelName, Option<ConstantValues>);

pub struct MetalKernelManager<'a> {
    libraries: RwLock<HashMap<LibraryName, Library>>,
    pipelines: RwLock<HashMap<PipelineKey, ComputePipeline>>,
    device: &'a MetalDevice,
}

#[derive(Error, Debug)]
pub enum MetalKernelLoadingError {
    #[error("failed to acquire write lock for libraries, it was poisoned")]
    LockError(String),
    #[error("failed to load library")]
    LibraryLoadingError(String),
    #[error("failed to load function from library")]
    FunctionLoadingError(String),
    #[error("failed to create compute pipeline")]
    PipelineCreationError(String),
}

impl<T> From<PoisonError<T>> for MetalKernelLoadingError {
    fn from(value: PoisonError<T>) -> Self {
        Self::LockError(value.to_string())
    }
}

impl<'a> MetalKernelManager<'a> {
    pub fn new(device: &'a MetalDevice) -> Self {
        MetalKernelManager {
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
    ) -> Result<Library, MetalKernelLoadingError> {
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
                            .map_err(|e| {
                                MetalKernelLoadingError::LibraryLoadingError(e.to_string())
                            })?;
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
    ) -> Result<Function, MetalKernelLoadingError> {
        self.load_metal_library(library_name, code)?
            .get_function(kernel_name, constants)
            .map_err(|e| MetalKernelLoadingError::FunctionLoadingError(e.to_string()))
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
    ) -> Result<ComputePipeline, MetalKernelLoadingError> {
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
                                MetalKernelLoadingError::PipelineCreationError(e.to_string())
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
    ) -> Result<ComputePipeline, MetalKernelLoadingError> {
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

#[cfg(test)]
mod tests {
    use std::panic;

    use candle_core::{Device, Tensor, metal_backend::buffer_o};
    use candle_metal_kernels::{metal::ComputeCommandEncoder, set_params, utils::get_tile_size};
    use objc2_metal::MTLResourceUsage;

    use crate::metal_kernels::{BINARY, linear_split};

    use super::*;

    #[test]
    fn test() {
        let device = Device::new_metal(0).unwrap();
        let metal_device = device.as_metal_device().unwrap();
        let kernel_manager = MetalKernelManager::new(metal_device);
        let tensor = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 0.0], 4, &device).unwrap();
        let pipeline = kernel_manager
            .load_metal_pipeline("example_name", "add_f32", BINARY)
            .unwrap();
        let encoder = metal_device.command_encoder().unwrap();
        let (storage, layout) = tensor.storage_and_layout();
        match &*storage {
            candle_core::Storage::Metal(metal_storage) => {
                let encoder: &ComputeCommandEncoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipeline);
                let length = tensor.layout().shape().elem_count();
                let dtype = storage.dtype();
                let lhs = buffer_o(metal_storage.buffer(), layout, dtype);

                set_params!(encoder, (length, &lhs, 1.0_f32));

                let tile_size = get_tile_size(dtype.size_in_bytes());
                let tiles = length.div_ceil(tile_size);
                let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);

                encoder.use_resource(lhs.buffer, MTLResourceUsage::Write);
                encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            }
            _ => panic!("wrong storage type"),
        }
        println!("{}", tensor);
    }
}
