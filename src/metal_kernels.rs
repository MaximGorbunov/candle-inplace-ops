use candle_core::{InplaceOp1, Result, Tensor, WithDType, backend::BackendStorage, bail, metal_backend::buffer_o};
use candle_metal_kernels::{
    metal::{ComputeCommandEncoder, ComputePipeline},
    set_params,
    utils::get_tile_size,
};
use objc2_metal::{MTLResourceUsage, MTLSize};
use half::f16;

use crate::{Code, dtype::MutWithDType, metal_kernel_manager::MetalKernelManager};

pub const BINARY: Code = include_str!("metal_src/binary.metal");

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

pub trait TensorBroadcastAdd<T: MutWithDType> {
    fn inplace_broadcast_add<'a>(&self, value: T, kernel_manager: &'a MetalKernelManager) -> Result<()>;
}
macro_rules! broadcast_op {
    ($struct_name:ident, $operation:tt, $kernel_name:expr, $ty:ty) => {
        pub struct $struct_name<'a> {
            value: $ty,
            kernel_manager: &'a MetalKernelManager<'a>,
        }

        impl<'a> InplaceOp1
            for $struct_name<'a>
        {
            fn name(&self) -> &'static str {
                stringify!($type_name)
            }
            fn cpu_fwd(&self, storage: &mut candle_core::CpuStorage, layout: &candle_core::Layout) -> candle_core::Result<()> {
                if storage.dtype() != <$ty as WithDType>::DTYPE {
                    bail!(
                        "Types not matches for tensor: {:?}, scalar: {:?}",
                        storage.dtype(),
                        <$ty as WithDType>::DTYPE
                    );
                }
                let shape = layout.shape();
                let dims = shape.dims();
                let strides = layout.stride();
                let buf = <$ty as MutWithDType>::to_mut_slice(storage).unwrap();

                let buf = &mut buf[layout.start_offset()..];
                for i in 0..shape.elem_count() {
                    let mut idx = 0;
                    let mut local_i = i;
                    strides
                        .iter()
                        .enumerate()
                        .rev()
                        .for_each(|(stride_index, x)| {
                            idx = idx + (i % dims[stride_index]) * x;
                            local_i /= dims[stride_index];
                        });
                    buf[idx] = buf[idx] $operation self.value;
                }
                Ok(())
            }

            fn metal_fwd(&self,storage: &mut candle_core::MetalStorage,layout: &candle_core::Layout) -> Result<()> {
                if layout.is_contiguous() {
                    let pipeline = self
                        .kernel_manager
                        .load_metal_pipeline(stringify!($type_name), concat!($kernel_name, "_", stringify!($ty)), BINARY)
                        .unwrap();
                    let device = storage.device();
                    let encoder = device.command_encoder().unwrap();

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
                        .load_metal_pipeline(stringify!($type_name), concat!($kernel_name, "_", stringify!($ty), "_strided"), BINARY)
                        .unwrap();
                    let device = storage.device();
                    let encoder = device.command_encoder().unwrap();

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
        impl TensorBroadcastAdd<$ty> for Tensor {
            fn inplace_broadcast_add<'a>(&self, value: $ty, kernel_manager: &'a MetalKernelManager) -> Result<()> {
                let op = $struct_name {
                    value,
                    kernel_manager
                };
                self.inplace_op1(&op)
            }
        }
    };
}

broadcast_op!(InplaceBroadcastAddF16, +, "add", f16);
broadcast_op!(InplaceBroadcastAddF32, +, "add", f32);
broadcast_op!(InplaceBroadcastAddU8, +, "add", u8);
broadcast_op!(InplaceBroadcastAddU32, +, "add", u32);
broadcast_op!(InplaceBroadcastAddI64, +, "add", i64);

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn test() {
        let device = Device::new_metal(0).unwrap();
        let metal_device = device.as_metal_device().unwrap();
        let kernel_manager = MetalKernelManager::new(metal_device);
        let tensor = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 0.0], 4, &device).unwrap();
        let op = InplaceBroadcastAddF32 {
            value: 1.0_f32,
            kernel_manager: &kernel_manager
        };
        println!("before: {}", tensor);
        tensor.inplace_broadcast_add(1.0_f32, &kernel_manager);
        println!("before: {}", tensor);
    }
}
