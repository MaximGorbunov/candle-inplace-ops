use candle_core::{InplaceOp1, bail, metal_backend::buffer_o};
use candle_metal_kernels::{metal::{ComputeCommandEncoder, ComputePipeline}, set_params, utils::get_tile_size};
use objc2_metal::{MTLResourceUsage, MTLSize};

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

struct InplaceBroadcastAdd<'a, T: MutWithDType> {
    value: T,
    kernel_manager: &'a MetalKernelManager<'a>
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

