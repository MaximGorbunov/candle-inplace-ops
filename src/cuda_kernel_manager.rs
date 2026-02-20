use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, PoisonError, RwLock},
};

use candle_core::{
    CudaDevice, InplaceOp1, WithDType, backend::BackendStorage, bail, builder_arg, cuda::WrapErr,
};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
};
use thiserror::Error;

use crate::{KernelName, LibraryName, dtype::MutWithDType};

pub const BINARY: &str = include_str!(concat!(env!("OUT_DIR"), "/binary.ptx"));

struct InplaceBroadcastAdd<T: MutWithDType> {
    value: T,
}

impl<T: MutWithDType + DeviceRepr> InplaceOp1 for InplaceBroadcastAdd<T> {
    fn name(&self) -> &'static str {
        "broadcast_add"
    }

    fn cpu_fwd(
        &self,
        storage: &mut candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        if storage.dtype() != T::DTYPE {
            bail!(
                "types not matches for tensor: {:?}, scalar: {:?}",
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
            let mut local_index = i;
            strides.iter().enumerate().rev().for_each(|(stride_index, x)| {
                idx = idx + (i % dims[stride_index]) * x;
                local_index /= dims[stride_index];
            });
            buf[idx] = buf[idx] + self.value;
        }
        Ok(())
    }

    fn cuda_fwd(
        &self,
        storage: &mut candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        let shape = layout.shape();
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let device: &CudaDevice = storage.device();
        let func =
            device.get_or_load_custom_func("broadcast_add_f32", "broadcast_scalar", BINARY)?;
        let mut builder = func.builder();
        let lhs = storage.as_cuda_slice_mut::<f32>()?;
        builder.arg(lhs);
        builder.arg(&self.value);
        builder_arg!(builder, elem_count);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use cudarc::nvrtc::{CompileOptions, Ptx};

    use crate::cuda_kernel_manager::BINARY;

    #[test]
    fn test() {
        let device = Device::new_cuda(0).unwrap();
        let t = Tensor::from_slice(&[0.0_f32, 0.0, 0.0], 3, &device).unwrap();
        let op = InplaceBroadcastAdd { value: 1.0_f32 };
        println!("{}", t);
        t.inplace_op1(&op).unwrap();
        println!("{}", t);
    }
}
