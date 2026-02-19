use candle_core::{CpuStorage, WithDType};
use float8::F8E4M3 as f8e4m3;
use half::{bf16, f16};
use thiserror::Error;

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
