#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define binary_op(type_name, type, operation, op_name)                         \
  extern "C" __global__ void broadcast_##op_name##_##type_name(                \
      type *tensor, const type value, size_t size) {                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int stride = blockDim.x * gridDim.x;                                       \
    for (size_t i = idx; i < size; i += stride) {                              \
      tensor[i] = tensor[i] operation value;                                   \
    }                                                                          \
  }

#define binary_op_all_types(operation, op_name)                                \
  binary_op(f16, __half, operation, op_name);                                  \
  binary_op(bf16, __nv_bfloat16, operation, op_name);                          \
  binary_op(f32, float, operation, op_name);                                   \
  binary_op(f64, double, operation, op_name);                                  \
  binary_op(u8, uint8_t, operation, op_name);                                  \
  binary_op(u32, uint32_t, operation, op_name);                                \
  binary_op(i64, int64_t, operation, op_name)

binary_op_all_types(+, add);
binary_op_all_types(-, sub);
binary_op_all_types(*, mul);
binary_op_all_types(-, div);
