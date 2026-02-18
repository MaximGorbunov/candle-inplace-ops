#include <metal_stdlib>
using namespace metal;

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

struct strided_indexer {
    METAL_FUNC uint operator()(
        uint idx,
        constant size_t &num_dims,
        constant size_t *dims,
        constant size_t *strides
    ) {
        return get_strided_index(idx, num_dims, dims, strides);
    }
};

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

template<typename T, typename binary, uint W = work_per_thread<T>()>
[[kernel]] void inplace_binary_kernel(
  constant size_t &dim,
  device T *left,
  constant T &right,
  uint tid [[thread_position_in_grid]]
) {
  const uint step = div_ceil<W>(dim);
  #pragma clang loop unroll(full)
  for (uint i = tid; i < dim; i += step) {
    left[i] += right;
  }
}

template<typename T, typename binary, typename indexer = strided_indexer, uint W = work_per_thread<T>()>
[[kernel]] void inplace_binary_kernel_strided(
  constant size_t &dim,
  constant size_t &num_dims,
  constant size_t *dims,
  constant size_t *strides,
  device T *left,
  constant T &right,
  uint tid [[ thread_position_in_grid ]]
) {
  binary op;
  indexer l_index;
  const uint step = div_ceil<W>(dim);
  #pragma clang loop unroll(full)
  for (uint i = tid; i < dim; i += step) {
    uint l_idx = l_index(i, num_dims, dims, strides);
    left[l_idx] = static_cast<T>(op(left[l_idx], right));
  }
}

#define define_op(name, op)           \
struct name {                         \
  template <typename T>               \
  METAL_FUNC T operator()(T x, T y) { \
    return static_cast<T>(op);        \
  }                                   \
};

define_op(add, x + y)
define_op(sub, x - y)
define_op(mul, x * y)
define_op(div, x / y)

#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define init_binary_k(op_name, binary_op, tname, t)                                                 \
  init_kernel(#op_name "_" #tname, inplace_binary_kernel, t, binary_op)                                     \
  init_kernel(#op_name "_" #tname "_strided", inplace_binary_kernel_strided, t, binary_op, strided_indexer)

#if defined(__HAVE_BFLOAT__)
#define init_binary(op)                               \
  init_binary_k(op, op, f16, half)                    \
  init_binary_k(op, op, bf16, bfloat)                 \
  init_binary_k(op, op, f32, float)                   \
  init_binary_k(op, op, u8, uint8_t)                  \
  init_binary_k(op, op, u32, uint32_t)                \
  init_binary_k(op, op, i64, int64_t)                 
#else
#define init_binary(op)                               \
  init_binary_k(op, op, f16, half)                    \
  init_binary_k(op, op, f32, float)                   \
  init_binary_k(op, op, u8, uint8_t)                  \
  init_binary_k(op, op, u32, uint32_t)                \
  init_binary_k(op, op, i64, int64_t)                 
#endif

init_binary(add);
init_binary(sub);
init_binary(mul);
init_binary(div);
