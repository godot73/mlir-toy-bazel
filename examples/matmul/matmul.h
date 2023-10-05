#ifndef _EXAMPLE_MATMUL_MATMUL_H_
#define _EXAMPLE_MATMUL_MATMUL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

// Computes matrix multiplication of input matrices given by `input_a` and
// `input_b`. The output is written at `output`.
// PRECONDITIONS:
// - All versions below assume the `output` is propperly initialized, either
// zeros or any previous values to be accumulated. In other words, it is the
// caller's responsibility to initialize `output` beforehand.
// - It is the caller's responsibility to reserve the input and the output
// memory buffers in valid sizes.

// The matrices should be shaped as follows.
// - shape(input_a) = (row_dim, inner_dim)
// - shape(input_b) = (inner_dim, col_dim)
// - shape(output) = (row_dim, col_dim)
// where (row_dim, inner_dim, col_dim) MUST be (4, 2048, 2048). Otherwise, they
// exit with code 1.


struct matmul_i8_params_t {
  const int8_t* input_a;
  const int8_t* input_b;
  int32_t* output;
  size_t row_dim;
  size_t inner_dim;
  size_t col_dim;
};

struct matmul_float_params_t {
  const float* input_a;
  const float* input_b;
  float* output;
  size_t row_dim;
  size_t inner_dim;
  size_t col_dim;
};

// This version accepts int8 inputs and int32 outputs.
void matmul_i8(const int8_t* input_a, const int8_t* input_b, int32_t* output,
               size_t row_dim, size_t inner_dim, size_t col_dim);

// This version accepts f32 inputs outputs.
void matmul_f32(const float* input_a, const float* input_b, float* output,
                size_t row_dim, size_t inner_dim, size_t col_dim);

void matmul_i8_impl_with_params(const struct matmul_i8_params_t* params);
void matmul_i8_impl(const int8_t* input_a, const int8_t* input_b,
                    int32_t* output, size_t row_dim, size_t inner_dim,
                    size_t col_dim);
// This version supports strided inputs and outputs.
// i.e. elem(input_a, i, j) == input_a[i * a_stride[0] + j * a_stride[1]]
void matmul_i8_strided_impl(const int8_t* input_a, const int8_t* input_b,
                            int32_t* output, size_t row_dim, size_t inner_dim,
                            size_t col_dim, const size_t* a_stride,
                            const size_t* b_stride,
                            const size_t* output_stride);

void matmul_f32_impl_with_params(const struct matmul_float_params_t* params);
void matmul_f32_impl(const float* input_a, const float* input_b, float* output,
                     size_t row_dim, size_t inner_dim, size_t col_dim);
// This version supports strided inputs and outputs.
// i.e. elem(input_a, i, j) == input_a[i * a_stride[0] + j * a_stride[1]]
void matmul_f32_strided_impl(const float* input_a, const float* input_b,
                             float* output, size_t row_dim, size_t inner_dim,
                             size_t col_dim, const size_t* a_stride,
                             const size_t* b_stride,
                             const size_t* output_stride);


void matmul_f32_t_impl(const float* input_a, const float* input_b, float* output,
                     size_t row_dim, size_t inner_dim, size_t col_dim);
// This version supports strided inputs and outputs.
// i.e. elem(input_a, i, j) == input_a[i * a_stride[0] + j * a_stride[1]]
void matmul_f32_strided_t_impl(const float* input_a, const float* input_b,
                             float* output, size_t row_dim, size_t inner_dim,
                             size_t col_dim, const size_t* a_stride,
                             const size_t* b_stride,
                             const size_t* output_stride);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // _EXAMPLE_MATMUL_MATMUL_H_
