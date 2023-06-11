#ifndef _EXAMPLE_MATMUL_MATMUL_H_
#define _EXAMPLE_MATMUL_MATMUL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Computes matrix multiplication of input matrices given by `input_a` and
// `input_b`. The output is written at `output`.
// It is the caller's responsibility to reserve the input and the output memory
// buffers in valid sizes.

// The matrices should be shaped as follows.
// - shape(input_a) = (row_dim, inner_dim)
// - shape(input_b) = (inner_dim, col_dim)
// - shape(output) = (row_dim, col_dim)
// where (row_dim, inner_dim, col_dim) MUST be (4, 2048, 2048). Otherwise, they
// exit with code 1.

// This version accepts int8 inputs and int32 outputs.
void matmul_i8(int8_t* input_a, int8_t* input_b, int32_t* output,
               int16_t row_dim, int16_t inner_dim, int16_t col_dim);
// This version accepts f32 inputs outputs.
void matmul_f32(float* input_a, float* input_b, float* output, int16_t row_dim,
                int16_t inner_dim, int16_t col_dim);

//============================================================================//
// The functions below are defined only for the testing purposes. All other
// callers MUST use the above ones instead. These do not require row_dim,
// inner_dim, col_dim to be (4, 2048, 2048).
//============================================================================//
void matmul_i8_impl(int8_t* input_a, int8_t* input_b, int32_t* output,
                    int16_t row_dim, int16_t inner_dim, int16_t col_dim);
void matmul_f32_impl(float* input_a, float* input_b, float* output,
                     int16_t row_dim, int16_t inner_dim, int16_t col_dim);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // _EXAMPLE_MATMUL_MATMUL_H_
