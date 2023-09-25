#include "matmul.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {
bool has_valid_dims(int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  const bool valid_dims = row_dim == 4 && inner_dim && 2048 && col_dim == 2048;
  std::cout << (valid_dims ? "Valid dimension" : "Invalid dimension")
            << std::endl;
  return valid_dims;
}
}  // namespace

void matmul_i8_impl_with_params(const matmul_i8_params_t* params) {
  matmul_i8_impl(params->input_a, params->input_b, params->output,
                 params->row_dim, params->inner_dim, params->col_dim);
}

void matmul_f32_impl_with_params(const matmul_float_params_t* params) {
  matmul_f32_impl(params->input_a, params->input_b, params->output,
                  params->row_dim, params->inner_dim, params->col_dim);
}

void matmul_i8(const int8_t* input_a, const int8_t* input_b, int32_t* output,
               int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  if (!has_valid_dims(row_dim, inner_dim, col_dim)) exit(1);
  matmul_i8_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_f32(const float* input_a, const float* input_b, float* output,
                int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  if (!has_valid_dims(row_dim, inner_dim, col_dim)) exit(1);
  matmul_f32_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_i8_impl(const int8_t* input_a, const int8_t* input_b,
                    int32_t* output, int16_t row_dim, int16_t inner_dim,
                    int16_t col_dim) {
  // - shape(input_a) = (row_dim, inner_dim)
  // - shape(input_b) = (inner_dim, col_dim)
  // - shape(output) = (row_dim, col_dim)
  for (int64_t row = 0; row < row_dim; ++row) {
    for (int64_t col = 0; col < col_dim; ++col) {
      int64_t accumulated = 0;
      for (int64_t inner = 0; inner < inner_dim; ++inner) {
        // input_a[row][inner] * input_b[inner][col]
        accumulated +=
            input_a[row * inner_dim + inner] * input_b[inner * col_dim + col];
      }
      // output[row][col]
      output[row * col_dim + col] = accumulated;
    }
  }
}

void matmul_f32_impl(const float* input_a, const float* input_b, float* output,
                     int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  for (int64_t row = 0; row < row_dim; ++row) {
    for (int64_t col = 0; col < col_dim; ++col) {
      float accumulated = 0;
      for (int64_t inner = 0; inner < inner_dim; ++inner) {
        // input_a[row][inner] * input_b[inner][col]
        accumulated +=
            input_a[row * inner_dim + inner] * input_b[inner * col_dim + col];
      }
      // output[row][col]
      output[row * col_dim + col] = accumulated;
    }
  }
}
