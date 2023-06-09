#include "examples/matmul/matmul.h"

#include <cstdlib>

namespace {
int has_valid_dims(int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  return row_dim == 4 && inner_dim && 2048 && col_dim == 2048;
}
}  // namespace

void matmul_i8(int8_t* input_a, int8_t* input_b, int32_t* output,
               int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  if (!has_valid_dims(row_dim, inner_dim, col_dim)) exit(1);
  matmul_i8_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_f32(float* input_a, float* input_b, float* output, int16_t row_dim,
                int16_t inner_dim, int16_t col_dim) {
  if (!has_valid_dims(row_dim, inner_dim, col_dim)) exit(1);
  matmul_f32_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_i8_impl(int8_t* input_a, int8_t* input_b, int32_t* output,
                    int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  // - shape(input_a) = (row_dim, inner_dim)
  // - shape(input_b) = (inner_dim, col_dim)
  // - shape(output) = (row_dim, col_dim)
  for (int16_t row = 0; row < row_dim; ++row) {
    for (int16_t col = 0; col < col_dim; ++col) {
      int32_t accumulated = 0;
      for (int16_t inner = 0; inner < inner_dim; ++inner) {
        // input_a[row][inner] * input_b[inner][col]
        accumulated +=
            input_a[row * inner_dim + inner] * input_b[inner * col_dim + col];
      }
      // output[row][col]
      output[row * col_dim + col] = accumulated;
    }
  }
}

void matmul_f32_impl(float* input_a, float* input_b, float* output,
                     int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  for (int16_t row = 0; row < row_dim; ++row) {
    for (int16_t col = 0; col < col_dim; ++col) {
      float accumulated = 0;
      for (int16_t inner = 0; inner < inner_dim; ++inner) {
        // input_a[row][inner] * input_b[inner][col]
        accumulated +=
            input_a[row * inner_dim + inner] * input_b[inner * col_dim + col];
      }
      // output[row][col]
      output[row * col_dim + col] = accumulated;
    }
  }
}
