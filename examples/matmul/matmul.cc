#include "matmul.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <inttypes.h>

namespace {
bool has_valid_dims(size_t row_dim, size_t inner_dim, size_t col_dim) {
  const bool valid_dims = row_dim == 4 && inner_dim && 2048 && col_dim == 2048;
  std::cout << (valid_dims ? "Valid dimension" : "Invalid dimension")
            << std::endl;
  return valid_dims;
}

template <typename InputType, typename OutputType>
void matmul_strided_impl(const InputType* input_a, const InputType* input_b,
                         OutputType* output, size_t row_dim, size_t inner_dim,
                         size_t col_dim, const size_t* a_stride,
                         const size_t* b_stride, const size_t* output_stride) {
  // - shape(input_a) = (row_dim, inner_dim)
  // - shape(input_b) = (inner_dim, col_dim)
  // - shape(output) = (row_dim, col_dim)

  for (int64_t row = 0; row < row_dim; ++row) {
    for (int64_t col = 0; col < col_dim; ++col) {
      OutputType* output_elem =
          output + (row * output_stride[0] + col * output_stride[1]);
      for (int64_t inner = 0; inner < inner_dim; ++inner) {
        // input_a[row][inner] * input_b[inner][col]
        *output_elem += input_a[row * a_stride[0] + inner * a_stride[1]] *
                        input_b[inner * b_stride[0] + col * b_stride[1]];
      }
    }
  }
}

template <typename InputType, typename OutputType>
void matmul_strided_t_impl(const InputType* input_a, const InputType* input_b,
                         OutputType* output, size_t row_dim, size_t inner_dim,
                         size_t col_dim, const size_t* a_stride,
                         const size_t* b_stride, const size_t* output_stride) {
  // - shape(input_a) = (row_dim, inner_dim)
  // - shape(input_b) = (inner_dim, col_dim)
  // - shape(output) = (row_dim, col_dim)

  for (int64_t row = 0; row < row_dim; ++row) {
    //printf("\n");
    for (int64_t col = 0; col < col_dim; ++col) {
      OutputType* output_elem =
          output + (row * output_stride[0] + col * output_stride[1]);
      for (int64_t inner = 0; inner < inner_dim; ++inner) {
        // input_a[row][inner] * input_b[col][inner]
        *output_elem += input_a[row * a_stride[0] + inner * a_stride[1]] *
                        input_b[col * b_stride[0] + inner * b_stride[1]];
      //printf("%f\t", output_elem[0]);
      }
    }
  }
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
               size_t row_dim, size_t inner_dim, size_t col_dim) {
  if (!has_valid_dims(row_dim, inner_dim, col_dim)) exit(1);
  matmul_i8_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_f32(const float* input_a, const float* input_b, float* output,
                size_t row_dim, size_t inner_dim, size_t col_dim) {
  if (!has_valid_dims(row_dim, inner_dim, col_dim)) exit(1);
  matmul_f32_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_i8_impl(const int8_t* input_a, const int8_t* input_b,
                    int32_t* output, size_t row_dim, size_t inner_dim,
                    size_t col_dim) {
  matmul_strided_impl<int8_t, int32_t>(
      input_a, input_b, output, row_dim, inner_dim, col_dim,
      /*a_stride=*/std::vector<size_t>{inner_dim, 1}.data(),
      /*b_stride=*/std::vector<size_t>{col_dim, 1}.data(),
      /*output_stride=*/std::vector<size_t>{col_dim, 1}.data());
}

void matmul_i8_strided_impl(const int8_t* input_a, const int8_t* input_b,
                            int32_t* output, size_t row_dim, size_t inner_dim,
                            size_t col_dim, const size_t* a_stride,
                            const size_t* b_stride,
                            const size_t* output_stride) {
  matmul_strided_impl<int8_t, int32_t>(input_a, input_b, output, row_dim,
                                       inner_dim, col_dim, a_stride, b_stride,
                                       output_stride);
}

void matmul_f32_impl(const float* input_a, const float* input_b, float* output,
                     size_t row_dim, size_t inner_dim, size_t col_dim) {
  matmul_strided_impl<float, float>(
      input_a, input_b, output, row_dim, inner_dim, col_dim,
      /*a_stride=*/std::vector<size_t>{inner_dim, 1}.data(),
      /*b_stride=*/std::vector<size_t>{col_dim, 1}.data(),
      /*output_stride=*/std::vector<size_t>{col_dim, 1}.data());
}

void matmul_f32_t_impl(const float* input_a, const float* input_b, float* output,
                     size_t row_dim, size_t inner_dim, size_t col_dim) {
  matmul_strided_t_impl<float, float>(
      input_a, input_b, output, row_dim, inner_dim, col_dim,
      /*a_stride=*/std::vector<size_t>{inner_dim, 1}.data(),
      /*b_stride=*/std::vector<size_t>{inner_dim, 1}.data(),
      /*output_stride=*/std::vector<size_t>{col_dim, 1}.data());
}

void matmul_f32_strided_impl(const float* input_a, const float* input_b,
                             float* output, size_t row_dim, size_t inner_dim,
                             size_t col_dim, const size_t* a_stride,
                             const size_t* b_stride,
                             const size_t* output_stride) {
  matmul_strided_impl<float, float>(input_a, input_b, output, row_dim,
                                    inner_dim, col_dim, a_stride, b_stride,
                                    output_stride);
}

void matmul_f32_strided_t_impl(const float* input_a, const float* input_b,
                             float* output, size_t row_dim, size_t inner_dim,
                             size_t col_dim, const size_t* a_stride,
                             const size_t* b_stride,
                             const size_t* output_stride) {
  matmul_strided_t_impl<float, float>(input_a, input_b, output, row_dim,
                                    inner_dim, col_dim, a_stride, b_stride,
                                    output_stride);
}
