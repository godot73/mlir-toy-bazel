#include "examples/matmul/matmul.h"

#include "absl/log/log.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::ExitedWithCode;

// Wrappers of i8/f32 input output types for typed tests below.
struct Int8TypeWrapper {
  using InputType = int8_t;
  using OutputType = int32_t;
};

struct Float32TypeWrapper {
  using InputType = float;
  using OutputType = float;
};

template <typename ElemType>
void FillValues(const std::vector<std::vector<ElemType>>& values,
                ElemType* dst) {
  for (const auto& row : values) {
    for (const auto& elem : row) *(dst++) = elem;
  }
}

// Container for inputs and outputs. TypeWraper may be one of Int8TypeWrapper or
// Float32TypeWrapper.
template <typename TypeWrapper>
struct ParamsTmpl {
  using InputType = typename TypeWrapper::InputType;
  using OutputType = typename TypeWrapper::OutputType;

  ParamsTmpl(const std::vector<std::vector<InputType>>& a,
             const std::vector<std::vector<InputType>>& b,
             std::vector<std::vector<OutputType>> initial_output0 = {})
      : initial_output(std::move(initial_output0)) {
    if (a[0].size() != b.size()) {
      LOG(FATAL) << "Invalid shapes for multiplication: " << a[0].size()
                 << " vs " << b.size();
    }
    if (initial_output.empty()) {
      // Initialize output to zeros.
      initial_output = std::vector<std::vector<OutputType>>(
          a.size(), std::vector<OutputType>(b[0].size(), 0));
    } else {
      if (a.size() != initial_output.size() ||
          b[0].size() != initial_output[0].size()) {
        LOG(FATAL) << "Invalid output shape for multiplication";
      }
    }

    row_dim = a.size();
    inner_dim = a[0].size();
    col_dim = b[0].size();

    input_a = new InputType[row_dim * inner_dim];
    input_b = new InputType[inner_dim * col_dim];
    output = new OutputType[row_dim * col_dim];

    FillValues<InputType>(a, input_a);
    FillValues<InputType>(b, input_b);
    FillValues<OutputType>(initial_output, output);
  }

  ~ParamsTmpl() {
    delete[] input_a;
    delete[] input_b;
    delete[] output;
  }

  std::vector<OutputType> GetOutputVector() const {
    return std::vector<OutputType>(output, output + row_dim * col_dim);
  }

  void CleanOutputVector() {
    for (int i = 0; i < row_dim * col_dim; ++i) output[i] = 0;
    FillValues<OutputType>(initial_output, output);
  }

  size_t row_dim;
  size_t inner_dim;
  size_t col_dim;

  InputType* input_a;
  InputType* input_b;
  OutputType* output;

  matmul_i8_params_t MakeParams(const matmul_i8_params_t& params) {
    return params;
  }
  matmul_float_params_t MakeParams(const matmul_float_params_t& params) {
    return params;
  }

  std::vector<std::vector<OutputType>> initial_output;
};

// Overload matmul*() for unified typed tests.
void matmul(const int8_t* input_a, const int8_t* input_b, int32_t* output,
            size_t row_dim, size_t inner_dim, size_t col_dim) {
  matmul_i8(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_impl(const int8_t* input_a, const int8_t* input_b, int32_t* output,
                 size_t row_dim, size_t inner_dim, size_t col_dim) {
  matmul_i8_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul(const float* input_a, const float* input_b, float* output,
            size_t row_dim, size_t inner_dim, size_t col_dim) {
  matmul_f32(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_impl(const float* input_a, const float* input_b, float* output,
                 size_t row_dim, size_t inner_dim, size_t col_dim) {
  matmul_f32_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_impl_with_params(const matmul_i8_params_t* params) {
  matmul_impl(params->input_a, params->input_b, params->output, params->row_dim,
              params->inner_dim, params->col_dim);
}

void matmul_impl_with_params(const matmul_float_params_t* params) {
  matmul_impl(params->input_a, params->input_b, params->output, params->row_dim,
              params->inner_dim, params->col_dim);
}

template <typename T>
std::vector<T> Flatten(const std::vector<std::vector<T>>& mat) {
  std::vector<T> ret;
  for (const auto& row : mat) {
    for (auto col : row) ret.push_back(col);
  }
  return ret;
}

template <typename TypeWrapper>
class MatmulTestTmpl : public ::testing::Test {
 protected:
  using InputType = typename TypeWrapper::InputType;
  using OutputType = typename TypeWrapper::OutputType;
  using Params = ParamsTmpl<TypeWrapper>;

  // Constructs a 2-D vector representing an identity matrix of `size`.
  static std::vector<std::vector<InputType>> Identity(int32_t size) {
    std::vector<std::vector<InputType>> ret;
    ret.reserve(size);
    for (int32_t row = 0; row < size; ++row) {
      std::vector<InputType> a_row;
      a_row.reserve(size);
      for (int32_t col = 0; col < size; ++col) {
        a_row.push_back(row == col ? 1 : 0);
      }
      ret.push_back(std::move(a_row));
    }
    return ret;
  }
};

// Test with i8 and f32.
using TestTypes = ::testing::Types<Int8TypeWrapper, Float32TypeWrapper>;
TYPED_TEST_SUITE(MatmulTestTmpl, TestTypes);

TYPED_TEST(MatmulTestTmpl, Basic) {
  typename TestFixture::Params p(this->Identity(2), this->Identity(2));
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(),
              ElementsAreArray(Flatten(this->Identity(2))));

  p.CleanOutputVector();
  auto params_struct = p.MakeParams({.input_a = p.input_a,
                                     .input_b = p.input_b,
                                     .output = p.output,
                                     .row_dim = p.row_dim,
                                     .inner_dim = p.inner_dim,
                                     .col_dim = p.col_dim});
  matmul_impl_with_params(&params_struct);
  EXPECT_THAT(p.GetOutputVector(),
              ElementsAreArray(Flatten(this->Identity(2))));
}

TYPED_TEST(MatmulTestTmpl, AllIdentity2x2) {
  const auto identity2 = this->Identity(2);
  typename TestFixture::Params p(identity2, identity2);
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(identity2)));
}

TYPED_TEST(MatmulTestTmpl, FirstIdentyty2x2) {
  const std::vector<std::vector<typename TypeParam::InputType>> b = {{1, 2},
                                                                     {3, 4}};
  typename TestFixture::Params p(this->Identity(2), b);
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(b)));

  p.CleanOutputVector();
  auto params_struct = p.MakeParams({.input_a = p.input_a,
                                     .input_b = p.input_b,
                                     .output = p.output,
                                     .row_dim = p.row_dim,
                                     .inner_dim = p.inner_dim,
                                     .col_dim = p.col_dim});
  matmul_impl_with_params(&params_struct);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(b)));
}

TYPED_TEST(MatmulTestTmpl, SecondIdentyty2x2) {
  const std::vector<std::vector<typename TypeParam::InputType>> a = {{1, 2},
                                                                     {3, 4}};
  typename TestFixture::Params p(a, this->Identity(2));
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(a)));

  p.CleanOutputVector();
  auto params_struct = p.MakeParams({.input_a = p.input_a,
                                     .input_b = p.input_b,
                                     .output = p.output,
                                     .row_dim = p.row_dim,
                                     .inner_dim = p.inner_dim,
                                     .col_dim = p.col_dim});
  matmul_impl_with_params(&params_struct);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(a)));
}

TYPED_TEST(MatmulTestTmpl, Basic2x2) {
  typename TestFixture::Params p({{1, 2}, {3, 4}}, {{5, -6}, {7, -8}});
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAre(19, -22,  //
                                               43, -50));

  p.CleanOutputVector();
  auto params_struct = p.MakeParams({.input_a = p.input_a,
                                     .input_b = p.input_b,
                                     .output = p.output,
                                     .row_dim = p.row_dim,
                                     .inner_dim = p.inner_dim,
                                     .col_dim = p.col_dim});
  matmul_impl_with_params(&params_struct);
  EXPECT_THAT(p.GetOutputVector(), ElementsAre(19, -22,  //
                                               43, -50));
}

TYPED_TEST(MatmulTestTmpl, Basic2x3x4) {
  typename TestFixture::Params p({{1, 2, 3},  //
                                  {4, 5, 6}},
                                 {{5, -6, 7, -8},     //
                                  {9, -10, 11, -12},  //
                                  {13, -14, 15, -16}});
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAre(62, -68, 74, -80,  //
                                               143, -158, 173, -188));

  p.CleanOutputVector();
  auto params_struct = p.MakeParams({.input_a = p.input_a,
                                     .input_b = p.input_b,
                                     .output = p.output,
                                     .row_dim = p.row_dim,
                                     .inner_dim = p.inner_dim,
                                     .col_dim = p.col_dim});
  matmul_impl_with_params(&params_struct);
  EXPECT_THAT(p.GetOutputVector(), ElementsAre(62, -68, 74, -80,  //
                                               143, -158, 173, -188));
}

TYPED_TEST(MatmulTestTmpl, Accumulated2x3x4) {
  typename TestFixture::Params p({{1, 2, 3},  //
                                  {4, 5, 6}},
                                 {{5, -6, 7, -8},     //
                                  {9, -10, 11, -12},  //
                                  {13, -14, 15, -16}},
                                 {{-1, 2, -3, 4},  //
                                  {-5, 6, -7, 8}}

  );
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAre(61, -66, 71, -76,  //
                                               138, -152, 166, -180));

  p.CleanOutputVector();
  auto params_struct = p.MakeParams({.input_a = p.input_a,
                                     .input_b = p.input_b,
                                     .output = p.output,
                                     .row_dim = p.row_dim,
                                     .inner_dim = p.inner_dim,
                                     .col_dim = p.col_dim});
  matmul_impl_with_params(&params_struct);
  EXPECT_THAT(p.GetOutputVector(), ElementsAre(61, -66, 71, -76,  //
                                               138, -152, 166, -180));
}

TYPED_TEST(MatmulTestTmpl, CheckTestFail) {
  typename TestFixture::Params p(this->Identity(2), this->Identity(2));
  EXPECT_EXIT(
      matmul(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim, p.col_dim),
      ExitedWithCode(1), "");
}

TYPED_TEST(MatmulTestTmpl, CheckTestPass) {
  // Setup
  // Construct a matrix of 4x2k with data filled:
  // [
  //  [0, 1, 2, 3, ..., 99, 0, 1, 2, 3, 99, ...],
  //  [0, 1, 2, 3, ..., 99, 0, 1, 2, 3, 99, ...],
  //  [0, 1, 2, 3, ..., 99, 0, 1, 2, 3, 99, ...],
  //  [0, 1, 2, 3, ..., 99, 0, 1, 2, 3, 99, ...],
  // ]
  const size_t row_dim = 4;
  const size_t inner_dim = 2048;
  std::vector<std::vector<typename TypeParam::InputType>> a;
  a.reserve(row_dim);
  for (int row = 0; row < row_dim; ++row) {
    std::vector<typename TypeParam::InputType> a_row;
    a_row.reserve(inner_dim);
    for (int col = 0; col < inner_dim; ++col) {
      a_row.push_back(col % 100);
    }
    a.push_back(std::move(a_row));
  }
  typename TestFixture::Params p(a, this->Identity(inner_dim));

  // Run
  matmul(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim, p.col_dim);

  // Verify
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(a)));
}

}  // namespace
