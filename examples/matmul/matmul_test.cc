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

// Container for inputs and outputs. TypeWraper may be one of Int8TypeWrapper or
// Float32TypeWrapper.
template <typename TypeWrapper>
struct ParamsTmpl {
  using InputType = typename TypeWrapper::InputType;
  using OutputType = typename TypeWrapper::OutputType;

  ParamsTmpl(const std::vector<std::vector<InputType>>& a,
             const std::vector<std::vector<InputType>>& b) {
    if (a[0].size() != b.size()) {
      LOG(FATAL) << "Invalid shapes for multiplication: " << a[0].size()
                 << " vs " << b.size();
    }
    row_dim = a.size();
    inner_dim = a[0].size();
    col_dim = b[0].size();

    input_a = new InputType[row_dim * inner_dim];
    input_b = new InputType[inner_dim * col_dim];
    output = new OutputType[row_dim * col_dim];

    InputType* a_dst = input_a;
    for (const auto& row : a) {
      for (const auto elem : row) {
        *(a_dst++) = elem;
      }
    }
    InputType* b_dst = input_b;
    for (const auto& row : b) {
      for (const auto elem : row) {
        *(b_dst++) = elem;
      }
    }
  }
  ~ParamsTmpl() {
    delete[] input_a;
    delete[] input_b;
    delete[] output;
  }

  std::vector<OutputType> GetOutputVector() const {
    return std::vector<OutputType>(output, output + row_dim * col_dim);
  }

  int16_t row_dim;
  int16_t inner_dim;
  int16_t col_dim;

  InputType* input_a;
  InputType* input_b;
  OutputType* output;
};

// Overload matmul*() for unified typed tests.
void matmul(int8_t* input_a, int8_t* input_b, int32_t* output, int16_t row_dim,
            int16_t inner_dim, int16_t col_dim) {
  matmul_i8(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_impl(int8_t* input_a, int8_t* input_b, int32_t* output,
                 int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  matmul_i8_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul(float* input_a, float* input_b, float* output, int16_t row_dim,
            int16_t inner_dim, int16_t col_dim) {
  matmul_f32(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_impl(float* input_a, float* input_b, float* output, int16_t row_dim,
                 int16_t inner_dim, int16_t col_dim) {
  matmul_f32_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
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
}

TYPED_TEST(MatmulTestTmpl, SecondIdentyty2x2) {
  const std::vector<std::vector<typename TypeParam::InputType>> a = {{1, 2},
                                                                     {3, 4}};
  typename TestFixture::Params p(a, this->Identity(2));
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(a)));
}

TYPED_TEST(MatmulTestTmpl, Basic2x2) {
  typename TestFixture::Params p({{1, 2}, {3, 4}}, {{5, -6}, {7, -8}});
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
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
  const int16_t row_dim = 4;
  const int16_t inner_dim = 2048;
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
