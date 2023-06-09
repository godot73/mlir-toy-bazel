#include "examples/matmul/matmul.h"

#include "absl/log/log.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::ExitedWithCode;

template <typename InTy, typename OutTy>
struct ParamsTmpl {
  ParamsTmpl(const std::vector<std::vector<InTy>>& a,
             const std::vector<std::vector<InTy>>& b) {
    if (a[0].size() != b.size()) {
      LOG(FATAL) << "Invalid shapes for multiplication: " << a[0].size()
                 << " vs " << b.size();
    }
    row_dim = a.size();
    inner_dim = a[0].size();
    col_dim = b[0].size();

    input_a = new InTy[row_dim * inner_dim];
    input_b = new InTy[inner_dim * col_dim];
    output = new OutTy[row_dim * col_dim];

    int8_t* a_dst = input_a;
    for (const auto& row : a) {
      for (const auto elem : row) {
        *(a_dst++) = elem;
      }
    }
    int8_t* b_dst = input_b;
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

  std::vector<OutTy> GetOutputVector() const {
    return std::vector<OutTy>(output, output + row_dim * col_dim);
  }

  int16_t row_dim;
  int16_t inner_dim;
  int16_t col_dim;

  InTy* input_a;
  InTy* input_b;
  OutTy* output;
};

void matmul(int8_t* input_a, int8_t* input_b, int32_t* output, int16_t row_dim,
            int16_t inner_dim, int16_t col_dim) {
  matmul_i8(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

void matmul_impl(int8_t* input_a, int8_t* input_b, int32_t* output,
                 int16_t row_dim, int16_t inner_dim, int16_t col_dim) {
  matmul_i8_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
}

class MatmulTest : public ::testing::Test {
 protected:
  using InputType = int8_t;
  using OutputType = int32_t;
  using Params = ParamsTmpl<InputType, OutputType>;

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

  static std::vector<InputType> Flatten(
      const std::vector<std::vector<InputType>>& mat) {
    std::vector<InputType> ret;
    for (const auto& row : mat) {
      for (auto col : row) ret.push_back(col);
    }
    return ret;
  }
};

TEST_F(MatmulTest, AllIdentity2x2) {
  Params p(Identity(2), Identity(2));
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(Identity(2))));
}

TEST_F(MatmulTest, FirstIdentyty2x2) {
  const std::vector<std::vector<InputType>> b = {{1, 2}, {3, 4}};
  Params p(Identity(2), b);
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(b)));
}

TEST_F(MatmulTest, SecondIdentyty2x2) {
  const std::vector<std::vector<InputType>> a = {{1, 2}, {3, 4}};
  Params p(a, Identity(2));
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(a)));
}

TEST_F(MatmulTest, Basic2x2) {
  Params p({{1, 2}, {3, 4}}, {{5, -6}, {7, -8}});
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAre(19, -22,  //
                                               43, -50));
}

TEST_F(MatmulTest, Basic2x3x4) {
  Params p({{1, 2, 3},  //
            {4, 5, 6}},
           {{5, -6, 7, -8},     //
            {9, -10, 11, -12},  //
            {13, -14, 15, -16}});
  matmul_impl(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim,
              p.col_dim);
  EXPECT_THAT(p.GetOutputVector(), ElementsAre(62, -68, 74, -80,  //
                                               143, -158, 173, -188));
}

TEST_F(MatmulTest, CheckTestFail) {
  Params p(Identity(2), Identity(2));
  EXPECT_EXIT(
      matmul(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim, p.col_dim),
      ExitedWithCode(1), "");
}

TEST_F(MatmulTest, CheckTestPass) {
  // Setup
  // Construct a matrix of 4x2k with data filled:
  // [
  //  [0, 1, 2, 3, ..., 99, 0, 1, 2, 3, 99, ...],
  //  [0, 1, 2, 3, ..., 99, 0, 1, 2, 3, 99, ...],
  //  [0, 1, 2, 3, ..., 99, 0, 1, 2, 3, 99, ...],
  //  [0, 1, 2, 3, ..., 99, 0, 1, 2, 3, 99, ...],
  // ]
  const int32_t row_dim = 4;
  const int32_t inner_dim = 2048;
  std::vector<std::vector<InputType>> a;
  a.reserve(row_dim);
  for (int row = 0; row < row_dim; ++row) {
    std::vector<InputType> a_row;
    a_row.reserve(inner_dim);
    for (int col = 0; col < inner_dim; ++col) {
      a_row.push_back(col % 100);
    }
    a.push_back(std::move(a_row));
  }
  Params p(a, Identity(inner_dim));

  // Run
  matmul(p.input_a, p.input_b, p.output, p.row_dim, p.inner_dim, p.col_dim);

  // Verify
  EXPECT_THAT(p.GetOutputVector(), ElementsAreArray(Flatten(a)));
}

}  // namespace