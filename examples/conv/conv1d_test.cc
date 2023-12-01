#include "examples/conv/conv1d.h"

#include "absl/log/log.h"
#include "gtest/gtest.h"

namespace sungcho {
namespace {

int count_output_size(int input_size, int kernel_size, int stride) {
  int output_size = 0, input_pos = 0, output_pos = 0;
  for (int input_pos = 0; input_pos + kernel_size - 1 < input_size;
       input_pos += stride) {
    ++output_size;
  }
  return output_size;
}

TEST(Conv1DTest, Stride1) {
  EXPECT_EQ(get_output_size(/*input_size=*/5, /*kernel_size=*/3, /*stride=*/1),
            3);
  EXPECT_EQ(get_output_size(/*input_size=*/50, /*kernel_size=*/3, /*stride=*/1),
            48);
}

TEST(Conv1DTest, Stride2) {
  EXPECT_EQ(get_output_size(/*input_size=*/5, /*kernel_size=*/3, /*stride=*/2),
            2);

  EXPECT_EQ(get_output_size(/*input_size=*/50, /*kernel_size=*/3, /*stride=*/2),
            24);
  EXPECT_EQ(get_output_size(/*input_size=*/50, /*kernel_size=*/4, /*stride=*/2),

            24);
  EXPECT_EQ(get_output_size(/*input_size=*/50, /*kernel_size=*/5, /*stride=*/2),
            23);
}

TEST(Conv1DTest, RatherExhaustive) {
  const int input_size = 100;
  for (int kernel_size = 2; kernel_size < 10; ++kernel_size) {
    for (int stride = 1; stride < 5; ++stride) {
      EXPECT_EQ(get_output_size(input_size, kernel_size, stride),
                count_output_size(input_size, kernel_size, stride));
    }
  }
}

}  // namespace
}  // namespace sungcho
