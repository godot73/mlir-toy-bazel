#include "util/matrix_util.h"

#include "gtest/gtest.h"

namespace sungcho {
namespace {

TEST(PrintVectorTest, PrintAll) {
  const std::vector<float> vs = {1, 2, 3, 4, 5};
  const std::string expected = "[1 2 3 4 5]";
  EXPECT_EQ(PrintVector(vs, /*print_limit=*/10), expected);
  EXPECT_EQ(PrintVector(vs), expected);
  EXPECT_EQ(PrintVector(vs.data(), vs.size(), /*print_limit=*/10), expected);
  EXPECT_EQ(PrintVector(vs.data(), vs.size()), expected);
}

TEST(PrintVectorTest, Truncated) {
  const std::vector<float> vs = {1, 2, 3, 4, 5};
  const std::string expected = "[1 2 3 ...]";
  EXPECT_EQ(PrintVector(vs, /*print_limit=*/3), expected);
  EXPECT_EQ(PrintVector(vs.data(), vs.size(), /*print_limit=*/3), expected);
}

TEST(PrintMatrixTest, PrintAll2x5) {
  const std::vector<float> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::string expected = "[[1 2 3 4 5]\n[6 7 8 9 10]\n]";
  EXPECT_EQ(PrintMatrix(vs.data(), {2, 5}, /*print_limit=*/10), expected);
  EXPECT_EQ(PrintMatrix(vs.data(), {2, 5}), expected);
}

TEST(PrintMatrixTest, PrintAll5x2) {
  const std::vector<float> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::string expected = "[[1 2]\n[3 4]\n[5 6]\n[7 8]\n[9 10]\n]";
  EXPECT_EQ(PrintMatrix(vs.data(), {5, 2}, /*print_limit=*/10), expected);
  EXPECT_EQ(PrintMatrix(vs.data(), {5, 2}), expected);
}

TEST(PrintMatrixTest, PrintColTruncated) {
  const std::vector<float> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(PrintMatrix(vs.data(), {2, 5}, /*print_limit=*/3),
            "[[1 2 3 ...]\n[6 7 8 ...]\n]");
}

TEST(PrintMatrixTest, PrintRowTruncated) {
  const std::vector<float> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(PrintMatrix(vs.data(), {5, 2}, /*print_limit=*/3),
            "[[1 2]\n[3 4]\n[5 6]\n...\n]");
}

}  // namespace
}  // namespace sungcho
