#include "util/matrix_util.h"

#include "gtest/gtest.h"

namespace sungcho {
namespace {

TEST(PrintMatrixRowTest, PrintAll) {
  const std::vector<float> vs = {1, 2, 3, 4, 5};
  EXPECT_EQ(PrintMatrixRow(vs.data(), vs.size(), /*print_limit=*/10),
            "[1 2 3 4 5]");
}

TEST(PrintMatrixRowTest, Truncated) {
  const std::vector<float> vs = {1, 2, 3, 4, 5};
  EXPECT_EQ(PrintMatrixRow(vs.data(), vs.size(), /*print_limit=*/3),
            "[1 2 3 ...]");
}

TEST(PrintMatrixTest, PrintAll2x5) {
  const std::vector<float> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(PrintMatrix(vs.data(), {2, 5}, /*print_limit=*/10),
            "[[1 2 3 4 5]\n[6 7 8 9 10]\n]");
}

TEST(PrintMatrixTest, PrintAll5x2) {
  const std::vector<float> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(PrintMatrix(vs.data(), {5, 2}, /*print_limit=*/10),
            "[[1 2]\n[3 4]\n[5 6]\n[7 8]\n[9 10]\n]");
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
