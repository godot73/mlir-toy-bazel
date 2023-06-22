add_library(matmul matmul.cc)

add_gtest(matmul_test matmul_test.cc matmul absl::log)
