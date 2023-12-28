add_library(examples_conv_conv1d conv1d.cc)

add_gtest(
  examples_conv_conv1d_test
  conv1d_test.cc
  examples_conv_conv1d absl::log)

