# PRIVATE

# matrix_util
add_library(util_matrix_util INTERFACE)
target_link_libraries(util_matrix_util INTERFACE util_to_string)
add_gtest(util_matrix_util_test matrix_util_test.cc util_matrix_util)

# to_string
add_library(util_to_string INTERFACE)
target_link_libraries(util_to_string INTERFACE absl::log absl::strings)
add_gtest(util_to_string_test to_string_test.cc util_to_string)

# util
add_library(util_util INTERFACE)
add_gtest(util_util_test util_test.cc util_util absl::log)

## PUBLIC
add_library(util INTERFACE)

target_link_libraries(
  util INTERFACE

  util_matrix_util
  util_to_string
  util_util
)

