## PUBLIC
add_library(sungcho_util INTERFACE)

target_link_libraries(
  sungcho_util INTERFACE

  util_base_util
  util_matrix_util
  util_to_string
)

# PRIVATE

# base_util
add_library(util_base_util INTERFACE)
add_gtest(util_base_util_test base_util_test.cc util_base_util absl::log)

# matrix_util
add_library(util_matrix_util INTERFACE)
target_link_libraries(util_matrix_util INTERFACE util_to_string)
add_gtest(util_matrix_util_test matrix_util_test.cc util_matrix_util)

# to_string
add_library(util_to_string INTERFACE)
target_link_libraries(util_to_string INTERFACE absl::log absl::strings)
add_gtest(util_to_string_test to_string_test.cc util_to_string)
