# PRIVATE

# matrix_util
add_library(_util_matrix_util INTERFACE)
target_link_libraries(_util_matrix_util INTERFACE _util_to_string)

# to_string
add_library(_util_to_string INTERFACE)
target_link_libraries(_util_to_string INTERFACE absl::log absl::strings)
add_gtest(_util_to_string_test to_string_test.cc _util_to_string)

# util
add_library(_util_util INTERFACE)
add_gtest(_util_util_test util_test.cc _util_util absl::log)

## PUBLIC
add_library(util INTERFACE)

target_link_libraries(
  util INTERFACE

  _util_matrix_util
  _util_to_string
  _util_util
)

