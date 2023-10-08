set(IREE_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../third_party/iree)

if (WIN32)
  add_library(examples_matmul_matmul matmul.cc)
else()
  add_library(examples_matmul_matmul SHARED matmul.cc)
endif()

add_library(examples_matmul_system_plugin SHARED system_plugin.c)

target_include_directories(examples_matmul_system_plugin
  PRIVATE
    ${IREE_SOURCE_DIR}/runtime/src/
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(examples_matmul_system_plugin
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "system_plugin"
)

target_link_libraries(examples_matmul_system_plugin examples_matmul_matmul)

add_gtest(
  examples_matmul_matmul_test
  matmul_test.cc
  examples_matmul_matmul absl::log)
