set(IREE_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../third_party/iree)

if (WIN32)
  add_library(matmul matmul.cc)
else()
  add_library(matmul SHARED matmul.cc)
endif()

add_library(system_plugin SHARED system_plugin.c)

target_include_directories(system_plugin
  PRIVATE
    ${IREE_SOURCE_DIR}/runtime/src/
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(system_plugin
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "system_plugin"
)

target_link_libraries(system_plugin matmul)

add_gtest(matmul_test matmul_test.cc matmul absl::log)
