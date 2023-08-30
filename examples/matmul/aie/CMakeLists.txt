
set(IREE_PACKAGE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
# Canonicalize path.
cmake_path(ABSOLUTE_PATH IREE_PACKAGE_ROOT_DIR
  BASE_DIRECTORY ${IREE_PACKAGE_ROOT_DIR}
  NORMALIZE
  OUTPUT_VARIABLE IREE_PACKAGE_ROOT_DIR)
set(IREE_PACKAGE_ROOT_PREFIX iree)

add_library(matmul SHARED matmul.cc)

add_library(aie_system_plugin SHARED
  aie_system_plugin.c
)
target_include_directories(aie_system_plugin
  PRIVATE
    ${IREE_SOURCE_DIR}/runtime/src/
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(aie_system_plugin
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "aie_system_plugin"
)

add_dependencies(iree-sample-deps aie_system_plugin)

target_link_libraries(aie_system_plugin matmul)
