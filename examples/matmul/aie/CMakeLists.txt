
set(IREE_PACKAGE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
# Canonicalize path.
cmake_path(ABSOLUTE_PATH IREE_PACKAGE_ROOT_DIR
  BASE_DIRECTORY ${IREE_PACKAGE_ROOT_DIR}
  NORMALIZE
  OUTPUT_VARIABLE IREE_PACKAGE_ROOT_DIR)
set(IREE_PACKAGE_ROOT_PREFIX iree)

add_dependencies(iree-sample-deps aie_standalone_plugin)

add_library(matmul SHARED matmul.cc)
# standalone_plugin_library()

# Creates the architecture-specific part of the standalone plugin, for the given
# architecture (as in IREE_ARCH), and appends the generated library filename to
# the output-variable _DST_VAR_ARCH_LIBRARIES. Skips if the compiler does not
# support the specified architecture.
function(aie_standalone_plugin_library _DST_VAR_ARCH_LIBRARIES _ARCH)
  iree_compiler_targeting_iree_arch(_ENABLED "${_ARCH}")
  if(NOT _ENABLED)
    return()
  endif()
  set(_NAME aie_standalone_plugin_${_ARCH})
  iree_arch_to_llvm_arch(LLVM_ARCH "${_ARCH}")
  add_custom_command(
    OUTPUT
      aie_standalone_plugin_${_ARCH}.o
    DEPENDS
      aie_standalone_plugin.c
      ${IREE_CLANG_TARGET}
    COMMAND ${IREE_CLANG_BINARY}
      -target ${LLVM_ARCH}-none-elf
      -isystem ${IREE_CLANG_BUILTIN_HEADERS_PATH}
      -std=c17
      -fPIC
      -ffreestanding
      -fvisibility=hidden
      -fno-plt
      -fno-rtti
      -fno-exceptions
      -fdata-sections
      -ffunction-sections
      -funique-section-names
      -I ${IREE_SOURCE_DIR}/runtime/src/
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      -c ${CMAKE_CURRENT_SOURCE_DIR}/aie_standalone_plugin.c
      -o ${CMAKE_CURRENT_BINARY_DIR}/aie_standalone_plugin_${_ARCH}.o
    VERBATIM
  )
  set(_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/aie_standalone_plugin_${_ARCH}.so")
  add_custom_command(
    OUTPUT
      aie_standalone_plugin_${_ARCH}.so
    DEPENDS
      aie_standalone_plugin_${_ARCH}.o
      matmul.cc
      ${IREE_LLD_TARGET}
    COMMAND ${IREE_LLD_BINARY}
      -flavor gnu
      --build-id=none
      -nostdlib
      -static
      -shared
      --no-undefined
      --no-allow-shlib-undefined
      --allow-multiple-definition
      --gc-sections
      -z now
      -z relro
      --discard-all
      --icf=all
      --ignore-data-address-equality
      --ignore-function-address-equality
      --hash-style=sysv
      --strip-debug
      ${CMAKE_CURRENT_BINARY_DIR}/aie_standalone_plugin_${_ARCH}.o
      -o "${_LIBRARY}"
    VERBATIM
  )
  add_custom_target(${_NAME} DEPENDS
    "${_LIBRARY}"
  )
  set(_LOCAL_ARCH_LIBRARIES "${${_DST_VAR_ARCH_LIBRARIES}}")
  list(APPEND _LOCAL_ARCH_LIBRARIES "${_LIBRARY}")
  set("${_DST_VAR_ARCH_LIBRARIES}" "${_LOCAL_ARCH_LIBRARIES}" PARENT_SCOPE)
endfunction()

# Build the standalone_plugin_*.so files for each architecture we target.
set(_STANDALONE_PLUGIN_ARCH_LIBRARIES "")

aie_standalone_plugin_library(_STANDALONE_PLUGIN_ARCH_LIBRARIES arm_64)
aie_standalone_plugin_library(_STANDALONE_PLUGIN_ARCH_LIBRARIES x86_64)

add_custom_command(
  OUTPUT
    aie_standalone_plugin.sos
  DEPENDS
    ${_STANDALONE_PLUGIN_ARCH_LIBRARIES}
    iree-fatelf
  COMMAND iree-fatelf join
    ${_STANDALONE_PLUGIN_ARCH_LIBRARIES}
    > ${CMAKE_CURRENT_BINARY_DIR}/aie_standalone_plugin.sos
  VERBATIM
)
add_custom_target(aie_standalone_plugin DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/aie_standalone_plugin.sos
)
