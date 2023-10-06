cmake_minimum_required(VERSION 3.14)
project(mlir_toy_bazel)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.13.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Abseil
add_subdirectory(third_party/abseil-cpp)

enable_testing()

include(GoogleTest)

function(add_gtest test_name test_filename)
  add_executable(${test_name} ${test_filename})
  gtest_discover_tests(${test_name})
  target_link_libraries(${test_name} GTest::gmock_main ${ARGN})
endfunction()

# This project allows fully qualified directories for #include "...".
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(examples)
add_subdirectory(util)
