cmake_minimum_required(VERSION 3.14)
project(mlir_toy_bazel)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  abseil
  URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.3.zip
)
FetchContent_MakeAvailable(abseil)

enable_testing()

include(GoogleTest)

function(add_gtest test_name test_filename)
  add_executable(${test_name} ${test_filename})
  gtest_discover_tests(${test_name})
  target_link_libraries(${test_name} GTest::gmock_main ${ARGN})
endfunction()

# Throughout the whole project, the fully qualified directory is allowed for
# #include "...".
include_directories(${CMAKE_CURRENT_SOURCE_DIR})



#set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

add_subdirectory(examples)
