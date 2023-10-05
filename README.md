# mlir-toy-bazel

## Build and Run C++ Tests.

### Using `cmake`

- Install `cmake` and `ccache`.
- Run `cmake`.

  ```
  cmake -B build/ -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  cmake --build build/
  ctest --test-dir build/ *
  ```

### Using `bazel`

- Install [`bazel`](https://github.com/bazelbuild/bazelisk).
  - For convenience, add a symlink (in PATH) or an alias `bazel` pointing at `bazelisk`.
  - If necessary, adjust the attribute/mod so we could run it. Check by running `bazel --version`.
  - On Linux, add `export BAZEL_CXXOPTS=-std=c++20` to `~/.bashrc`.
- Run desired tests by running `bazel test [...]`. For example:

  ```
  bazel test examples/matmul:all [--test_output=streamed]
  ```
