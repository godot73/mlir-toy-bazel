# This setup works with bazel verison 4.2.0.

workspace(name = "com_jaeyoonyu_scratch")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# bazel-skylib
git_repository(
  name = "bazel_skylib",
  remote = "https://github.com/bazelbuild/bazel-skylib",
  branch = "main",
)

# abseil-cpp
git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp",
    branch = "lts_2023_01_25",
)

# Google Test
git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest",
    branch = "v1.10.x",
)

# C++ rules for Bazel.
http_archive(
    name = "rules_cc",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/9e10b8a6db775b1ecd358d8ddd3dab379a2c29a5.zip"],
    strip_prefix = "rules_cc-9e10b8a6db775b1ecd358d8ddd3dab379a2c29a5",
    sha256 = "954b7a3efc8752da957ae193a13b9133da227bdacf5ceb111f2e11264f7e8c95",
)

# llvm project
new_local_repository(
    name = "llvm-raw",
    build_file_content = "# empty",
    # Or wherever your submodule is located.
    path = "third_party/llvm-project",
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

llvm_configure(name = "llvm-project")

# Disables optional dependencies for Support like zlib and terminfo. You may
# instead want to configure them using the macros in the corresponding bzl
# files.
llvm_disable_optional_support_deps()

