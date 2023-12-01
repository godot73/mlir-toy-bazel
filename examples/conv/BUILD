load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "conv1d",
    srcs = ["conv1d.cc"],
    hdrs = ["conv1d.h"],
)

cc_test(
    name = "conv1d_test",
    size = "small",
    srcs = ["conv1d_test.cc"],
    deps = [
        ":conv1d",
        "@absl//absl/log",
        "@googletest//:gtest_main",
    ],
)

