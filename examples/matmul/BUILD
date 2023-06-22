
cc_library(
    name = "matmul",
    srcs = ["matmul.cc"],
    hdrs = ["matmul.h"],
)

cc_test(
    name = "matmul_test",
    srcs = ["matmul_test.cc"],
    deps = [
        ":matmul",
        "@absl//absl/log",
        "@googletest//:gtest_main",
    ],
)
