cc_library(
    name = "base_util",
    hdrs = ["base_util.h"],
    deps = [],
)

cc_test(
    name = "base_util_test",
    srcs = ["base_util_test.cc"],
    deps = [
        ":base_util",
        "@absl//absl/log",
        "@absl//absl/strings",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "matrix_util",
    hdrs = ["matrix_util.h"],
    deps = [
      ":to_string",
    ],
)

cc_test(
    name = "matrix_util_test",
    srcs = ["matrix_util_test.cc"],
    deps = [
        ":matrix_util",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "to_string",
    hdrs = ["to_string.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_util",
        "@absl//absl/strings",
        "@absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "to_string_test",
    srcs = ["to_string_test.cc"],
    deps = [
        ":to_string",
        "@googletest//:gtest_main",
    ],
)
