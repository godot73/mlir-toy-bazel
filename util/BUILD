cc_library(
    name = "to_string",
    hdrs = ["to_string.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":util",
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

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "@absl//absl/log",
        "@absl//absl/strings",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "matrix_util",
    srcs = ["matrix_util.cc"],
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
